#!/usr/bin/env python3
"""
inference.py — Baseline inference script for Disaster Response OpenEnv

Mandatory stdout format (must match exactly):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.0000> rewards=<r1,r2,...,rn>

Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN (also accepts OPENAI_API_KEY)
"""

import os
import sys
import json
import time
import traceback
import threading
from typing import Dict, List, Optional

# Force unbuffered stdout — critical for validator to capture output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from openai import OpenAI
import requests
import uvicorn

# ── Constants ──────────────────────────────────────────────────────────────────

TASKS = ["task_1_earthquake", "task_2_flood", "task_3_multi_disaster"]
BENCHMARK = "disaster-response-env"
SEED = 42
MAX_STEPS = 50  # safety cap — well within 20 min limit
ENV_PORT = 7860
SUCCESS_THRESHOLD = 0.60  # grader score >= this → success=true


# ── Mandatory log helpers ──────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()  # "true" or "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # score clamped strictly in (0,1)
    score = max(0.001, min(0.999, score))
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ── Server bootstrap ───────────────────────────────────────────────────────────


def _start_server() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.app import app as fastapi_app

    uvicorn.run(fastapi_app, host="0.0.0.0", port=ENV_PORT, log_level="error")


def ensure_server_running() -> None:
    """Start the FastAPI env server in a background thread if not already up."""
    try:
        requests.get(f"http://localhost:{ENV_PORT}/health", timeout=2)
        return  # already running (Docker entrypoint started it)
    except Exception:
        pass

    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    for _ in range(30):  # wait up to 15 s
        time.sleep(0.5)
        try:
            r = requests.get(f"http://localhost:{ENV_PORT}/health", timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass

    raise RuntimeError("Environment server failed to start within 15 seconds")


# ── Client factory ─────────────────────────────────────────────────────────────


def make_client():
    api_base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1").strip()
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
    api_key = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("Missing API key — set HF_TOKEN or OPENAI_API_KEY")
    return OpenAI(base_url=api_base_url, api_key=api_key), model_name


# ── Environment HTTP wrappers ──────────────────────────────────────────────────

BASE = f"http://localhost:{ENV_PORT}"


def env_reset(task_id: str, seed: int = SEED):
    r = requests.post(
        f"{BASE}/reset", json={"task_id": task_id, "seed": seed}, timeout=30
    )
    r.raise_for_status()
    data = r.json()
    # ISSUE 2 fix — read session_id from body (most reliable after BUG 1 fix)
    session_id = data.get("session_id") or r.headers.get("X-Session-Id")
    if not session_id:
        raise RuntimeError("No session_id returned from /reset")
    return data["observation"], session_id


def env_step(session_id: str, action: Dict, retries: int = 3) -> Dict:
    # ISSUE 6 fix — retry with exponential backoff on transient errors
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{BASE}/step",
                json={"session_id": session_id, "action": action},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def env_grade(session_id: str) -> float:
    r = requests.get(f"{BASE}/grade/{session_id}", timeout=10)
    r.raise_for_status()
    score = r.json().get("grader_score", 0.001)
    # Ensure strictly in (0, 1) — validator requirement
    return max(0.001, min(0.999, float(score)))


# ── Agent ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an emergency response coordinator AI.
Allocate rescue resources across disaster zones to maximize lives saved.

AVAILABLE ACTION TYPES:
- allocate_search_rescue : Deploy S&R teams to a zone
- allocate_medical       : Deploy medical teams to a zone
- allocate_firefighting  : Deploy firefighting units (fire/multi zones only)
- allocate_water_rescue  : Deploy water rescue teams (flood/multi zones only)
- evacuate_zone          : Evacuate a zone (only when trapped_casualties < 10)
- prioritize_zone        : Mark a zone as priority
- standby               : Do nothing (penalized)

RULES:
1. Prioritize CRITICAL and HIGH severity zones first
2. Match resource type to disaster type (flood->water_rescue, fire->firefighting)
3. Never leave CRITICAL zones unattended — casualties increase each step
4. Allocate enough units to make meaningful progress

Respond ONLY with valid JSON:
{
  "action_type": "<action_type>",
  "zone_id": "<zone_id>",
  "units": <integer 1-10>,
  "reasoning": "<brief explanation>"
}"""


def build_user_prompt(obs: Dict) -> str:
    zones_summary = []
    for z in obs["zones"]:
        if z["severity"] == "none":
            continue
        zones_summary.append(
            f"  {z['zone_id']} [{z['name']}] "
            f"type={z['disaster_type']} sev={z['severity']} "
            f"trapped={z['trapped_casualties']} injured={z['injured']} "
            f"rescued={z['rescued']} fatalities={z['fatalities']} "
            f"unattended={z['turns_unattended']} access={z['accessibility']:.2f}"
        )
    res = obs["available_resources"]
    return (
        f"DISASTER: {obs['disaster_scenario']} ({obs['disaster_type']})\n"
        f"STEP: {obs['time_step']} / {obs['max_steps']}\n"
        f"FEEDBACK: {obs.get('last_action_feedback', '')}\n\n"
        f"RESOURCES: SR={res['search_rescue_teams']} Med={res['medical_teams']} "
        f"Fire={res['firefighting_units']} Water={res['water_rescue_teams']} "
        f"Evac={res['evacuation_vehicles']}\n\n"
        f"ZONES:\n"
        + ("\n".join(zones_summary) if zones_summary else "  All zones stable.")
        + f"\n\nRESCUED: {obs['total_rescued']}  FATALITIES: {obs['total_fatalities']}\n\n"
        "Choose ONE action. Output ONLY valid JSON."
    )


def agent_select_action(
    obs: Dict, history: List[Dict], client: OpenAI, model_name: str
) -> Dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-2:]:
        messages.append({"role": "assistant", "content": json.dumps(h["action"])})
        messages.append({"role": "user", "content": f"Result: {h['feedback']}"})
    messages.append({"role": "user", "content": build_user_prompt(obs)})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or ""
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                action = json.loads(raw[start:end])
                assert "action_type" in action and "zone_id" in action
                action.setdefault("units", 1)
                action.setdefault("reasoning", "")
                return action
        except Exception:
            if attempt < 2:
                time.sleep(1)

    # Always return a safe fallback — never None
    active = [
        z for z in obs.get("zones", []) if z.get("severity") not in ("none", "low")
    ]
    zone_id = (
        active[0]["zone_id"]
        if active
        else (obs["zones"][0]["zone_id"] if obs.get("zones") else "Z1")
    )
    return {
        "action_type": "allocate_search_rescue",
        "zone_id": zone_id,
        "units": 2,
        "reasoning": "fallback: LLM unavailable or parse failed",
    }


# ── Task runner ────────────────────────────────────────────────────────────────


def run_task(task_id: str, client: OpenAI, model_name: str) -> Dict:
    # ── [START] ───────────────────────────────────────────────────────────────
    log_start(task=task_id, env=BENCHMARK, model=model_name)

    obs, session_id = env_reset(task_id, seed=SEED)

    step_num = 0
    done = False
    rewards: List[float] = []
    last_error: Optional[str] = None
    history: List[Dict] = []
    step_resp: Dict = {}  # guard: defined before loop

    while not done and step_num < MAX_STEPS:
        try:
            action = agent_select_action(obs, history, client, model_name)
            step_resp = env_step(session_id, action)

            reward_val = float(step_resp["reward"]["total"])
            done = step_resp["done"]
            feedback = step_resp["observation"].get("last_action_feedback", "")
            last_error = None
            step_num += 1
            rewards.append(reward_val)

            # action string: compact representation for the log
            action_str = f"{action['action_type']}(zone={action['zone_id']},units={action.get('units',1)})"

            # ── [STEP] ────────────────────────────────────────────────────────
            log_step(
                step=step_num,
                action=action_str,
                reward=reward_val,
                done=done,
                error=last_error,
            )

            history.append({"action": action, "feedback": feedback})
            obs = step_resp["observation"]

        except Exception as exc:
            last_error = str(exc)
            step_num += 1
            rewards.append(0.0)
            log_step(
                step=step_num, action="null", reward=0.0, done=False, error=last_error
            )
            if step_num >= MAX_STEPS:
                break

    # ── Score & [END] ─────────────────────────────────────────────────────────
    try:
        final_score = env_grade(session_id)
    except Exception:
        final_score = 0.001  # never exactly 0.0

    success = final_score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=step_num, score=final_score, rewards=rewards)

    return {
        "task_id": task_id,
        "grader_score": final_score,
        "grade": step_resp.get("info", {}).get("grade", "F"),
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    # Boot the environment server if not already running
    try:
        ensure_server_running()
    except Exception as e:
        print(f"[ERROR] Could not start environment server: {e}", flush=True)
        sys.exit(0)

    try:
        client, model_name = make_client()
    except RuntimeError as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(0)

    all_results = []
    for task_id in TASKS:
        try:
            result = run_task(task_id, client, model_name)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] task={task_id} error={e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            # Always emit [END] even on exception
            log_end(success=False, steps=0, score=0.001, rewards=[])

    avg = (
        sum(r["grader_score"] for r in all_results) / len(all_results)
        if all_results
        else 0.0
    )
    print(f"[SUMMARY] tasks={len(all_results)} avg_score={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
