#!/usr/bin/env python3
"""
validate.py — Pre-submission validation for Disaster Response OpenEnv

Run this before deploying to confirm everything works:
    python validate.py

All tests must pass before submitting.
"""

import sys
import copy
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_files_exist():
    """All required files are present."""
    required = [
        "inference.py",
        "openenv.yaml",
        "Dockerfile",
        "requirements.txt",
        "README.md",
        "baseline_results.json",
        "app/__init__.py",
        "app/models.py",
        "app/tasks.py",
        "app/environment.py",
        "server/__init__.py",
        "server/app.py",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        logger.error("Missing files: %s", ", ".join(missing))
        return False
    logger.info("[OK] All required files present")
    return True


def test_openenv_yaml():
    """openenv.yaml is valid and has all required fields."""
    try:
        import yaml

        doc = yaml.safe_load(Path("openenv.yaml").read_text())
        required = [
            "name",
            "version",
            "description",
            "tasks",
            "observation_space",
            "action_space",
        ]
        missing = [f for f in required if f not in doc]
        if missing:
            logger.error("openenv.yaml missing fields: %s", ", ".join(missing))
            return False
        if len(doc.get("tasks", [])) < 3:
            logger.error("openenv.yaml must define at least 3 tasks")
            return False
        for t in doc["tasks"]:
            for field in ["id", "name", "difficulty", "score_range"]:
                if field not in t:
                    logger.error("Task '%s' missing field: %s", t.get("id", "?"), field)
                    return False
        logger.info("[OK] openenv.yaml valid (%d tasks)", len(doc["tasks"]))
        return True
    except Exception as e:
        logger.error("openenv.yaml validation failed: %s", e)
        return False


def test_imports():
    """All modules import without error."""
    try:
        from app.models import (
            Observation,
            Action,
            Reward,
            StepResponse,
            StateResponse,
            ZoneState,
            ResourcePool,
            DisasterType,
            ZoneSeverity,
        )
        from app.tasks import TASKS, grade_task
        from app.environment import DisasterResponseEnv
        from openai import OpenAI

        logger.info("[OK] All imports successful")
        return True
    except ImportError as e:
        logger.error("Import failed: %s", e)
        return False


def test_env_init():
    """All 3 task environments initialise cleanly."""
    try:
        from app.environment import DisasterResponseEnv

        for tid in ["task_1_earthquake", "task_2_flood", "task_3_multi_disaster"]:
            env = DisasterResponseEnv(task_id=tid, seed=42)
            logger.info("  [OK] DisasterResponseEnv(%s)", tid)
        return True
    except Exception as e:
        logger.error("Environment init failed: %s", e)
        return False


def test_reset():
    """reset() returns a clean Observation."""
    try:
        from app.environment import DisasterResponseEnv
        from app.models import Observation

        env = DisasterResponseEnv("task_1_earthquake", seed=42)
        obs = env.reset()
        assert isinstance(obs, Observation), "reset() must return Observation"
        assert obs.time_step == 0, "time_step must be 0 after reset"
        assert obs.total_rescued == 0, "total_rescued must be 0 after reset"
        assert obs.done is False, "done must be False after reset"
        assert len(obs.zones) == 3, "task_1 must have 3 zones"
        logger.info("[OK] reset() returns clean Observation")
        return True
    except Exception as e:
        logger.error("reset() failed: %s", e)
        return False


def test_step():
    """step() returns StepResponse with reward in [-1, 1]."""
    try:
        from app.environment import DisasterResponseEnv
        from app.models import Action, StepResponse

        env = DisasterResponseEnv("task_1_earthquake", seed=42)
        env.reset()
        action = Action(action_type="allocate_search_rescue", zone_id="Z1", units=4)
        resp = env.step(action)
        assert isinstance(resp, StepResponse), "step() must return StepResponse"
        assert isinstance(resp.done, bool), "done must be bool"
        assert -1.0 <= resp.reward.total <= 1.0, "reward.total must be in [-1, 1]"
        assert resp.observation.time_step == 1, "time_step must advance"
        logger.info("[OK] step() returns StepResponse, reward=%.4f", resp.reward.total)
        return True
    except Exception as e:
        logger.error("step() failed: %s", e)
        return False


def test_state():
    """state() returns StateResponse with grader_score in [0, 1]."""
    try:
        from app.environment import DisasterResponseEnv
        from app.models import Action, StateResponse

        env = DisasterResponseEnv("task_1_earthquake", seed=42)
        env.reset()
        env.step(Action(action_type="allocate_search_rescue", zone_id="Z1", units=4))
        st = env.state()
        assert isinstance(st, StateResponse), "state() must return StateResponse"
        assert 0.0 <= st.grader_score <= 1.0, "grader_score must be in [0, 1]"
        assert "task_id" in st.task_metadata, "task_metadata must have task_id"
        logger.info(
            "[OK] state() returns StateResponse, grader_score=%.4f", st.grader_score
        )
        return True
    except Exception as e:
        logger.error("state() failed: %s", e)
        return False


def test_graders():
    """All 3 graders return deterministic scores in [0, 1]."""
    try:
        from app.environment import DisasterResponseEnv
        from app.models import Action
        from app.tasks import grade_task, TASKS

        for tid in TASKS:
            # Run twice with same seed — must be identical
            scores = []
            for _ in range(2):
                env = DisasterResponseEnv(tid, seed=42)
                env.reset()
                env.step(
                    Action(
                        action_type="allocate_search_rescue",
                        zone_id=TASKS[tid].zones[0].zone_id,
                        units=3,
                    )
                )
                s = grade_task(
                    tid,
                    {
                        "zones": copy.deepcopy(env._zones),
                        "resources_used": env._resources_used,
                        "resources_total": env._resources_total,
                    },
                )
                assert 0.0 <= s <= 1.0, "score must be in [0, 1]"
                scores.append(s)
            assert scores[0] == scores[1], "grader must be deterministic"
            logger.info("  [OK] %s: score=%.4f (deterministic)", tid, scores[0])
        return True
    except Exception as e:
        logger.error("Grader test failed: %s", e)
        return False


def test_server_endpoints():
    """FastAPI server responds correctly to all required endpoints."""
    try:
        from fastapi.testclient import TestClient
        from server.app import app

        client = TestClient(app)

        # /health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

        # /tasks
        r = client.get("/tasks")
        assert r.status_code == 200
        assert len(r.json()) == 3

        # /reset with empty body (what the validator sends)
        r = client.post("/reset", json={})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert "observation" in data
        sid = data["session_id"]

        # /step
        r = client.post(
            "/step",
            json={
                "session_id": sid,
                "action": {
                    "action_type": "allocate_search_rescue",
                    "zone_id": "Z1",
                    "units": 3,
                },
            },
        )
        assert r.status_code == 200
        sd = r.json()
        assert all(k in sd for k in ["observation", "reward", "done", "info"])
        assert -1.0 <= sd["reward"]["total"] <= 1.0

        # /state
        r = client.get(f"/state/{sid}")
        assert r.status_code == 200
        assert "grader_score" in r.json()

        # /grade
        r = client.get(f"/grade/{sid}")
        assert r.status_code == 200
        assert "grader_score" in r.json()

        logger.info("[OK] All server endpoints respond correctly")
        return True
    except Exception as e:
        logger.error("Server endpoint test failed: %s", e)
        return False


def test_inference_format():
    """inference.py emits the mandatory [START]/[STEP]/[END] format."""
    try:
        src = Path("inference.py").read_text()
        checks = {
            "[START] task= env= model=": "[START] task={task} env={env} model={model}"
            in src,
            "[STEP] step= action= reward= done= error=": "[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}"
            in src,
            "[END] success= steps= score= rewards=": "[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}"
            in src,
            "reward is 2dp": "reward:.2f" in src,
            "done is lowercase bool": "str(done).lower()" in src,
            "success is lowercase bool": "str(success).lower()" in src,
            "rewards is comma list": '",".join(f"{r:.2f}" for r in rewards)' in src,
            "HF_TOKEN supported": "HF_TOKEN" in src,
            "OPENAI_API_KEY supported": "OPENAI_API_KEY" in src,
            "server self-start": "ensure_server_running" in src,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            for f in failed:
                logger.error("  inference.py missing: %s", f)
            return False
        logger.info("[OK] inference.py log format is compliant")
        return True
    except Exception as e:
        logger.error("inference.py format check failed: %s", e)
        return False


def main():
    print("=" * 65)
    print("  DISASTER RESPONSE OPENENV — PRE-SUBMISSION VALIDATION")
    print("=" * 65)
    print()

    tests = [
        ("File Structure", test_files_exist),
        ("openenv.yaml", test_openenv_yaml),
        ("Imports", test_imports),
        ("Environment Init", test_env_init),
        ("reset()", test_reset),
        ("step()", test_step),
        ("state()", test_state),
        ("Graders (3 tasks)", test_graders),
        ("Server Endpoints", test_server_endpoints),
        ("inference.py Format", test_inference_format),
    ]

    results = []
    for name, fn in tests:
        result = fn()
        results.append((name, result))
        print()

    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print("  [%s] %s" % ("PASS" if result else "FAIL", name))
    print("-" * 65)
    print("  %d / %d tests passed" % (passed, len(results)))
    print()

    if passed == len(results):
        print("  All tests passed. Ready for submission.")
        return 0
    else:
        print("  Fix failing tests before submitting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
