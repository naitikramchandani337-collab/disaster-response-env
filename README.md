---
title: Disaster Response Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - openenv-environment
  - disaster-response
  - emergency-management
  - resource-allocation
  - real-world
---

# 🚨 Disaster Response System — OpenEnv Environment

> An OpenEnv-compliant simulation of real-world emergency disaster response.
> Agents act as incident commanders allocating scarce rescue resources across
> disaster zones — exactly as FEMA and emergency management agencies do in
> real crises.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://openenv.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)

---

## 🌍 Why This Environment?

Disaster response is one of the highest-stakes real-world decision problems humans face. Every year, natural disasters kill over **60,000 people** globally. The difference between life and death often comes down to how quickly and accurately rescue resources are allocated in the first hours.

This environment directly models that challenge:

- **Resources are always scarce** — teams, vehicles, and equipment must be triaged across competing zones simultaneously
- **Wrong sequencing causes fatalities** — CRITICAL zones that go unattended for even one time step suffer accelerating casualty rates
- **Hazards compound** — unattended fire zones escalate in severity; flood zones lose accessibility as water levels rise each turn
- **Multi-domain reasoning required** — earthquakes need S&R teams, floods need water rescue, fires need firefighting units — a wrong deployment wastes irreplaceable resources

**Who would use this?** Emergency management agencies, FEMA training programs, humanitarian AI research labs, and RL practitioners building agents for high-stakes operational domains.

---

## 🎯 Tasks

Three tasks of increasing difficulty, each with a deterministic grader returning a score in `[0.0, 1.0]`.

| Task ID | Name | Difficulty | Zones | Steps | Disaster Type |
|---------|------|-----------|-------|-------|---------------|
| `task_1_earthquake` | Urban Earthquake Response | 🟢 Easy | 3 | 5 | Earthquake |
| `task_2_flood` | Regional Flash Flood Response | 🟡 Medium | 5 | 8 | Flood |
| `task_3_multi_disaster` | Compound Multi-Disaster | 🔴 Hard | 8 | 12 | Earthquake + Fire + Flood |

---

### Task 1 — Urban Earthquake Response 🟢 Easy

**Scenario**: A magnitude-7.2 earthquake strikes a city. Three districts have collapsed buildings with trapped survivors. Resources are sufficient for a competent agent. The agent must read severity levels, triage correctly, and allocate S&R + medical teams before the time window closes.

| Property | Value |
|----------|-------|
| `task_id` | `task_1_earthquake` |
| Zones | Z1 Downtown Core (CRITICAL), Z2 Residential North (HIGH), Z3 Industrial South (MEDIUM) |
| Resources | 10 S&R, 8 Medical, 2 Fire, 0 Water, 5 Evac |
| Max Steps | 5 |
| Target Score | ≥ 0.70 |

**Grader weights** (actual code):

| Component | Weight |
|-----------|--------|
| Rescue score (normalized to achievable) | 45% |
| Coverage of critical/high zones | 35% |
| Fatality avoidance | 15% |
| Resource efficiency | 5% |

---

### Task 2 — Regional Flash Flood Response 🟡 Medium

**Scenario**: A catastrophic flash flood inundates 5 zones with widely varying accessibility (0.3–0.85). Water rescue assets are critically limited. Zones that are not reached in time lose accessibility as water levels rise. The agent must triage, sequence allocations, and coordinate evacuations.

| Property | Value |
|----------|-------|
| `task_id` | `task_2_flood` |
| Zones | Z1 Riverside Village (CRITICAL), Z2 Lowland Farms (HIGH), Z3 City Underpass (CRITICAL), Z4 Suburb East (MEDIUM), Z5 Mountain Pass Road (LOW) |
| Resources | 5 S&R, 4 Medical, 0 Fire, 6 Water, 3 Evac |
| Max Steps | 8 |
| Target Score | ≥ 0.65 |

**Grader weights** (actual code):

| Component | Weight |
|-----------|--------|
| Rescue score (normalized to achievable) | 40% |
| Coverage of critical/high zones | 30% |
| Fatality avoidance | 15% |
| Resource efficiency | 10% |
| Evacuation success | 5% |

---

### Task 3 — Compound Multi-Disaster 🔴 Hard

**Scenario**: A 6.8 earthquake triggers gas-line wildfires in industrial areas and ruptures a dam causing downstream flooding — all simultaneously. Eight zones with three different disaster types require specialized assets that cannot be substituted. Zone accessibility degrades each turn without attention. Secondary hazards worsen unattended zones.

| Property | Value |
|----------|-------|
| `task_id` | `task_3_multi_disaster` |
| Zones | Z1 Downtown Collapse (EQ/CRITICAL), Z2 Gas District Fire (FIRE/CRITICAL), Z3 Flood Basin A (FLOOD/HIGH), Z4 School District (EQ/HIGH), Z5 Chemical Plant (FIRE/CRITICAL), Z6 Flood Basin B (FLOOD/MEDIUM), Z7 Highway Collapse (EQ/MEDIUM), Z8 Hospital District (EQ/HIGH) |
| Resources | 8 S&R, 6 Medical, 4 Fire, 3 Water, 4 Evac |
| Max Steps | 12 |
| Target Score | ≥ 0.60 |

**Grader weights** (actual code):

| Component | Weight |
|-----------|--------|
| Rescue score (normalized to achievable) | 35% |
| Coverage of critical/high zones | 30% |
| Fatality avoidance | 15% |
| Resource efficiency | 10% |
| Evacuation success | 10% |

---

## 🎮 Action Space

Schema: `app.models.Action` (Pydantic)

```json
{
  "action_type": "allocate_search_rescue",
  "zone_id": "Z1",
  "units": 4,
  "reasoning": "Z1 is CRITICAL with 80 trapped and 0.6 accessibility"
}
```

| Action | Description | Resource Used |
|--------|-------------|---------------|
| `allocate_search_rescue` | Deploy S&R teams to a zone | `search_rescue_teams` |
| `allocate_medical` | Deploy medical teams to a zone | `medical_teams` |
| `allocate_firefighting` | Deploy fire units (fire/multi zones only) | `firefighting_units` |
| `allocate_water_rescue` | Deploy water rescue (flood/multi zones only) | `water_rescue_teams` |
| `evacuate_zone` | Evacuate zone (only when `trapped_casualties < 10`) | `evacuation_vehicles` |
| `prioritize_zone` | Mark zone as priority (no resource cost) | — |
| `standby` | No action (penalized) | — |

> One action per step. Teams are re-allocated each step — unused teams return to the pool.

---

## 👁️ Observation Space

Schema: `app.models.Observation` (Pydantic)

```json
{
  "task_id": "task_2_flood",
  "disaster_scenario": "Regional Flash Flood Response",
  "disaster_type": "flood",
  "time_step": 3,
  "max_steps": 8,
  "zones": [
    {
      "zone_id": "Z1",
      "name": "Riverside Village",
      "disaster_type": "flood",
      "severity": "critical",
      "population": 3000,
      "trapped_casualties": 52,
      "injured": 130,
      "rescued": 38,
      "fatalities": 4,
      "search_rescue_teams": 0,
      "medical_teams": 0,
      "firefighting_units": 0,
      "water_rescue_teams": 0,
      "is_evacuated": false,
      "is_prioritized": true,
      "turns_unattended": 1,
      "accessibility": 0.45
    }
  ],
  "available_resources": {
    "search_rescue_teams": 3,
    "medical_teams": 2,
    "firefighting_units": 0,
    "water_rescue_teams": 4,
    "evacuation_vehicles": 2
  },
  "total_rescued": 38,
  "total_fatalities": 4,
  "total_injured_treated": 20,
  "cumulative_reward": 0.184,
  "active_disaster_zones": ["Z1", "Z2", "Z3"],
  "prioritized_zones": ["Z1"],
  "last_action_feedback": "Deployed 2 water rescue teams to Riverside Village.",
  "done": false
}
```

---

## 🏆 Reward Function

Dense reward at every step in `[-1.0, 1.0]`, with a breakdown:

| Component | Signal |
|-----------|--------|
| `lives_saved_bonus` | Per rescue, weighted by zone severity |
| `resource_efficiency` | Bonus for deploying resources vs idle |
| `response_time_bonus` | Decaying bonus for acting early |
| `unattended_penalty` | Penalty per step a zone with trapped survivors goes unattended |
| `waste_penalty` | Penalty for mismatched resource type (e.g. water rescue on earthquake zone) |
| `evacuation_bonus` | Bonus per successful zone evacuation |

Grader score `[0.0, 1.0]` is computed at episode end via `GET /grade/{session_id}`.

---

## 📊 Baseline Scores

Tested with a deterministic greedy agent (seed=42). An LLM agent is expected to score higher.

| Task | Score | Grade | Rescued | Fatalities |
|------|-------|-------|---------|------------|
| `task_1_earthquake` | 0.7461 | B | 35 | 11 |
| `task_2_flood` | 0.6063 | C | 30 | 44 |
| `task_3_multi_disaster` | 0.5569 | C | 36 | 104 |
| **Average** | **0.6364** | C | — | — |

---

##  Setup & Usage

### Docker

```bash
docker build -t disaster-response-env .
docker run -p 7860:7860 \
  -e HF_TOKEN="hf_..." \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  disaster-response-env
```

### API

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset — returns observation + session_id
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_earthquake", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "action": {
      "action_type": "allocate_search_rescue",
      "zone_id": "Z1",
      "units": 4,
      "reasoning": "Z1 is CRITICAL with highest trapped count"
    }
  }'

# Full state
curl http://localhost:7860/state/YOUR_SESSION_ID

# Grader score
curl http://localhost:7860/grade/YOUR_SESSION_ID
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

Emits structured `[START]`, `[STEP]`, `[END]` logs to stdout.

---

##  Project Structure

```
disaster-response-env/
├── app/
│   ├── __init__.py
│   ├── models.py        # Pydantic models: Observation, Action, Reward
│   ├── tasks.py         # Task definitions + deterministic graders
│   └── environment.py   # Core step/reset/state logic
├── server/
│   ├── __init__.py
│   └── app.py           # FastAPI REST server
├── inference.py         # Baseline inference script (OpenAI client)
├── openenv.yaml         # OpenEnv spec manifest
├── baseline_results.json
├── Dockerfile
├── requirements.txt
└── README.md
```

---

##  Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | Hugging Face / API key | — |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

---

##  Submission Checklist

| Check | Status |
|-------|--------|
| HF Space deploys + `/health` returns 200 | ✅ |
| `openenv.yaml` valid with all required fields | ✅ |
| `Dockerfile` builds and runs cleanly | ✅ |
| `inference.py` in root, uses OpenAI client | ✅ |
| Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | ✅ |
| Emits `[START]`, `[STEP]`, `[END]` structured logs | ✅ |
| 3 tasks with graders, scores in `[0.0, 1.0]` | ✅ |
| Graders deterministic and reproducible (seed=42) | ✅ |
| Dense reward (not sparse binary) | ✅ |
| Typed Pydantic models (Observation / Action / Reward) | ✅ |
| `reset()` / `step()` / `state()` implemented | ✅ |
| Runtime < 20 min | ✅ |
| Runs on 2 vCPU / 8 GB RAM | ✅ |
| Real-world task (not a game or toy) | ✅ |

---

## 📄 License

MIT
