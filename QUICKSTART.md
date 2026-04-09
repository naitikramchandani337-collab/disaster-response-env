# Quick Start — Disaster Response OpenEnv

## What You Have

A production-ready OpenEnv environment simulating real-world emergency disaster response:

- Full OpenEnv spec (Pydantic models, reset/step/state API, openenv.yaml)
- 3 tasks: Easy earthquake → Medium flood → Hard multi-disaster
- Dense reward function with partial progress signals
- Baseline inference script with reproducible scores
- Docker-ready for HF Spaces deployment

---

## Project Structure

```
disaster-response-env/
├── app/
│   ├── models.py        # Pydantic: Observation, Action, Reward
│   ├── tasks.py         # Task definitions + deterministic graders
│   └── environment.py   # Core step/reset/state logic
├── server/
│   └── app.py           # FastAPI REST server
├── inference.py         # Baseline inference (OpenAI client)
├── validate.py          # Pre-submission test suite
├── openenv.yaml         # OpenEnv spec manifest
├── baseline_results.json
├── Dockerfile
└── requirements.txt
```

---

## Get Started in 3 Steps

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run validation
```bash
python validate.py
# All 10 tests must pass
```

### 3. Test the environment
```python
from app.environment import DisasterResponseEnv
from app.models import Action

env = DisasterResponseEnv("task_1_earthquake", seed=42)
obs = env.reset()

print(f"Scenario: {obs.disaster_scenario}")
print(f"Zones: {len(obs.zones)}")
print(f"Resources: {obs.available_resources}")

action = Action(action_type="allocate_search_rescue", zone_id="Z1", units=5)
resp = env.step(action)
print(f"Reward: {resp.reward.total:.4f}")
print(f"Rescued: {resp.observation.total_rescued}")
```

---

## Run the Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then test it:
```bash
# Health
curl http://localhost:7860/health

# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_earthquake", "seed": 42}'

# Step (use session_id from reset response)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_ID", "action": {"action_type": "allocate_search_rescue", "zone_id": "Z1", "units": 4}}'
```

---

## Run Baseline Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."

python inference.py
```

Output format:
```
[START] task=task_1_earthquake env=disaster-response-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=allocate_search_rescue(zone=Z1,units=4) reward=0.06 done=false error=null
...
[END] success=true steps=5 score=0.75 rewards=0.06,0.08,0.07,0.06,0.05
```

---

## Tasks at a Glance

| Task | Difficulty | Zones | Steps | Disaster |
|------|-----------|-------|-------|---------|
| `task_1_earthquake` | Easy | 3 | 5 | Earthquake |
| `task_2_flood` | Medium | 5 | 8 | Flood |
| `task_3_multi_disaster` | Hard | 8 | 12 | EQ + Fire + Flood |

---

## Docker

```bash
docker build -t disaster-response-env .
docker run -p 7860:7860 \
  -e HF_TOKEN="hf_..." \
  disaster-response-env
```

---

## Deploy to HF Spaces

See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions.
