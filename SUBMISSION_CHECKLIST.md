# Submission Checklist — Disaster Response OpenEnv

## Phase 1: Automated Validation Gate (must all pass)

### HF Space
- [ ] Space created and set to **Public**
- [ ] Space URL responds: `curl https://naitikramchandani337-disaster-response-env.hf.space/health`
- [ ] `POST /reset` with `{}` returns HTTP 200

### OpenEnv Spec
- [ ] `openenv.yaml` present in root
- [ ] `openenv.yaml` has: `name`, `version`, `description`, `tasks`, `observation_space`, `action_space`
- [ ] 3 tasks defined with `id`, `name`, `difficulty`, `score_range: [0.0, 1.0]`
- [ ] Typed Pydantic models: `Observation`, `Action`, `Reward`
- [ ] `reset()` → `Observation`, `step()` → `StepResponse`, `state()` → `StateResponse`

### Docker
- [ ] `Dockerfile` in repo root
- [ ] `docker build .` completes without errors
- [ ] No fake packages in `requirements.txt` (no `openenv-core`)

### Baseline Inference
- [ ] `inference.py` in root directory
- [ ] Uses `OpenAI` client
- [ ] Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] Emits `[START]`, `[STEP]`, `[END]` in exact mandatory format
- [ ] Completes all 3 tasks without crashing
- [ ] Scores in `[0.0, 1.0]`

### Tasks + Graders
- [ ] 3 tasks: easy, medium, hard
- [ ] All graders return `float` in `[0.0, 1.0]`
- [ ] Graders are deterministic (same input → same score)
- [ ] Graders do NOT always return the same score

---

## Phase 2: Agentic Evaluation

- [ ] Baseline agent re-run produces consistent scores
- [ ] Standard LLM agent can make meaningful progress
- [ ] Score variance across runs is reasonable

---

## Disqualification Criteria (avoid these)

- [ ] Space does not deploy or respond → **DISQUALIFIED**
- [ ] Plagiarised or trivially modified environment → **DISQUALIFIED**
- [ ] Graders always return the same score → **DISQUALIFIED**
- [ ] No `inference.py` in root → **DISQUALIFIED**
- [ ] `[START]`/`[STEP]`/`[END]` format wrong → **incorrect scoring**

---

## Run Validation Locally

```bash
python validate.py
```

Expected output:
```
  [PASS] File Structure
  [PASS] openenv.yaml
  [PASS] Imports
  [PASS] Environment Init
  [PASS] reset()
  [PASS] step()
  [PASS] state()
  [PASS] Graders (3 tasks)
  [PASS] Server Endpoints
  [PASS] inference.py Format
  10 / 10 tests passed
```

---

## Baseline Scores

| Task | Score | Grade |
|------|-------|-------|
| `task_1_earthquake` (easy) | 0.7461 | B |
| `task_2_flood` (medium) | 0.6063 | C |
| `task_3_multi_disaster` (hard) | 0.5569 | C |
| Average | 0.6364 | C |

---

## Submission

1. Run `python validate.py` — all 10 must pass
2. Confirm Space URL is live and public
3. Submit Space URL to competition portal
