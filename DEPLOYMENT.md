# Deployment Guide — Disaster Response OpenEnv

## Deploying to Hugging Face Spaces

### Step 1: Create a HF Account and Space

1. Go to https://huggingface.co/join and create an account
2. Go to https://huggingface.co/spaces → "New Space"
3. Fill in:
   - Space name: `disaster-response-env`
   - SDK: **Docker**
   - Visibility: **Public**
4. Click "Create Space"

### Step 2: Push Your Files

```bash
# Clone your new Space repo
git clone https://huggingface.co/spaces/naitikramchandani337/disaster-response-env
cd disaster-response-env

# Copy all project files
cp -r /path/to/disaster-response-env/app .
cp -r /path/to/disaster-response-env/server .
cp /path/to/disaster-response-env/inference.py .
cp /path/to/disaster-response-env/openenv.yaml .
cp /path/to/disaster-response-env/Dockerfile .
cp /path/to/disaster-response-env/requirements.txt .
cp /path/to/disaster-response-env/README.md .
cp /path/to/disaster-response-env/baseline_results.json .

# Push
git add .
git commit -m "Initial deployment: Disaster Response OpenEnv"
git push origin main
```

### Step 3: Set Environment Variables

In your Space settings (Settings → Variables and secrets):

| Variable | Value |
|----------|-------|
| `HF_TOKEN` | Your HF API token |
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` |

### Step 4: Verify Deployment

Once the Space shows "Running":

```bash
SPACE_URL="https://naitikramchandani337-disaster-response-env.hf.space"

# Health check
curl $SPACE_URL/health

# Reset
curl -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_earthquake", "seed": 42}'
```

---

## Local Docker Testing

Test the Docker image before pushing:

```bash
# Build
docker build -t disaster-response-env .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN="hf_..." \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  disaster-response-env

# Test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Troubleshooting

**Space won't build**
- Check Logs tab for pip install errors
- Ensure `requirements.txt` has no fake packages (`openenv-core` does not exist)
- Verify all files are committed

**`/reset` returns 404**
- The server takes ~10s to start — wait for "Running" status
- Check Logs for uvicorn startup errors

**Inference script hangs**
- `MAX_STEPS = 50` is the safety cap — each task runs at most 50 steps
- Total runtime for 3 tasks should be under 10 minutes

**API key errors**
- Set `HF_TOKEN` in Space secrets, not as a plain variable
- The script also accepts `OPENAI_API_KEY` as a fallback

---

## Pre-Submission Checklist

Run locally before submitting:

```bash
python validate.py
# Must show: 10 / 10 tests passed
```

Then verify:
- [ ] Space is **Public**
- [ ] `/health` returns 200
- [ ] `/reset` with `{}` returns 200 with `session_id` + `observation`
- [ ] `inference.py` is in the root directory
- [ ] `openenv.yaml` is in the root directory
- [ ] Docker builds without errors
