import sys
import os

# Ensure root is on path so `app.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Any
import uuid

from app.models import Action, StepResponse, StateResponse
from app.environment import DisasterResponseEnv
from app.tasks import TASKS

app = FastAPI(
    title       = "Disaster Response OpenEnv",
    description = "OpenEnv-compliant Disaster Response Simulation",
    version     = "1.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# In-memory session store
_sessions: Dict[str, DisasterResponseEnv] = {}


class ResetRequest(BaseModel):
    task_id:    str = "task_1_earthquake"
    seed:       int = 42
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action:     Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "healthy", "env": "disaster-response-openenv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        tid: {
            "name":         t.name,
            "difficulty":   t.difficulty,
            "max_steps":    t.max_steps,
            "disaster_type": t.disaster_type.value,
            "description":  t.description,
            "num_zones":    len(t.zones),
        }
        for tid, t in TASKS.items()
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset (or create) an episode. Returns initial observation + session_id."""
    if req is None:
        req = ResetRequest()
    session_id = req.session_id or str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env
    return {
        "session_id":  session_id,
        "observation": obs.dict(),
    }


@app.post("/step")
def step(req: StepRequest):
    """Apply action to environment. Returns observation, reward, done, info."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first."
        )
    try:
        action = Action(**{k: v for k, v in req.action.items() if k in Action.model_fields})
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    return result.dict()


@app.get("/state/{session_id}")
def state(session_id: str):
    """Return full current state for a session."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state().dict()


@app.get("/grade/{session_id}")
def grade(session_id: str):
    """Return current grader score for a session."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    st = env.state()
    return {
        "session_id":   session_id,
        "task_id":      env.task_id,
        "grader_score": st.grader_score,
        "step":         env._time_step,
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
