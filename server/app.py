import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Any
import uuid

from app.models import Action, StepResponse, StateResponse
from app.environment import DisasterResponseEnv, TTLSessionStore
from app.tasks import TASKS

app = FastAPI(
    title       = "Disaster Response OpenEnv",
    description = "OpenEnv-compliant Disaster Response Simulation",
    version     = "1.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ISSUE 4 fix — bounded TTL session store, no unbounded dict
_sessions = TTLSessionStore(maxsize=100, ttl=3600)


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
            "name":          t.name,
            "difficulty":    t.difficulty,
            "max_steps":     t.max_steps,
            "disaster_type": t.disaster_type.value,
            "description":   t.description,
            "num_zones":     len(t.zones),
        }
        for tid, t in TASKS.items()
    }


@app.post("/reset")
def reset(req: ResetRequest = None, response: Response = None):
    """
    Reset (or create) an episode.
    BUG 1 fix: session_id is returned BOTH in the JSON body AND as a header.
    """
    if req is None:
        req = ResetRequest()
    session_id = req.session_id or str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions.set(session_id, env)

    # BUG 1 fix — inject into header AND body
    if response is not None:
        response.headers["X-Session-Id"] = session_id

    obs_dict = obs.model_dump()          # ISSUE 8 fix — model_dump() not dict()
    obs_dict["session_id"] = session_id  # also embed in observation body
    return {
        "session_id":  session_id,
        "observation": obs_dict,
    }


@app.post("/step")
def step(req: StepRequest):
    """Apply action to environment. Returns observation, reward, done, info."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first.",
        )
    try:
        action = Action(**{k: v for k, v in req.action.items() if k in Action.model_fields})
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    return result.model_dump()           # ISSUE 8 fix


@app.get("/state/{session_id}")
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state().model_dump()      # ISSUE 8 fix


@app.get("/grade/{session_id}")
def grade(session_id: str):
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
    _sessions.delete(session_id)
    return {"deleted": session_id}


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
