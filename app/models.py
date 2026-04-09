from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class DisasterType(str, Enum):
    EARTHQUAKE = "earthquake"
    FLOOD      = "flood"
    FIRE       = "fire"
    MULTI      = "multi"


class ActionType(str, Enum):
    ALLOCATE_SEARCH_RESCUE = "allocate_search_rescue"
    ALLOCATE_MEDICAL       = "allocate_medical"
    ALLOCATE_FIREFIGHTING  = "allocate_firefighting"
    ALLOCATE_WATER_RESCUE  = "allocate_water_rescue"
    EVACUATE_ZONE          = "evacuate_zone"
    PRIORITIZE_ZONE        = "prioritize_zone"
    STANDBY                = "standby"


class ZoneSeverity(str, Enum):
    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class ZoneState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    zone_id:             str
    name:                str
    disaster_type:       Optional[DisasterType] = None
    severity:            ZoneSeverity
    population:          int
    trapped_casualties:  int
    injured:             int
    rescued:             int
    fatalities:          int
    search_rescue_teams: int   = 0
    medical_teams:       int   = 0
    firefighting_units:  int   = 0
    water_rescue_teams:  int   = 0
    is_evacuated:        bool  = False
    is_prioritized:      bool  = False
    turns_unattended:    int   = 0
    accessibility:       float = Field(1.0, ge=0.0, le=1.0)


class ResourcePool(BaseModel):
    search_rescue_teams: int
    medical_teams:       int
    firefighting_units:  int
    water_rescue_teams:  int
    evacuation_vehicles: int


class Observation(BaseModel):
    # session_id included so agents can read it from the JSON body (BUG 1 fix)
    session_id:            Optional[str] = None
    task_id:               str
    disaster_scenario:     str
    disaster_type:         str
    time_step:             int
    max_steps:             int
    zones:                 List[ZoneState]
    available_resources:   ResourcePool
    total_rescued:         int
    total_fatalities:      int
    total_injured_treated: int
    cumulative_reward:     float
    active_disaster_zones: List[str]
    prioritized_zones:     List[str]
    last_action_feedback:  Optional[str] = None
    done:                  bool


class Action(BaseModel):
    action_type: str
    zone_id:     str
    units:       int = Field(default=1, ge=0, le=20)
    reasoning:   Optional[str] = None

    # ISSUE 1 — validate zone_id is non-empty and starts with Z
    @field_validator("zone_id")
    @classmethod
    def zone_id_valid(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("zone_id cannot be empty")
        if not v.startswith("Z"):
            raise ValueError(f"zone_id must start with 'Z', got '{v}'")
        return v


class Reward(BaseModel):
    total:               float
    lives_saved_bonus:   float
    resource_efficiency: float
    response_time_bonus: float
    unattended_penalty:  float
    waste_penalty:       float
    evacuation_bonus:    float
    explanation:         str


class EpisodeInfo(BaseModel):
    task_id:          str
    steps_taken:      int
    final_score:      float = Field(..., ge=0.0, le=1.0)   # MINOR 11 fix
    total_rescued:    int
    total_fatalities: int
    efficiency:       float
    grade:            str


class StepResponse(BaseModel):
    observation: Observation
    reward:      Reward
    done:        bool
    info:        EpisodeInfo


class StateResponse(BaseModel):
    observation:     Observation
    episode_rewards: List[float]
    grader_score:    float
    task_metadata:   Dict
