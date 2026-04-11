from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from app.models import ZoneState, ResourcePool, DisasterType, ZoneSeverity


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    disaster_type: DisasterType
    zones: List[ZoneState] = field(default_factory=list)
    resources: Optional[ResourcePool] = None
    target_rescue_rate: float = 0.7
    # BUG 2 fix — frozen original zones, never mutated
    _original_zones: List[ZoneState] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self):
        self._original_zones = copy.deepcopy(self.zones)

    def get_fresh_zones(self) -> List[ZoneState]:
        """Always returns a deep copy of the original zone state."""
        return copy.deepcopy(self._original_zones)

    def get_fresh_resources(self) -> ResourcePool:
        return copy.deepcopy(self.resources)


TASKS: Dict[str, TaskDefinition] = {}

# ── TASK 1: Easy ──────────────────────────────────────────────────────────────
TASKS["task_1_earthquake"] = TaskDefinition(
    task_id="task_1_earthquake",
    name="Urban Earthquake Response",
    description=(
        "A magnitude-7.2 earthquake has struck a city. "
        "Three districts have collapsed buildings with trapped survivors. "
        "Allocate search-and-rescue teams and medical units to maximize lives saved "
        "within 5 time steps."
    ),
    difficulty="easy",
    max_steps=5,
    disaster_type=DisasterType.EARTHQUAKE,
    zones=[
        ZoneState(
            zone_id="Z1",
            name="Downtown Core",
            disaster_type=DisasterType.EARTHQUAKE,
            severity=ZoneSeverity.CRITICAL,
            population=5000,
            trapped_casualties=80,
            injured=120,
            rescued=0,
            fatalities=0,
            accessibility=0.6,
        ),
        ZoneState(
            zone_id="Z2",
            name="Residential North",
            disaster_type=DisasterType.EARTHQUAKE,
            severity=ZoneSeverity.HIGH,
            population=8000,
            trapped_casualties=45,
            injured=200,
            rescued=0,
            fatalities=0,
            accessibility=0.8,
        ),
        ZoneState(
            zone_id="Z3",
            name="Industrial South",
            disaster_type=DisasterType.EARTHQUAKE,
            severity=ZoneSeverity.MEDIUM,
            population=1200,
            trapped_casualties=20,
            injured=60,
            rescued=0,
            fatalities=0,
            accessibility=0.9,
        ),
    ],
    resources=ResourcePool(
        search_rescue_teams=10,
        medical_teams=8,
        firefighting_units=2,
        water_rescue_teams=0,
        evacuation_vehicles=5,
    ),
    target_rescue_rate=0.7,
)

# ── TASK 2: Medium ────────────────────────────────────────────────────────────
TASKS["task_2_flood"] = TaskDefinition(
    task_id="task_2_flood",
    name="Regional Flash Flood Response",
    description=(
        "A catastrophic flash flood has inundated 5 zones. "
        "Water rescue teams and medical units are severely limited. "
        "You must triage which zones to prioritize, manage water rescue assets, "
        "coordinate evacuations, and minimize fatalities across 8 time steps."
    ),
    difficulty="medium",
    max_steps=8,
    disaster_type=DisasterType.FLOOD,
    zones=[
        ZoneState(
            zone_id="Z1",
            name="Riverside Village",
            disaster_type=DisasterType.FLOOD,
            severity=ZoneSeverity.CRITICAL,
            population=3000,
            trapped_casualties=90,
            injured=150,
            rescued=0,
            fatalities=0,
            accessibility=0.5,
        ),
        ZoneState(
            zone_id="Z2",
            name="Lowland Farms",
            disaster_type=DisasterType.FLOOD,
            severity=ZoneSeverity.HIGH,
            population=800,
            trapped_casualties=40,
            injured=80,
            rescued=0,
            fatalities=0,
            accessibility=0.7,
        ),
        ZoneState(
            zone_id="Z3",
            name="City Underpass",
            disaster_type=DisasterType.FLOOD,
            severity=ZoneSeverity.CRITICAL,
            population=400,
            trapped_casualties=60,
            injured=30,
            rescued=0,
            fatalities=0,
            accessibility=0.4,
        ),
        ZoneState(
            zone_id="Z4",
            name="Suburb East",
            disaster_type=DisasterType.FLOOD,
            severity=ZoneSeverity.MEDIUM,
            population=5000,
            trapped_casualties=25,
            injured=200,
            rescued=0,
            fatalities=0,
            accessibility=0.85,
        ),
        ZoneState(
            zone_id="Z5",
            name="Mountain Pass Road",
            disaster_type=DisasterType.FLOOD,
            severity=ZoneSeverity.LOW,
            population=200,
            trapped_casualties=10,
            injured=20,
            rescued=0,
            fatalities=0,
            accessibility=0.3,
        ),
    ],
    resources=ResourcePool(
        search_rescue_teams=5,
        medical_teams=4,
        firefighting_units=0,
        water_rescue_teams=6,
        evacuation_vehicles=3,
    ),
    target_rescue_rate=0.65,
)

# ── TASK 3: Hard ──────────────────────────────────────────────────────────────
TASKS["task_3_multi_disaster"] = TaskDefinition(
    task_id="task_3_multi_disaster",
    name="Compound Disaster: Earthquake + Wildfire + Flood",
    description=(
        "A perfect storm: a 6.8 earthquake triggers a gas-line wildfire in industrial areas "
        "and ruptures a dam causing downstream flooding — simultaneously. "
        "8 zones affected with 3 different disaster types requiring specialized resources. "
        "Resources are critically scarce. Zone accessibility degrades each turn without attention. "
        "You have 12 steps to coordinate a multi-agency response."
    ),
    difficulty="hard",
    max_steps=12,
    disaster_type=DisasterType.MULTI,
    zones=[
        ZoneState(
            zone_id="Z1",
            name="Downtown Collapse",
            disaster_type=DisasterType.EARTHQUAKE,
            severity=ZoneSeverity.CRITICAL,
            population=6000,
            trapped_casualties=120,
            injured=300,
            rescued=0,
            fatalities=0,
            accessibility=0.5,
        ),
        ZoneState(
            zone_id="Z2",
            name="Gas District Fire",
            disaster_type=DisasterType.FIRE,
            severity=ZoneSeverity.CRITICAL,
            population=2000,
            trapped_casualties=50,
            injured=180,
            rescued=0,
            fatalities=0,
            accessibility=0.6,
        ),
        ZoneState(
            zone_id="Z3",
            name="Flood Basin A",
            disaster_type=DisasterType.FLOOD,
            severity=ZoneSeverity.HIGH,
            population=4000,
            trapped_casualties=70,
            injured=120,
            rescued=0,
            fatalities=0,
            accessibility=0.55,
        ),
        ZoneState(
            zone_id="Z4",
            name="School District",
            disaster_type=DisasterType.EARTHQUAKE,
            severity=ZoneSeverity.HIGH,
            population=1500,
            trapped_casualties=45,
            injured=90,
            rescued=0,
            fatalities=0,
            accessibility=0.7,
        ),
        ZoneState(
            zone_id="Z5",
            name="Chemical Plant",
            disaster_type=DisasterType.FIRE,
            severity=ZoneSeverity.CRITICAL,
            population=500,
            trapped_casualties=30,
            injured=60,
            rescued=0,
            fatalities=0,
            accessibility=0.45,
        ),
        ZoneState(
            zone_id="Z6",
            name="Flood Basin B",
            disaster_type=DisasterType.FLOOD,
            severity=ZoneSeverity.MEDIUM,
            population=3000,
            trapped_casualties=35,
            injured=80,
            rescued=0,
            fatalities=0,
            accessibility=0.65,
        ),
        ZoneState(
            zone_id="Z7",
            name="Highway Collapse",
            disaster_type=DisasterType.EARTHQUAKE,
            severity=ZoneSeverity.MEDIUM,
            population=800,
            trapped_casualties=25,
            injured=40,
            rescued=0,
            fatalities=0,
            accessibility=0.75,
        ),
        ZoneState(
            zone_id="Z8",
            name="Hospital District",
            disaster_type=DisasterType.EARTHQUAKE,
            severity=ZoneSeverity.HIGH,
            population=900,
            trapped_casualties=15,
            injured=400,
            rescued=0,
            fatalities=0,
            accessibility=0.8,
        ),
    ],
    resources=ResourcePool(
        search_rescue_teams=8,
        medical_teams=6,
        firefighting_units=4,
        water_rescue_teams=3,
        evacuation_vehicles=4,
    ),
    target_rescue_rate=0.60,
)


def grade_task(task_id: str, env_state: Dict[str, Any]) -> float:
    """
    Deterministic grader. Returns score in [0.0, 1.0].
    grade_task() does not mutate zones — caller may pass live or snapshot.
    """
    zones: List[ZoneState] = env_state["zones"]  # snapshot — caller must deepcopy
    task = TASKS[task_id]

    # Use original zone data from frozen task definition (BUG 2/5 fix)
    original_zones = task._original_zones

    total_trapped = sum(z.trapped_casualties + z.rescued + z.fatalities for z in zones)
    total_rescued = sum(z.rescued for z in zones)
    total_fatalities = sum(z.fatalities for z in zones)

    # 1. Rescue score — normalized to achievable given step budget
    steps_per_zone = max(1, task.max_steps // max(1, len(original_zones)))
    achievable = min(
        sum(z.trapped_casualties + z.rescued + z.fatalities for z in zones),
        sum(min(z.trapped_casualties, steps_per_zone * 26) for z in original_zones),
    )
    rescue_score = min(1.0, total_rescued / achievable) if achievable > 0 else 1.0

    # 2. Fatality score
    fatality_score = 1.0 - min(1.0, total_fatalities / max(1, total_trapped))

    # 3. Coverage — critical/high zones with any rescues
    critical_zone_ids = {
        z.zone_id
        for z in original_zones
        if z.severity in (ZoneSeverity.CRITICAL, ZoneSeverity.HIGH, "critical", "high")
    }
    attended_count = sum(
        1 for z in zones if z.zone_id in critical_zone_ids and z.rescued > 0
    )
    coverage_score = (
        attended_count / len(critical_zone_ids) if critical_zone_ids else 1.0
    )

    # 4. Efficiency
    resources_used = env_state.get("resources_used", 0)
    resources_total = env_state.get("resources_total", 1)
    efficiency = min(1.0, resources_used / max(1, resources_total))

    # 5. Evacuation bonus
    evacuated = sum(1 for z in zones if z.is_evacuated)
    evacuation_score = min(1.0, evacuated / max(1, len(original_zones) // 2))

    if task.difficulty == "easy":
        score = (
            0.45 * rescue_score
            + 0.35 * coverage_score
            + 0.15 * fatality_score
            + 0.05 * efficiency
        )
    elif task.difficulty == "medium":
        score = (
            0.40 * rescue_score
            + 0.30 * coverage_score
            + 0.15 * fatality_score
            + 0.10 * efficiency
            + 0.05 * evacuation_score
        )
    else:  # hard
        score = (
            0.35 * rescue_score
            + 0.30 * coverage_score
            + 0.15 * fatality_score
            + 0.10 * efficiency
            + 0.10 * evacuation_score
        )

    # Clamp strictly within (0, 1) — validator requires score > 0 and score < 1
    return round(max(0.001, min(0.999, score)), 4)
