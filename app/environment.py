from __future__ import annotations
import copy, math, random
from typing import Dict, List, Optional, Tuple, Any

from app.models import (
    Observation, Action, Reward, StepResponse, EpisodeInfo,
    StateResponse, ZoneState, ResourcePool,
    DisasterType, ZoneSeverity
)
from app.tasks import TASKS, TaskDefinition, grade_task

SEVERITY_ORDER = [
    ZoneSeverity.NONE, ZoneSeverity.LOW,
    ZoneSeverity.MEDIUM, ZoneSeverity.HIGH, ZoneSeverity.CRITICAL
]
# Build index for both enum values and their string representations (use_enum_values=True)
SEVERITY_INDEX = {s: i for i, s in enumerate(SEVERITY_ORDER)}
SEVERITY_INDEX.update({s.value: i for i, s in enumerate(SEVERITY_ORDER)})

RESCUE_RATE = {
    DisasterType.EARTHQUAKE: 6,
    DisasterType.FLOOD:      5,
    DisasterType.FIRE:       4,
    DisasterType.MULTI:      5,
    "earthquake": 6,
    "flood":      5,
    "fire":       4,
    "multi":      5,
}
TREATMENT_RATE    = 12
FIREFIGHT_RATE    = 0.25
WATER_RESCUE_RATE = 7
UNATTENDED_DECAY  = 0.02   # accessibility drop per step unattended
FATALITY_RATE     = 0.02   # fraction of remaining trapped → fatalities per step unattended
FIRE_SPREAD_THRESH = 3


class DisasterResponseEnv:
    def __init__(self, task_id: str = "task_1_earthquake", seed: int = 42):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")
        self.task_id   = task_id
        self.seed      = seed
        self._rng      = random.Random(seed)
        self._task_def = TASKS[task_id]

        self._zones:           List[ZoneState] = []
        self._resources:       ResourcePool    = None
        self._time_step:       int             = 0
        self._done:            bool            = False
        self._episode_rewards: List[float]     = []
        self._cumulative_reward: float         = 0.0
        self._last_feedback:   str             = ""
        self._resources_used:  int             = 0
        self._resources_total: int             = 0

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)
        self._zones     = copy.deepcopy(self._task_def.zones)
        self._resources = copy.deepcopy(self._task_def.resources)
        self._time_step = 0
        self._done      = False
        self._episode_rewards   = []
        self._cumulative_reward = 0.0
        self._last_feedback     = "Disaster event detected. Awaiting resource allocation orders."
        self._resources_used    = 0
        self._resources_total   = (
            self._resources.search_rescue_teams
            + self._resources.medical_teams
            + self._resources.firefighting_units
            + self._resources.water_rescue_teams
            + self._resources.evacuation_vehicles
        )
        return self._build_observation()

    def step(self, action: Action) -> StepResponse:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        feedback, waste = self._apply_action(action)
        self._simulate_dynamics()
        self._time_step += 1
        self._done = (
            self._time_step >= self._task_def.max_steps
            or self._all_zones_stable()
        )

        reward = self._compute_reward(waste, feedback)
        self._episode_rewards.append(reward.total)
        self._cumulative_reward += reward.total
        self._last_feedback = feedback

        obs  = self._build_observation()
        info = self._build_info()
        return StepResponse(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> StateResponse:
        return StateResponse(
            observation     = self._build_observation(),
            episode_rewards = self._episode_rewards,
            grader_score    = self._get_grader_score(),
            task_metadata   = {
                "task_id":     self._task_def.task_id,
                "name":        self._task_def.name,
                "difficulty":  self._task_def.difficulty,
                "max_steps":   self._task_def.max_steps,
                "description": self._task_def.description,
            }
        )

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> Tuple[str, float]:
        zone = self._get_zone(action.zone_id)
        if zone is None:
            return f"Zone {action.zone_id} not found.", 0.1

        at    = action.action_type
        n     = max(0, action.units)
        waste = 0.0

        if at == "allocate_search_rescue":
            available = self._resources.search_rescue_teams
            deploy    = min(n, available)
            over      = n - deploy
            self._resources.search_rescue_teams -= deploy
            zone.search_rescue_teams += deploy
            zone.turns_unattended = 0
            self._resources_used += deploy
            waste = 0.05 * over
            feedback = (f"Deployed {deploy} S&R teams to {zone.name}."
                        + (f" {over} over-requested." if over else ""))

        elif at == "allocate_medical":
            available = self._resources.medical_teams
            deploy    = min(n, available)
            over      = n - deploy
            self._resources.medical_teams -= deploy
            zone.medical_teams += deploy
            zone.turns_unattended = 0
            self._resources_used += deploy
            waste = 0.05 * over
            feedback = (f"Deployed {deploy} medical teams to {zone.name}."
                        + (f" {over} over-requested." if over else ""))

        elif at == "allocate_firefighting":
            if zone.disaster_type not in ("fire", "multi"):
                return f"Zone {zone.name} has no fire hazard. Wasted deployment.", 0.15
            available = self._resources.firefighting_units
            deploy    = min(n, available)
            self._resources.firefighting_units -= deploy
            zone.firefighting_units += deploy
            zone.turns_unattended = 0
            self._resources_used += deploy
            feedback = f"Deployed {deploy} firefighting units to {zone.name}."

        elif at == "allocate_water_rescue":
            if zone.disaster_type not in ("flood", "multi"):
                return f"Zone {zone.name} has no flood. Water rescue wasted.", 0.15
            available = self._resources.water_rescue_teams
            deploy    = min(n, available)
            self._resources.water_rescue_teams -= deploy
            zone.water_rescue_teams += deploy
            zone.turns_unattended = 0
            self._resources_used += deploy
            feedback = f"Deployed {deploy} water rescue teams to {zone.name}."

        elif at == "evacuate_zone":
            if self._resources.evacuation_vehicles <= 0:
                return "No evacuation vehicles available.", 0.05
            if zone.trapped_casualties > 10:
                return (f"Cannot evacuate {zone.name} — "
                        f"{zone.trapped_casualties} still trapped."), 0.0
            zone.is_evacuated = True
            self._resources.evacuation_vehicles -= 1
            feedback = f"Zone {zone.name} evacuated successfully."

        elif at == "prioritize_zone":
            zone.is_prioritized = True
            feedback = f"Zone {zone.name} marked as priority."

        elif at == "standby":
            return "No action taken this step.", 0.05

        else:
            return "Unknown action.", 0.1

        return feedback, waste

    # ── World dynamics ────────────────────────────────────────────────────────

    def _simulate_dynamics(self):
        for zone in self._zones:
            if zone.severity == "none":
                continue

            base_rate  = RESCUE_RATE.get(zone.disaster_type, 5)
            sr_rescued = int(zone.search_rescue_teams * base_rate * zone.accessibility)

            wr_rescued = 0
            if zone.disaster_type in ("flood", "multi"):
                wr_rescued = int(zone.water_rescue_teams * WATER_RESCUE_RATE * zone.accessibility)

            total_rescued = min(zone.trapped_casualties, sr_rescued + wr_rescued)
            zone.trapped_casualties = max(0, zone.trapped_casualties - total_rescued)
            zone.rescued += total_rescued

            treated = min(zone.injured, zone.medical_teams * TREATMENT_RATE)
            zone.injured = max(0, zone.injured - treated)

            if zone.firefighting_units > 0 and zone.disaster_type in ("fire", "multi"):
                zone.accessibility = min(
                    1.0,
                    zone.accessibility + zone.firefighting_units * FIREFIGHT_RATE
                )

            has_teams = (
                zone.search_rescue_teams + zone.medical_teams
                + zone.firefighting_units + zone.water_rescue_teams
            ) > 0

            if not has_teams and zone.trapped_casualties > 0:
                zone.turns_unattended += 1
                zone.accessibility = max(0.0, zone.accessibility - UNATTENDED_DECAY)

                sev_idx = SEVERITY_INDEX.get(zone.severity, 1)
                new_fatalities = math.ceil(
                    zone.trapped_casualties * FATALITY_RATE * sev_idx / 4
                )
                new_fatalities = min(zone.trapped_casualties, new_fatalities)
                zone.trapped_casualties -= new_fatalities
                zone.fatalities         += new_fatalities

                if (zone.disaster_type in ("fire", "multi")
                        and zone.turns_unattended >= FIRE_SPREAD_THRESH):
                    si = SEVERITY_INDEX.get(zone.severity, 0)
                    if si < len(SEVERITY_ORDER) - 1:
                        zone.severity = SEVERITY_ORDER[si + 1].value

            # Reset team counts — must be re-allocated each step
            zone.search_rescue_teams = 0
            zone.medical_teams       = 0
            zone.firefighting_units  = 0
            zone.water_rescue_teams  = 0

            if zone.trapped_casualties == 0 and zone.injured == 0:
                zone.severity = "none"

    # ── Reward computation ────────────────────────────────────────────────────

    def _compute_reward(self, waste_penalty: float, feedback: str) -> Reward:
        zones = self._zones

        lives_saved = 0.0
        for z in zones:
            sev_weight = SEVERITY_INDEX.get(z.severity, 1) / 4.0
            lives_saved += z.rescued * sev_weight * 0.005

        unattended = sum(
            z.turns_unattended * 0.02
            for z in zones
            if z.trapped_casualties > 0
        )

        steps_remaining = self._task_def.max_steps - self._time_step
        time_bonus  = 0.01 * steps_remaining / self._task_def.max_steps
        efficiency  = 0.02 if self._resources_used > 0 else -0.02
        evac_bonus  = sum(0.03 for z in zones if z.is_evacuated)

        raw_total = (
              lives_saved
            + efficiency
            + time_bonus
            + evac_bonus
            - waste_penalty
            - unattended
        )
        total = round(max(-1.0, min(1.0, raw_total)), 4)

        return Reward(
            total               = total,
            lives_saved_bonus   = round(lives_saved, 4),
            resource_efficiency = round(efficiency, 4),
            response_time_bonus = round(time_bonus, 4),
            unattended_penalty  = round(-unattended, 4),
            waste_penalty       = round(-waste_penalty, 4),
            evacuation_bonus    = round(evac_bonus, 4),
            explanation         = (
                f"Step {self._time_step}: saved={lives_saved:.3f} "
                f"eff={efficiency:.3f} time={time_bonus:.3f} "
                f"evac={evac_bonus:.3f} waste={-waste_penalty:.3f} "
                f"unattended={-unattended:.3f}"
            )
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_zone(self, zone_id: str) -> Optional[ZoneState]:
        return next((z for z in self._zones if z.zone_id == zone_id), None)

    def _all_zones_stable(self) -> bool:
        return all(
            z.trapped_casualties == 0 and z.severity == "none"
            for z in self._zones
        )

    def _get_grader_score(self) -> float:
        return grade_task(self.task_id, {
            "zones":           self._zones,
            "resources_used":  self._resources_used,
            "resources_total": self._resources_total,
        })

    def _build_observation(self) -> Observation:
        return Observation(
            task_id               = self.task_id,
            disaster_scenario     = self._task_def.name,
            disaster_type         = self._task_def.disaster_type.value if hasattr(self._task_def.disaster_type, "value") else str(self._task_def.disaster_type),
            time_step             = self._time_step,
            max_steps             = self._task_def.max_steps,
            zones                 = copy.deepcopy(self._zones),
            available_resources   = copy.deepcopy(self._resources),
            total_rescued         = sum(z.rescued    for z in self._zones),
            total_fatalities      = sum(z.fatalities for z in self._zones),
            total_injured_treated = sum(
                max(0, orig.injured - curr.injured)
                for orig, curr in zip(self._task_def.zones, self._zones)
            ),
            cumulative_reward     = round(self._cumulative_reward, 4),
            active_disaster_zones = [
                z.zone_id for z in self._zones
                if z.trapped_casualties > 0 or z.severity != "none"
            ],
            prioritized_zones     = [z.zone_id for z in self._zones if z.is_prioritized],
            last_action_feedback  = self._last_feedback,
            done                  = self._done,
        )

    def _build_info(self) -> EpisodeInfo:
        score = self._get_grader_score()
        rescued = sum(z.rescued for z in self._zones)
        grade = (
            "A" if score >= 0.85 else
            "B" if score >= 0.70 else
            "C" if score >= 0.55 else
            "D" if score >= 0.40 else "F"
        )
        return EpisodeInfo(
            task_id          = self.task_id,
            steps_taken      = self._time_step,
            final_score      = score,
            total_rescued    = rescued,
            total_fatalities = sum(z.fatalities for z in self._zones),
            efficiency       = round(
                self._resources_used / max(1, self._resources_total), 3
            ),
            grade            = grade,
        )
