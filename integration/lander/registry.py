"""Versioned, resolvable lander controllers and their original MAGUS actions."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from core.state import Action
from .contracts import BoundaryError, action_record, digest, file_digest
from .vendor import SUBREP, enable_subrep


@dataclass(frozen=True)
class CandidateSkill:
    skill_id: str
    action: Action
    loader: str
    artifact_digest: str
    behavior_version: str = "v1"
    provider: str = "controlled_lander_v1"
    environment: str = "mo-lunar-lander-v3"
    objective_schema: str = "safety_fuel_v1"

    def metadata(self):
        return {"skill_id": self.skill_id, "action": action_record(self.action),
                "loader": self.loader, "artifact_digest": self.artifact_digest,
                "behavior_version": self.behavior_version, "provider": self.provider,
                "environment": self.environment, "objective_schema": self.objective_schema}


def artifact_digest(loader):
    paths = [Path(__file__), SUBREP / "env/lunar_lander_wrapper.py",
             SUBREP / "env/skill_executor.py"]
    if loader == "ppo_deterministic":
        paths.extend([SUBREP / "pilot/rl_pilot.py", SUBREP / "models/pilot_ppo.pt"])
    elif loader not in ("noop", "left_engine", "main_engine", "right_engine", "random"):
        raise BoundaryError("UNRESOLVED_EXECUTABLE", loader)
    if any(not p.is_file() for p in paths):
        raise BoundaryError("UNRESOLVED_EXECUTABLE", f"missing artifact for {loader}")
    return digest([file_digest(p) for p in paths])


class Registry:
    def __init__(self, candidates):
        self._by_action = {}
        self._by_skill = {}
        for candidate in candidates:
            candidate.metadata()  # Strict action checks before any rollout.
            if candidate.action.id in self._by_action or candidate.skill_id in self._by_skill:
                raise BoundaryError("IDENTITY_CONFLICT", "duplicate action or skill version")
            if candidate.skill_id != f"lander.{candidate.loader}.{candidate.behavior_version}":
                raise BoundaryError("IDENTITY_CONFLICT", "skill ID must encode behavior version")
            self.verify(candidate)
            self._by_action[candidate.action.id] = candidate
            self._by_skill[candidate.skill_id] = candidate

    @staticmethod
    def verify(candidate):
        if candidate.environment != "mo-lunar-lander-v3" or candidate.objective_schema != "safety_fuel_v1":
            raise BoundaryError("SCHEMA_MISMATCH", candidate.skill_id)
        if candidate.artifact_digest != artifact_digest(candidate.loader):
            raise BoundaryError("ARTIFACT_CHANGED", candidate.skill_id)

    def resolve(self, action):
        action_record(action)
        candidate = self._by_action.get(action.id)
        if candidate is None:
            raise BoundaryError("UNKNOWN_BEHAVIOR", action.id)
        self.verify(candidate)
        if action_record(action) != action_record(candidate.action):
            raise BoundaryError("INVALID_ACTION", "descriptor differs from registered behavior")
        return candidate

    def by_skill(self, skill_id):
        candidate = self._by_skill.get(skill_id)
        if candidate is None:
            raise BoundaryError("UNKNOWN_BEHAVIOR", skill_id)
        self.verify(candidate)
        return candidate

    @property
    def candidates(self):
        return tuple(self._by_action.values())

    def policy(self, candidate, seed):
        self.verify(candidate)
        if candidate.loader == "ppo_deterministic":
            enable_subrep()
            from pilot.rl_pilot import RLPilot
            pilot = RLPilot.load(str(SUBREP / "models/pilot_ppo.pt"), map_location="cpu")
            return lambda obs: pilot.predict(obs, deterministic=True)
        if candidate.loader == "random":
            rng = np.random.default_rng(seed)
            return lambda obs: int(rng.integers(4))
        action = {"noop": 0, "left_engine": 1, "main_engine": 2, "right_engine": 3}[candidate.loader]
        return lambda obs: action


def controlled_registry():
    """SubRep's controlled pool, using its supplied PPO artifact; no training."""
    enable_subrep()
    profiles = {
        "ppo_deterministic": ([.8, .6, .8, .4, .2, .6, .8, .1], .2),
        "noop": ([.1, .1, .1, .0, .0, .9, .1, .0], .8),
        "left_engine": ([.2, .6, .2, .6, .5, .1, .1, .0], .8),
        "main_engine": ([.6, .4, .5, .3, .2, .2, .4, .0], .5),
        "right_engine": ([.2, .6, .2, .6, .5, .1, .1, .0], .8),
        "random": ([.1, .8, .1, .8, .9, .0, .0, .0], .95),
    }
    candidates = []
    for loader, (alignment, risk) in profiles.items():
        action = Action(f"lander.{loader}", np.asarray(alignment), risk, np.zeros(8))
        # Predictions are illustrative motivational descriptors, not evidence.
        # Zero delta avoids inventing eight observed goal effects from two rewards.
        candidates.append(CandidateSkill(f"lander.{loader}.v1", action, loader, artifact_digest(loader)))
    return Registry(candidates)
