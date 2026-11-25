"""Versioned, resolvable lander controllers and their original MAGUS actions."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from types import MappingProxyType
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
    lineage_id: str | None = None

    def metadata(self):
        return {"skill_id": self.skill_id, "action": action_record(self.action),
                "loader": self.loader, "artifact_digest": self.artifact_digest,
                "behavior_version": self.behavior_version, "provider": self.provider,
                "environment": self.environment, "objective_schema": self.objective_schema,
                "lineage_id": self.lineage_id or self.action.id}


@dataclass(frozen=True)
class PolicyBinding:
    """Explicit executable factory: no arbitrary import or unverified pickles."""
    factory: Callable
    artifact_paths: tuple
    provider_version: str

    def digest(self):
        if not self.artifact_paths or not self.provider_version:
            raise BoundaryError("UNRESOLVED_EXECUTABLE", "versioned provider artifacts required")
        import inspect
        import marshal
        import hashlib
        source = inspect.getsourcefile(self.factory)
        if not source or not Path(source).is_file():
            raise BoundaryError("UNRESOLVED_EXECUTABLE", "provider factory source must be inspectable")
        closure = [cell.cell_contents for cell in (self.factory.__closure__ or ())]
        try:
            closure_digest = digest(closure)
        except (TypeError, ValueError) as exc:
            raise BoundaryError("UNRESOLVED_EXECUTABLE", "provider factories require immutable serializable closure configuration") from exc
        code = getattr(self.factory, "__code__", None)
        if code is None:
            raise BoundaryError("UNRESOLVED_EXECUTABLE", "Python factory entrypoint required")
        entrypoint = {"module": self.factory.__module__, "qualname": self.factory.__qualname__,
                      "code": hashlib.sha256(marshal.dumps(code)).hexdigest(),
                      "defaults": self.factory.__defaults__, "kwdefaults": self.factory.__kwdefaults__}
        globals_used = {}
        for name, value in inspect.getclosurevars(self.factory).globals.items():
            try:
                digest(value)
                globals_used[name] = value
            except (TypeError, ValueError):
                origin = inspect.getsourcefile(value) if not inspect.ismodule(value) else getattr(value, "__file__", None)
                if not origin or not Path(origin).is_file():
                    raise BoundaryError("UNRESOLVED_EXECUTABLE", f"untracked factory global: {name}")
                code_value = getattr(value, "__code__", None)
                globals_used[name] = {"source": file_digest(Path(origin)), "name": getattr(value, "__qualname__", getattr(value, "__name__", name)),
                                      "code": hashlib.sha256(marshal.dumps(code_value)).hexdigest() if code_value else None}
        entrypoint["globals"] = globals_used
        return digest([self.provider_version, entrypoint, file_digest(Path(source)), closure_digest,
                       *[file_digest(Path(p)) for p in self.artifact_paths]])


class AtomicCandidateProvider(Protocol):
    def candidates(self) -> tuple[CandidateSkill, ...]: ...
    def bindings(self) -> dict[str, PolicyBinding]: ...


def artifact_digest(loader):
    paths = [Path(__file__), SUBREP / "env/lunar_lander_wrapper.py",
             SUBREP / "env/skill_executor.py"]
    if loader in ("ppo_deterministic", "ppo_stochastic", "ppo_noisy", "ppo_then_side_tradeoff"):
        paths.extend([SUBREP / "pilot/rl_pilot.py", SUBREP / "models/pilot_ppo.pt"])
    elif loader not in ("noop", "left_engine", "main_engine", "right_engine", "random"):
        raise BoundaryError("UNRESOLVED_EXECUTABLE", loader)
    if any(not p.is_file() for p in paths):
        raise BoundaryError("UNRESOLVED_EXECUTABLE", f"missing artifact for {loader}")
    return digest([file_digest(p) for p in paths])


class Registry:
    def __init__(self, candidates, bindings=None):
        self._bindings = MappingProxyType(dict(bindings or {}))
        self._by_action = {}
        self._by_skill = {}
        for candidate in candidates:
            candidate.metadata()  # Strict action checks before any rollout.
            if candidate.action.id in self._by_action or candidate.skill_id in self._by_skill:
                raise BoundaryError("IDENTITY_CONFLICT", "duplicate action or skill version")
            if not candidate.skill_id.endswith("." + candidate.behavior_version):
                raise BoundaryError("IDENTITY_CONFLICT", "skill ID must encode behavior version")
            self.verify(candidate)
            self._by_action[candidate.action.id] = candidate
            self._by_skill[candidate.skill_id] = candidate

    def verify(self, candidate):
        if candidate.environment != "mo-lunar-lander-v3" or candidate.objective_schema != "safety_fuel_v1":
            raise BoundaryError("SCHEMA_MISMATCH", candidate.skill_id)
        binding = self.bindings.get(candidate.loader)
        expected = binding.digest() if binding else artifact_digest(candidate.loader)
        if binding and candidate.provider != binding.provider_version:
            raise BoundaryError("SCHEMA_MISMATCH", "provider version differs")
        if candidate.artifact_digest != expected:
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
    def bindings(self):
        return self._bindings

    @property
    def candidates(self):
        return tuple(self._by_action.values())

    def policy(self, candidate, seed):
        self.verify(candidate)
        if candidate.loader in self.bindings:
            policy = self.bindings[candidate.loader].factory(seed)
            if not callable(policy):
                raise BoundaryError("UNRESOLVED_EXECUTABLE", "factory did not return a controller")
            return policy
        if candidate.loader in ("ppo_deterministic", "ppo_stochastic", "ppo_noisy", "ppo_then_side_tradeoff"):
            enable_subrep()
            from pilot.rl_pilot import RLPilot
            pilot = RLPilot.load(str(SUBREP / "models/pilot_ppo.pt"), map_location="cpu")
            rng = np.random.default_rng(seed)
            steps = 0
            def policy(obs):
                nonlocal steps
                steps += 1
                if candidate.loader == "ppo_then_side_tradeoff" and steps >= 30:
                    return 1
                if candidate.loader == "ppo_noisy" and rng.random() < .15:
                    return int(rng.integers(4))
                if candidate.loader == "ppo_stochastic":
                    import torch
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed + steps)
                        return pilot.predict(obs, deterministic=False)
                return pilot.predict(obs, deterministic=True)
            return policy
        if candidate.loader == "random":
            rng = np.random.default_rng(seed)
            return lambda obs: int(rng.integers(4))
        action = {"noop": 0, "left_engine": 1, "main_engine": 2, "right_engine": 3}[candidate.loader]
        return lambda obs: action


def controlled_registry(extended=False):
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
    if extended:
        profiles.update({"ppo_stochastic": ([.7, .7, .7, .5, .3, .5, .6, .1], .3),
                         "ppo_noisy": ([.5, .8, .5, .7, .6, .4, .4, .1], .5),
                         "ppo_then_side_tradeoff": ([.4, .7, .3, .6, .5, .4, .3, .1], .6)})
    candidates = []
    for loader, (alignment, risk) in profiles.items():
        action = Action(f"lander.{loader}", np.asarray(alignment), risk, np.zeros(8))
        # Predictions are illustrative motivational descriptors, not evidence.
        # Zero delta avoids inventing eight observed goal effects from two rewards.
        candidates.append(CandidateSkill(f"lander.{loader}.v1", action, loader, artifact_digest(loader)))
    return Registry(candidates)
