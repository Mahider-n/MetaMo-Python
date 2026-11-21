"""Strict, versioned integration contracts. Storage never repairs inputs."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np


class BoundaryError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(value, size, code="INVALID_CONTEXT", unit=False):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise BoundaryError(code, "expected numeric vector") from exc
    if a.shape != (size,) or not np.isfinite(a).all():
        raise BoundaryError(code, f"expected finite shape ({size},)")
    if unit and ((a < 0).any() or (a > 1).any()):
        raise BoundaryError(code, "expected values in [0, 1]")
    return a


def weights(value):
    a = vector(value, 2, "INVALID_GEOMETRY")
    if (a < 0).any() or not math.isclose(float(a.sum()), 1, abs_tol=1e-9, rel_tol=0):
        raise BoundaryError("INVALID_GEOMETRY", "weights must be nonnegative and sum to one")
    return tuple(float(x) for x in a)


def budget(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise BoundaryError("INVALID_GEOMETRY", "epsilon must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class MeasurementContract:
    environment: str = "mo-lunar-lander-v3"
    objective_mode: str = "environment_objectives"
    objective_names: tuple = ("safety", "fuel")
    objective_units: tuple = ("environment_reward", "environment_reward")
    reward_mapping: str = "terminal_plus_shaping__negative_engine_costs_v1"
    scalarization: str = "sum_safety_fuel_v1"
    aggregation: str = "matched_seed_arithmetic_mean_population_std_v1"
    reset_protocol: str = "env_reset_and_fresh_seeded_policy_per_episode_v1"
    gamma: float = 0.99
    horizon: int = 200
    seeds: tuple = (42, 43, 44, 45, 46)
    mo_gymnasium_version: str = "1.3.1"
    gymnasium_version: str = "1.4.0"
    box2d_version: str = "2.3.5"
    version: str = "lander_measurement_v1"

    def __post_init__(self):
        for name in ("objective_names", "objective_units", "seeds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.objective_names != ("safety", "fuel") or self.objective_units != ("environment_reward",) * 2:
            raise BoundaryError("SCHEMA_MISMATCH", "expected ordered safety/fuel reward coordinates")
        expected = {"environment": "mo-lunar-lander-v3", "objective_mode": "environment_objectives",
                    "reward_mapping": "terminal_plus_shaping__negative_engine_costs_v1",
                    "scalarization": "sum_safety_fuel_v1", "version": "lander_measurement_v1",
                    "aggregation": "matched_seed_arithmetic_mean_population_std_v1",
                    "reset_protocol": "env_reset_and_fresh_seeded_policy_per_episode_v1"}
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise BoundaryError("SCHEMA_MISMATCH", "unsupported measurement semantics")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise BoundaryError("SCHEMA_MISMATCH", "invalid discount")
        if type(self.horizon) is not int or self.horizon <= 0:
            raise BoundaryError("SCHEMA_MISMATCH", "horizon must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise BoundaryError("SCHEMA_MISMATCH", "unique nonnegative episode seeds required")

    @property
    def identity(self):
        return digest(asdict(self))

    def require(self, other):
        if asdict(self) != asdict(other):
            raise BoundaryError("SCHEMA_MISMATCH", "measurement contracts differ")


def validate_action(action):
    if not isinstance(action.id, str) or not action.id.strip():
        raise BoundaryError("INVALID_ACTION", "nonempty action identity required")
    vector(action.goal_correlations, 8, "INVALID_ACTION")
    vector(action.delta_g, 8, "INVALID_ACTION")
    vector([action.risk_estimate], 1, "INVALID_ACTION", unit=True)


def action_record(action):
    validate_action(action)
    return {"id": action.id, "goal_correlations": action.goal_correlations.tolist(),
            "risk_estimate": float(action.risk_estimate), "delta_g": action.delta_g.tolist()}
