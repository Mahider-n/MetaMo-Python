"""Versioned motive projection and validated global/contextual geometry."""
from dataclasses import dataclass
from core.config import G_IND, G_TRANS, G_ETHIC, G_SELF, M_SECURING, M_RESOLUTION
from .contracts import BoundaryError, budget, digest, vector, weights

MAPPING_VERSION = "metamo_safety_fuel_priority_v1"
BUDGET_VERSION = "fixed_global_epsilon_v1"
CONTEXT_SCHEMA = "metamo_context_v1_consensus_post_appraisal"


@dataclass(frozen=True)
class MotiveGeometry:
    context_id: str
    context: tuple
    weights: tuple
    epsilon: float
    mapping_version: str = MAPPING_VERSION
    budget_version: str = BUDGET_VERSION
    context_schema: str = CONTEXT_SCHEMA
    region: str = "FULL_SIMPLEX"
    zero_default_used: bool = False
    alpha: tuple | None = None
    support_values: tuple | None = None
    model_digest: str | None = None
    model_path: str | None = None
    weight_owner: str = "metamo"
    geometry_source: str = "explicit_metamo_mapping"
    fallback_reason: str | None = None
    contract_version: str = "motive_geometry_v2"
    objective_mode: str = "environment_objectives"
    objective_names: tuple = ("safety", "fuel")
    objective_units: tuple = ("environment_reward", "environment_reward")
    objective_schema: str = "safety_fuel_v1"

    def __post_init__(self):
        object.__setattr__(self, "context", tuple(vector(self.context, 18)))
        object.__setattr__(self, "weights", weights(self.weights))
        budget(self.epsilon)
        object.__setattr__(self, "objective_names", tuple(self.objective_names))
        object.__setattr__(self, "objective_units", tuple(self.objective_units))
        if (self.contract_version != "motive_geometry_v2" or self.objective_mode != "environment_objectives"
                or self.objective_names != ("safety", "fuel")
                or self.objective_units != ("environment_reward",) * 2 or self.objective_schema != "safety_fuel_v1"):
            raise BoundaryError("SCHEMA_MISMATCH", "unsupported motive objective schema")
        if self.context_id != digest(list(self.context)) or self.region not in ("FULL_SIMPLEX", "MDN_WX"):
            raise BoundaryError("INVALID_GEOMETRY", "invalid context or region")
        if self.weight_owner not in ("metamo", "mdn"):
            raise BoundaryError("INVALID_GEOMETRY", "unknown weight owner")
        if self.region == "MDN_WX":
            a = vector(self.alpha, 2, "INVALID_GEOMETRY")
            s = vector(self.support_values, 2, "INVALID_GEOMETRY", unit=True)
            if (a <= 0).any() or float(s.sum()) < 1 - 1e-9:
                raise BoundaryError("INVALID_GEOMETRY", "infeasible support or alpha")
            if any(w > cap + 1e-9 for w, cap in zip(self.weights, s)):
                raise BoundaryError("WEIGHT_OUTSIDE_SUPPORT", "current weight is not covered")
            if not self.model_digest or not self.model_path:
                raise BoundaryError("INVALID_GEOMETRY", "contextual geometry requires model identity")
            object.__setattr__(self, "alpha", tuple(float(x) for x in a))
            object.__setattr__(self, "support_values", tuple(float(x) for x in s))
        elif any(v is not None for v in (self.alpha, self.support_values, self.model_digest, self.model_path)):
            raise BoundaryError("INVALID_GEOMETRY", "global geometry cannot claim contextual coverage")


def robust_score(delta_r, delta_n, geometry=None):
    """Native exact support solver; global geometry is the full simplex."""
    if geometry is None or geometry.region == "FULL_SIMPLEX":
        return float(delta_r + min(delta_n))
    from .vendor import enable_subrep
    enable_subrep()
    from utils.support_geometry import worst_case_over_support_region
    return float(delta_r + worst_case_over_support_region(delta_n, geometry.support_values))


def project(state, stimulus, epsilon=0.0):
    g = vector(state.G, 8, unit=True)
    m = vector(state.M, 6, unit=True)
    s = vector([stimulus.novelty, stimulus.conduciveness, stimulus.risk, stimulus.effort], 4, unit=True)
    # Caution, ethics and observed risk prioritize the safety reward proxy.
    safety = g[G_IND] + g[G_ETHIC] + m[M_SECURING] + s[2]
    # Self preservation, deliberate control and effort prioritize fuel economy;
    # transcendence reduces the relative importance of resource conservation.
    fuel = g[G_SELF] + m[M_RESOLUTION] + s[3] + (1 - g[G_TRANS])
    total = safety + fuel
    w = (safety / total, fuel / total) if total else (0.5, 0.5)
    context = tuple(float(x) for x in (*g, *m, *s))
    return MotiveGeometry(digest(list(context)), context, w, budget(epsilon), zero_default_used=bool(total == 0))
