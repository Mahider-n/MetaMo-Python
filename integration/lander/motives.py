"""Interpretable Phase 1 projection; no MDN or learned outcome claims."""
from dataclasses import dataclass
from core.config import G_IND, G_TRANS, G_ETHIC, G_SELF, M_SECURING, M_RESOLUTION
from .contracts import budget, digest, vector, weights

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

    def __post_init__(self):
        object.__setattr__(self, "context", tuple(vector(self.context, 18)))
        object.__setattr__(self, "weights", weights(self.weights))
        budget(self.epsilon)
        if self.context_id != digest(list(self.context)) or self.region != "FULL_SIMPLEX":
            raise ValueError("Invalid Phase 1 geometry snapshot")


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
