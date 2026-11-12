"""Port of hyperon-openpsi feeling-updaters.metta (iCog-Labs-Dev/hyperon-openpsi, dev).

Each feeling is a template over (activation, resolution, securing, selection, pleasure).
Every dimension gets a fuzzy score for its level, and the scores are averaged.
Run `python -m openpsi.feelings` to check the port against the upstream tests.
"""

from core.config import M_AROUSAL, M_RESOLUTION, M_SECURING, M_THRESHOLD, M_VALENCE
from core.state import MotivationalState

WIDTH = 50  # fuzzy width used by every call in the .metta
EL, L, MID, H, EH, U = -2, -1, 0, 1, 2, 3  # extremely low .. extremely high, undefined
LEVELS = {EL: ("lt", 0.15), L: ("lt", 0.3), MID: ("eq", 0.5), H: ("gt", 0.7), EH: ("gt", 0.85)}
RESOLUTION = 1  # index of resolution in a template

# (levels for activation, resolution, securing, selection, pleasure), divisor
TEMPLATES = {
    "happiness": [((H, L, U, H, H), 4)],
    "sadness": [((L, H, U, EL, EL), 4)],
    "anger": [((H, L, U, L, EL), 4)],
    "fear": [((EL, EH, L, U, EL), 4)],
    "excitement": [((H, L, U, L, EH), 4), ((H, L, U, L, EL), 4)],  # best of the two
    "pride": [((U, L, H, H, H), 4)],
    "love": [((U, EL, EH, EH, EH), 4)],
    "hate": [((EH, EL, EH, U, EL), 4)],
    "gratitude": [((U, U, U, H, H), 2)],
}


def fuzzy_equal(x: float, t: float) -> float:
    return 1.0 / (1.0 + WIDTH * (x - t) ** 2)


def dimension(x: float, level: int) -> float:
    """Fuzzy score of x for a level; 1 inside the region, falling off outside it."""
    if level not in LEVELS:
        return 0.0  # undefined dimensions score 0 but still count in the divisor, as in the .metta
    kind, t = LEVELS[level]
    if kind == "eq" or (kind == "lt" and x > t) or (kind == "gt" and x < t):
        return fuzzy_equal(x, t)
    return 1.0


def score(values: tuple, levels: tuple, divisor: int) -> float:
    # shortcut: the .metta computes resolution but leaves it out of the sum; ported as-is, revisit after first results
    return sum(dimension(x, l) for i, (x, l) in enumerate(zip(values, levels)) if i != RESOLUTION) / divisor


def feelings(state: MotivationalState) -> dict[str, float]:
    """Score all nine feelings from the modulators of a MetaMo state."""
    m = state.M
    # Securing is inverted: MetaMo's securing rises with risk, Psi's securing threshold is the opposite.
    values = (m[M_AROUSAL], m[M_RESOLUTION], 1.0 - m[M_SECURING], m[M_THRESHOLD], m[M_VALENCE])
    return {name: max(score(values, levels, n) for levels, n in templates) for name, templates in TEMPLATES.items()}


if __name__ == "__main__":
    import numpy as np

    # Expected values copied from hyperon-openpsi/main/mind-agents/feeling-updater/tests (all modulators 0.5).
    expected = {"happiness": 0.25000000000000006, "sadness": 0.15350877192982454, "anger": 0.2017543859649123,
                "hate": 0.10526315789473686, "gratitude": 0.3333333333333334, "excitement": 0.2017543859649123}
    got = feelings(MotivationalState(G=np.full(8, 0.5), M=np.full(6, 0.5)))
    for name, value in expected.items():
        assert abs(got[name] - value) < 1e-9, (name, got[name], value)
    print("feelings port matches the upstream tests")
