"""Unfitted benchmark readouts and isolated scoring comparisons."""

import numpy as np

from benchmarks.extended_feelings import (TEMPLATES as EXTRA_TEMPLATES, all_feelings, normalized_feelings)
from core.config import M_AROUSAL, M_RESOLUTION, M_SECURING, M_THRESHOLD, M_VALENCE
from openpsi.feelings import TEMPLATES as NATIVE_TEMPLATES, H, L, EH, U, LEVELS, WIDTH

SCORERS = ("legacy", "legacy_fixed_excitement", "legacy_with_resolution",
           "normalized_original_excitement", "normalized_no_resolution", "normalized",
           "prototype_distance")


def model_spec(scorer):
    """Select emotion templates, resolution inclusion, and normalization rules."""
    if scorer not in SCORERS:
        raise ValueError(f"Unknown feeling scorer: {scorer}")
    templates = {**NATIVE_TEMPLATES, **EXTRA_TEMPLATES}
    if scorer not in {"legacy", "legacy_with_resolution", "normalized_original_excitement"}:
        templates["excitement"] = [((H, L, U, L, EH), 4)]
    include_resolution = scorer not in {"legacy", "legacy_fixed_excitement", "normalized_no_resolution"}
    normalize = scorer.startswith("normalized") or scorer == "prototype_distance"
    return templates, include_resolution, normalize


def state_values(state):
    """Arrange five modulator values in template order, inverting securing."""
    m = state.M
    return np.array([m[M_AROUSAL], m[M_RESOLUTION], 1 - m[M_SECURING],
                     m[M_THRESHOLD], m[M_VALENCE]])


def score_values(values, scorer):
    """Score emotions for one or many five-value vectors using the chosen readout."""
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != 5 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected finite template values in [0, 1] with five dimensions")
    templates, include_resolution, normalize = model_spec(scorer)
    result = {}
    for name, prototypes in templates.items():
        matches = []
        for levels, divisor in prototypes:
            defined = [(i, level) for i, level in enumerate(levels)
                       if level != U and (include_resolution or i != 1)]
            total = np.zeros(values.shape[:-1])
            for i, level in defined:
                kind, threshold = LEVELS[level]
                x = values[..., i]
                distance = (np.maximum(x - threshold, 0) if kind == "lt" else
                            np.maximum(threshold - x, 0) if kind == "gt" else abs(x - threshold))
                total += distance ** 2 if scorer == "prototype_distance" else 1 / (1 + WIDTH * distance ** 2)
            denominator = len(defined) if normalize else divisor
            value = total / denominator
            matches.append(1 - value if scorer == "prototype_distance" else value)
        result[name] = np.maximum.reduce(matches)
    return result


def score_feelings(state, scorer="normalized"):
    """Return emotion scores for one motivational state using the chosen readout."""
    if scorer == "legacy":
        return {name: float(value) for name, value in all_feelings(state).items()}
    if scorer == "normalized":
        return {name: float(value) for name, value in normalized_feelings(state).items()}
    return {name: float(value) for name, value in score_values(state_values(state), scorer).items()}
