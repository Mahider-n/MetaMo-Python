"""Shared state, candidate parsing, label mapping, and provenance helpers."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from benchmarks import config as benchmark_config
from benchmarks.scorers.templates import FEELING_NAMES
from benchmarks.steu_data import Item, LETTERS, sha256
from core import config as core_config
from core.config import NUM_MODULATORS
from core.state import Action, MotivationalState, Stimulus


EMOTION_BASE = {
    "angry": "anger", "sad": "sadness", "depressed": "depression", "irritated": "irritation", "frustrated": "frustration",
    "contemptuous": "contempt", "ashamed": "shame", "guilty": "guilt", "hopeful": "hope", "relieved": "relief",
    "surprised": "surprise", "distressed": "distress", "confused": "confusion",
}

FEELING_ALIASES = {"joy": "happiness", "happy": "happiness", "scared": "fear", "frightened": "fear"}


def option_feeling(label: str) -> str | None:
    """Normalize an emotion label and resolve its supported template."""
    word = label.strip().lower()
    word = EMOTION_BASE.get(word, word)
    name = FEELING_ALIASES.get(word, word)
    return name if name in FEELING_NAMES else None


def supports_emotion_selection(item: Item) -> bool:
    """Whether every choice has a feeling template, so raw emotion scores can rank them."""
    return all(option_feeling(text) is not None for text in item.options.values())


def initial_state() -> MotivationalState:
    """Create the default goal weights and neutral affect."""
    return MotivationalState(
        G=np.array([0.5, 0.5, 0.8, 0.6, 0.4, 0.3, 0.9, 0.2]),
        M=np.full(NUM_MODULATORS, 0.5),
    )


def snapshot(state: MotivationalState | None) -> dict | None:
    """Copy a state into lists for prompts, history, and JSON logs."""
    if state is None:
        return None
    return {"G": state.G.tolist(), "M": state.M.tolist()}


def config_snapshot() -> dict:
    """Copy public configuration defaults for logging, separate from run overrides."""
    settings = {
        module.__name__: {name: value for name, value in vars(module).items()
                          if name.isupper() and not name.startswith("_")}
        for module in (benchmark_config, core_config)
    }
    return json.loads(json.dumps(settings, allow_nan=False))


def bounded(value, lower: float, upper: float) -> float:
    """Validate a finite numeric value within inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Expected a numeric value")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Expected a finite number in [{lower}, {upper}]")
    return value


def vector(values, lower: float, upper: float) -> np.ndarray:
    """Validate and convert eight bounded goal values to an array."""
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError("Expected exactly eight goal dimensions")
    return np.array([bounded(value, lower, upper) for value in values])

def parse_plan(data: dict) -> list[Action]:
    """Validate five answer candidates and return actions in A–E order."""
    candidates = data["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("Expected all five answer candidates")
    actions = [Action(
        id=c["id"], risk_estimate=bounded(c["risk_estimate"], 0, 1),
        goal_correlations=vector(c["goal_correlations"], -1, 1),
        delta_g=vector(c["delta_g"], -0.1, 0.1),
    ) for c in candidates]
    if {a.id for a in actions} != set(LETTERS):
        raise ValueError("Candidate IDs must be A, B, C, D and E exactly once")
    return sorted(actions, key=lambda a: a.id)

def parse_plausibility(data: dict) -> dict[str, float]:
    """Validate task-correctness probabilities without utility transformations."""
    values = {c["id"]: bounded(c["plausibility"], 0, 1) for c in data["candidates"]}
    if set(values) != set(LETTERS):
        raise ValueError("Expected A–E plausibilities summing to 1")
    total = sum(values.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Expected A–E plausibilities summing to 1; got total {total:.12g}")
    return values


def parse_stimulus(data: dict) -> Stimulus:
    """Validate appraisal features and construct a stimulus."""
    return Stimulus(**{
        key: bounded(data[key], -1 if key == "conduciveness" else 0, 1)
        for key in ["novelty", "conduciveness", "risk", "effort"]
    })


def feeling_bonus(item: Item, scores: dict) -> dict[str, float]:
    """Center supported options' emotion scores; assign zero to unsupported options."""
    felt = {letter: scores[option_feeling(text)] for letter, text in item.options.items() if option_feeling(text)}
    mean = sum(felt.values()) / len(felt) if felt else 0.0
    return {letter: felt[letter] - mean if letter in felt else 0.0 for letter in LETTERS}


def benchmark_code_hashes() -> dict[str, str]:
    """Fingerprint benchmark modules for reproducibility."""
    directory = Path(__file__).parent
    modules = {
        "benchmark_code_sha256": directory / "steu_experiment.py",
        "benchmark_helpers_sha256": Path(__file__),
        "benchmark_feelings_sha256": directory / "scorers/templates.py",
        "feeling_models_sha256": directory / "scorers/feelings.py",
        "selection_sha256": directory / "scorers/selection.py",
        "emotion_formulas_sha256": directory / "scorers/quadratic.py",
        "benchmark_llm_sha256": directory / "scorers/llm.py",
    }
    hashes = {name: sha256(path) for name, path in modules.items()}
    from benchmarks.scorers.quadratic import formula_coefficients

    coefficients = json.dumps(formula_coefficients().tolist(), separators=(",", ":"))
    hashes["emotion_formula_coefficients_sha256"] = hashlib.sha256(coefficients.encode()).hexdigest()
    return hashes
