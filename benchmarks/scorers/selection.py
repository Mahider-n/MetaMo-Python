"""Execute configured answer-selection rules for live runs and offline diagnostics."""

import math
from benchmarks.steu_data import LETTERS

SELECTORS = ("emotion_only", "magus_only", "magus_with_emotion", "plausibility",
             "emotion_then_plausibility", "rank_fusion")
EMOTION_SELECTORS = {"emotion_only", "emotion_then_plausibility", "rank_fusion"}
TIE_BREAKERS = ("letter", "plausibility")


def top_choices(scores, tolerance=1e-9):
    """Return the highest-scoring options within tolerance, in A–E order."""
    best = max(scores.values())
    return [k for k in LETTERS if k in scores and best - scores[k] <= tolerance]


def select_answer(selector, magus, emotion, plausibility, combined, *, tie_breaker="letter", tolerance=1e-9):
    """Execute the named rule and return the answer, final scores, and tie details."""
    if selector not in SELECTORS or tie_breaker not in TIE_BREAKERS:
        raise ValueError("Unknown selector or tie breaker")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    if selector in EMOTION_SELECTORS and any(v is None for v in emotion.values()):
        raise ValueError("This selector requires all choices to be supported emotion names")
    if selector == "rank_fusion":
        def ranks(scores):
            """Assign higher ranks to higher scores, averaging tied positions."""
            return {k: sum(v < scores[k] - tolerance for v in scores.values()) +
                    0.5 * (sum(abs(v - scores[k]) <= tolerance for v in scores.values()) - 1)
                    for k in LETTERS}
        er, pr = ranks(emotion), ranks(plausibility)
        final = {k: er[k] + pr[k] for k in LETTERS}
    else:
        final = {"emotion_only": emotion, "emotion_then_plausibility": emotion,
                 "magus_only": magus, "magus_with_emotion": combined, "plausibility": plausibility}[selector]
    tied = top_choices(final, tolerance)
    use_plausibility = selector == "emotion_then_plausibility" or tie_breaker == "plausibility"
    answer = max(tied, key=lambda k: plausibility[k]) if use_plausibility else tied[0]
    return answer, final, {"top_choices": tied, "tie_breaker": "plausibility" if use_plausibility else "letter",
                           "tie_tolerance": tolerance, "resolved_by_plausibility": use_plausibility and len(tied) > 1}
