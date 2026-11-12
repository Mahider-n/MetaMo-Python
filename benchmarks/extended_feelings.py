"""Experimental MetaMo feelings expressed in the upstream OpenPsi template format.

The 17 templates below extend the nine feelings in openpsi feelings. They reuse
its level thresholds, fuzzy width, score(), and best-of-templates aggregation;
they are authored hypotheses, not upstream OpenPsi definitions or fitted STEU
Each new template specifies three other dimensions and uses divisor 4, giving the same
0.75 ceiling as eight native feelings. 

The comments below explain the qualitative hypotheses behind the chosen levels.
"""

from core.config import M_AROUSAL, M_RESOLUTION, M_SECURING, M_THRESHOLD, M_VALENCE
from core.state import MotivationalState
from openpsi.feelings import (
    EH, EL, H, L, MID, U,
    TEMPLATES as OPENPSI_TEMPLATES,
    feelings as openpsi_feelings,
    score,
)


FEELING_MODEL = "openpsi-plus-metamo-templates-v1"

# (levels for activation, resolution, securing, selection, pleasure), divisor.

# Every distinct label has its own effective signature; grammatical variants are normalized by the benchmark.
TEMPLATES = {
    # Positive anticipation: moderate activation/selection and high pleasure.
    # The moderate levels distinguish it from native happiness's high levels;
    "hope": [((MID, U, U, MID, H), 4)],
    # Settling after tension: low activation/selection and high pleasure.
    "relief": [((L, U, U, L, H), 4)],
    # An orienting response: very high activation and intermediate pleasure/
    # selection. This prototype approximates neutral surprise, not every
    # pleasant or unpleasant surprise.
    "surprise": [((EH, U, U, MID, MID), 4)],
    # Activated discomfort: high activation, low securing and low pleasure.
    "distress": [((H, U, L, U, L), 4)],
    # Withdrawn, cautious unpleasant affect: low activation, high selection,
    # low pleasure. 
    "shame": [((L, U, U, H, L), 4)],
    # More activated, strongly unpleasant caution than the shame prototype.
    "guilt": [((MID, U, U, H, EL), 4)],
    # Low activation, very low pleasure and intermediate selection approximate
    # subdued dissatisfaction. 
    "regret": [((L, U, U, MID, EL), 4)],
    # Activated uncertainty/caution: high activation/selection and low securing.
    "confusion": [((H, U, L, H, U), 4)],
    # Activated, strongly unpleasant insecurity. Its lower securing distinguishes
    # it from native anger.
    "jealousy": [((H, U, L, U, EL), 4)],

    # Intermediate selection distinguishes blocked-goal frustration from native
    # anger's low selection while retaining high activation/very low pleasure.
    "frustration": [((H, U, U, MID, EL), 4)],
    # A milder annoyance signature: intermediate activation and low pleasure,
    # rather than anger's high activation and extremely low pleasure.
    "irritation": [((MID, U, U, L, L), 4)],
    # An intensity hypothesis: raise anger's activation target to extremely high
    # while retaining its low selection and extremely low pleasure.
    "rage": [((EH, U, U, L, EL), 4)],
    # Stronger withdrawal than the native sadness template: extremely low
    # activation, selection and pleasure. 
    "depression": [((EL, U, U, EL, EL), 4)],
    # Moderate activation with low securing and high selection models sustained
    # caution. It differs from confusion's higher activation.
    "anxiety": [((MID, U, L, H, U), 4)],
    # An acute threat signature: high activation, extremely low securing and
    # pleasure. This differs from both native fear and the anxiety prototype.
    "horror": [((H, U, EL, U, EL), 4)],
    # Quiet aversion: low activation/pleasure with high securing. This avoids
    # assigning mild dislike native hate's extremely activated signature.
    "dislike": [((L, U, H, U, L), 4)],
    # A steadier, more negative aversion signature than dislike: intermediate
    # activation, high securing, very low pleasure. 
    "contempt": [((MID, U, H, U, EL), 4)],
}

FEELING_NAMES = frozenset(OPENPSI_TEMPLATES) | frozenset(TEMPLATES)


def additional_feelings(state: MotivationalState) -> dict[str, float]:
    """Score the 17 extensions with exactly the original fuzzy scoring engine."""
    m = state.M
    values = (m[M_AROUSAL], m[M_RESOLUTION], 1.0 - m[M_SECURING], m[M_THRESHOLD], m[M_VALENCE])
    return {name: max(score(values, levels, n) for levels, n in templates) for name, templates in TEMPLATES.items()}


def all_feelings(state: MotivationalState) -> dict[str, float]:
    """Return nine unchanged upstream scores plus the experimental extensions."""
    return {**openpsi_feelings(state), **additional_feelings(state)}
