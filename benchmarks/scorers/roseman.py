"""Roseman (2001) readout: feelings scored from seven appraisal ratings that Gemini returns in the same call.

This is a FEELING_SCORER value, not a modulator readout: it reads the ratings in the response instead of the motivational
state. The table is Figure 4.1 of Roseman (2001) (Roseman, Antoniou and Jose, 1996, Figure 2, gives the same rows).
"""

from benchmarks.helpers import bounded, option_feeling

SCORER = "roseman"
MODEL = "roseman-2001-appraisal-v1"
CONDITION = "full_metamo_appraisal"  # Prompt and schema name for full_metamo with ratings; not a CONDITIONS entry.
RATING_DIMS = ("unexpected", "consistent", "certain", "control", "intrinsic", "appetitive")
AGENTS = ("self", "other", "circumstance")

# Targets are 0/1 ratings; "agency" names the cause; a dimension that is not listed is ignored for that emotion.
TABLE = {
    "hope": {"consistent": 1, "certain": 0, "agency": "circumstance"},
    "joy": {"consistent": 1, "certain": 1, "appetitive": 1, "agency": "circumstance"},
    "relief": {"consistent": 1, "certain": 1, "appetitive": 0, "agency": "circumstance"},
    "fear": {"consistent": 0, "certain": 0, "control": 0, "agency": "circumstance"},
    "sadness": {"consistent": 0, "certain": 1, "control": 0, "appetitive": 1, "agency": "circumstance"},
    "distress": {"consistent": 0, "certain": 1, "control": 0, "appetitive": 0, "agency": "circumstance"},
    "frustration": {"consistent": 0, "control": 1, "intrinsic": 0, "agency": "circumstance"},
    "love": {"consistent": 1, "agency": "other"},
    "dislike": {"consistent": 0, "control": 0, "agency": "other"},
    "anger": {"consistent": 0, "control": 1, "intrinsic": 0, "agency": "other"},
    "contempt": {"consistent": 0, "control": 1, "intrinsic": 1, "agency": "other"},
    "pride": {"consistent": 1, "agency": "self"},
    "regret": {"consistent": 0, "control": 0, "agency": "self"},
    "guilt": {"consistent": 0, "control": 1, "intrinsic": 0, "agency": "self"},
    "shame": {"consistent": 0, "control": 1, "intrinsic": 1, "agency": "self"},
    "surprise": {"unexpected": 1},
}
# Feeling name (as option_feeling gives it) -> table row. Gratitude is a kind of love; irritation and rage are anger;
# anxiety is fear. A name that is not here (horror, jealousy, excitement, confusion, depression, hate) has no row.
ROWS = {
    "anger": "anger", "irritation": "anger", "rage": "anger", "anxiety": "fear", "fear": "fear",
    "contempt": "contempt", "dislike": "dislike", "distress": "distress", "frustration": "frustration",
    "gratitude": "love", "love": "love", "happiness": "joy", "hope": "hope", "pride": "pride", "regret": "regret",
    "relief": "relief", "sadness": "sadness", "shame": "shame", "guilt": "guilt", "surprise": "surprise",
}
DEFINITIONS = (
    "unexpected: 0 = the person fully expected it, 1 = it was completely unexpected. "
    "consistent: 0 = it goes against what the person wants, 1 = it is what the person wants. "
    "certain: 0 = it is only possible, and the person cannot tell whether it will happen, "
    "1 = it has happened or is certain to happen. "
    "control: 0 = there is nothing the person can do about it, 1 = there is something the person can do about it. "
    "intrinsic: 0 = the problem is that something stands in the way of a goal, "
    "1 = the problem is what a person or thing is like (their character or nature), apart from any goal. "
    "appetitive: 0 = the person wants to avoid or get rid of something painful, "
    "1 = the person wants to get or keep something pleasant. "
    "agency: who or what caused the event. Three numbers that add up to 1: self (the person caused it), "
    "other (another person caused it), circumstance (circumstances beyond anyone's control, or no one in particular)."
)
INSTRUCTION = (
    "Also return appraisal: seven ratings of the situation in the question, from the point of view of the "
    "person it happens to, using only what the situation says and naming no emotion. Every rating is a "
    "number from 0 to 1. " + DEFINITIONS + " "
)


def parse_rating(data: dict) -> dict:
    """Validate the seven appraisal ratings; the three agency shares must add up to 1."""
    rating = {key: bounded(data[key], 0, 1) for key in RATING_DIMS}
    rating["agency"] = {key: bounded(data["agency"][key], 0, 1) for key in AGENTS}
    if abs(sum(rating["agency"].values()) - 1) > 0.05:
        raise ValueError("Agency shares must add up to 1")
    return rating


def emotion_score(rating: dict, target: dict) -> float:
    """1 minus the mean squared gap between the ratings and the dimensions this emotion specifies."""
    gaps = [(1 - rating["agency"][value]) ** 2 if key == "agency" else (rating[key] - value) ** 2
            for key, value in target.items()]
    return 1 - sum(gaps) / len(gaps)


def score_options(item, rating: dict) -> dict:
    """Letter -> score for every option; an option with no table row gets None."""
    rows = {letter: ROWS.get(option_feeling(text)) for letter, text in item.options.items()}
    return {letter: emotion_score(rating, TABLE[row]) if row else None for letter, row in rows.items()}


def centred(scores: dict) -> dict:
    """Centre the scored options on their mean; an option with no score gets exactly 0."""
    scored = [v for v in scores.values() if v is not None]
    mean = sum(scored) / len(scored) if scored else 0.0
    return {letter: 0.0 if v is None else v - mean for letter, v in scores.items()}
