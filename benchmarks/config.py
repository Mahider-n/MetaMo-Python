"""Choose STEU systems, emotion scorers, answer rules, and state/history policies."""

from benchmarks.scorers.feelings import SCORERS as FEELING_SCORERS
from benchmarks.scorers.selection import SELECTORS as SELECTORS

# Any finite number >= 0; used only by magus_with_emotion.
# 0 disables the feeling bonus; larger values increase its influence.
FEELING_WEIGHT = 1.0

# Answer rule for full_metamo; choose one:
# emotion_only: highest emotion score among the five options.
# magus_only: highest MAGUS utility score.
# magus_with_emotion: MAGUS score plus the weighted, centered emotion score.
# plausibility: highest semantic plausibility from the question-aware LLM.
# emotion_then_plausibility: emotion scores first; plausibility breaks top ties.
# rank_fusion: add emotion and plausibility ranks with equal weight.
SELECTOR = "magus_with_emotion"

# True runs every selector with separate responses; False uses SELECTOR only.
# This setting affects full_metamo; CONDITIONS still controls baseline runs.
COMPARE_SELECTORS = True

# Choose one or more systems; a single-entry tuple needs a trailing comma.
# no_memory: LLM answers each question without previous interactions.
# memory: LLM answers with previous questions and generated answers.
# full_metamo: LLM extracts inputs; MetaMo appraises; SELECTOR chooses an answer.
CONDITIONS = ("no_memory", "memory", "full_metamo")

# Numerical state policy for full_metamo; choose one:
# independent: reset goals and all six modulators before each question.
# persistent: carry the updated state into the next question.
STATE_POLICY = "independent"

# Emotion scorer for full_metamo; choose one:
# legacy: original templates/divisors, skipped resolution, duplicated excitement profile.
# legacy_fixed_excitement: legacy scoring with the duplicated profile removed.
# legacy_with_resolution: original templates/divisors with resolution included.
# normalized_original_excitement: average defined dimensions, including resolution;
#                                 retain the duplicated excitement profile.
# normalized_no_resolution: average defined dimensions and fix excitement; skip resolution.
# normalized: average defined dimensions, include resolution, and fix excitement.
# prototype_distance: mean squared distance to template regions, converted to similarity.
# quadratic: signed quadratic equations using all six raw modulators, without fuzzy matching. They were fitted to STEU answers; those results are development results.
# roseman: feelings from seven appraisal ratings (Roseman 2001) that the model returns in the same response,
#          scored against a fixed table instead of the modulators; see scorers/roseman.py.
FEELING_SCORER = "roseman"

# Set to a saved experiment folder to continue an interrupted run; None starts a fresh experiment.
# The other settings are then read from the saved run. When the resumed run finishes, this line is set
# back to None for you (a run that crashes again leaves it as it is).
RESUME_DIRECTORY = None

# Supported history options; select one through HISTORY_POLICY below.
HISTORY_POLICIES = ("full", "answers_only", "none")

# History shown to the full_metamo extraction prompt; choose one:
# full: previous questions, answers, stimuli, and numerical states.
# answers_only: previous questions and answers.
# none: no previous interactions in the prompt.
# memory retains its own full interaction history.
HISTORY_POLICY = "full"

# Supported state options; select one through STATE_POLICY above.
STATE_POLICIES = ("independent", "persistent")
