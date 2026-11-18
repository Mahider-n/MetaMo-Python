"""Settings for the STEU benchmark."""

from benchmarks.selection import SELECTORS
from benchmarks.feeling_models import SCORERS as FEELING_SCORERS

# Scale the feeling bonus added to MAGUS; 0 disables it.
FEELING_WEIGHT = 1.0

# Choose how full_metamo ranks answer options.
# Options: emotion_only, magus_only, magus_with_emotion, plausibility,
# emotion_then_plausibility, rank_fusion.
SELECTOR = "magus_with_emotion"

# Compare all selectors; False runs only SELECTOR.
COMPARE_SELECTORS = True

# Choose benchmark systems, including baselines.
CONDITIONS = ("no_memory", "memory", "appraisal_prompting", "full_metamo")

# independent resets goals/modulators per question; "persistent" retains them.
STATE_POLICY = "independent"

# Choose the emotion readout from FEELING_SCORERS.
FEELING_SCORER = "normalized"

# Supported prompt history modes for full_metamo.
HISTORY_POLICIES = ("full", "answers_only", "none")

# Choose history shown in prompts; saved history is always retained.
HISTORY_POLICY = "full"

# Supported motivational state policies.
STATE_POLICIES = ("independent", "persistent")
