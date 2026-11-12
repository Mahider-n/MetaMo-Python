"""Settings for the STEU benchmark."""

# final_score = magus_score + FEELING_WEIGHT * centred_feeling_score
# Set to 0.0 to disable the feeling score adjustment.
# --feeling-weight overrides this default for a run.
FEELING_WEIGHT = 1.0

# Final selector for full_metamo.
SELECTORS = ("emotion_only", "magus_only", "magus_with_emotion")
SELECTOR = "magus_with_emotion"
