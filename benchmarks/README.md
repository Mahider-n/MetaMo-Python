# MetaMo STEU benchmark

Evaluate MetaMo appraisal and answer selection on the same **36 emotion-choice questions**, adapting [Croissant et al. (2024), Experiment 1](https://doi.org/10.1371/journal.pone.0301033). Six event-choice questions are excluded. Results report both the source and provisional CSV-compatible scoring keys.

## Run

From the repository root, install `requirements.txt` and set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`.

Edit [config.py](config.py) to choose systems, selectors, emotion scorers, and state/history policies. Its comments describe every option. Save your changes and run the experiment again to see their effect; change one setting at a time for easier comparison.

```bash
# Run the configured experiment: makes API requests.
venv/bin/python -m benchmarks.steu_experiment

# Test the implementation offline.
venv/bin/python -m unittest discover -s benchmarks/tests -q

# Analyze saved results offline; choose a new output folder.
venv/bin/python -m benchmarks.diagnostics \
  --run benchmarks/results/20261115T100554052705Z/primary \
  --output benchmarks/results/diagnostics/my-check
```

Each fresh run creates `benchmarks/results/<timestamp>/`:

- `plots/`: accuracy graphs, `analysis.json`, and question-level `question_results.csv`.
- Run subfolders such as `primary/`: `summary.json` and detailed `responses.jsonl`.

To continue an interrupted run, set `RESUME_DIRECTORY` in [config.py](config.py). Leave it as `None` to start a fresh experiment with your configuration changes. When the resumed run finishes, the runner sets it back to `None` for you; a run that crashes again leaves it set, so you can run the same command again.

## Settings

Current defaults:

| Setting | Current value | Purpose |
| --- | --- | --- |
| `CONDITIONS` | `no_memory`, `memory`, `full_metamo` | Systems to evaluate. |
| `SELECTOR` | `magus_with_emotion` | Answer-selection rule for MetaMo. |
| `COMPARE_SELECTORS` | `True` | Run all six selectors; `False` runs only `SELECTOR`. |
| `FEELING_SCORER` | `normalized` | Method for calculating emotion scores. |
| `FEELING_WEIGHT` | `1.0` | Feeling bonus weight in combined selection; `0` disables it. |
| `STATE_POLICY` | `independent` | Reset goals and modulators per question; `persistent` carries them forward. |
| `HISTORY_POLICY` | `full` | Show full history; alternatives are `answers_only` and `none`. |

`FEELING_SCORER = "roseman"` reads feelings from seven appraisal ratings (Roseman 2001) that the model returns in the same response, scored against a fixed table, instead of from the modulators. Only `emotion_only`, `magus_with_emotion`, `emotion_then_plausibility` and `rank_fusion` ask for the ratings; `magus_only` and `plausibility` send the same prompt as before. An option the table has no row for (horror, jealousy, excitement, confusion, depression, hate) has no score, is never picked by an emotion rule, and adds 0 to the combined score. It is not in `SCORERS`, so [diagnostics.py](diagnostics.py) does not replay it.

State, history, and feeling settings apply to `full_metamo`. State reset preserves history. Comparing selectors keeps the scorer, state policy, and history policy fixed.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [config.py](config.py) | Choose systems, selectors, emotion scorers, and state/history policies. |
| [steu_experiment.py](steu_experiment.py) | Run or resume experiments, manage state and history, save results, and generate plots. |
| [helpers.py](helpers.py) | Parse candidates and stimuli, map emotion labels, snapshot state, and record code fingerprints. |
| [scorers/llm.py](scorers/llm.py) | Build LLM prompts, validate responses, and handle Gemini requests and retries. |
| [scorers/feelings.py](scorers/feelings.py) | Register emotion scorers and evaluate template or quadratic scores. |
| [scorers/roseman.py](scorers/roseman.py) | The `roseman` emotion scorer: appraisal ratings from the model, scored against Roseman's table. |
| [scorers/quadratic.py](scorers/quadratic.py) | Store fitted coefficients and calculate emotion scores from six modulators. |
| [scorers/templates.py](scorers/templates.py) | Define additional emotion templates and legacy/normalized reference scorers. |
| [scorers/selection.py](scorers/selection.py) | Apply answer-selection rules, tie-breaking, and rank fusion. |
| [steu_data.py](steu_data.py) | Prepare questions and scoring keys, validate datasets, and audit results. |
| [generate_plots.py](generate_plots.py) | Generate accuracy plots, correctness heatmaps, selection summaries, and question-level CSV files. |
| [diagnostics.py](diagnostics.py) | Analyze competing emotion pairs, representation overlap, and saved-input scoring/state comparisons offline. |
