# MetaMo STEU benchmark

Evaluate MetaMo appraisal and answer selection on the same **36 emotion-choice questions**, adapting [Croissant et al. (2024), Experiment 1](https://doi.org/10.1371/journal.pone.0301033). Six event-choice questions are excluded. Results report both the source and provisional CSV-compatible scoring keys.

## Run

From the repository root, install `requirements.txt` and set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`.

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

Runs save settings, responses, summaries, and plots in `benchmarks/results/<timestamp>/`. Set `RESUME_DIRECTORY` in [steu_experiment.py](steu_experiment.py) to continue an interrupted run. 

## Settings

Edit [config.py](config.py). Its current settings are:

| Setting | Current value | Purpose |
| --- | --- | --- |
| `CONDITIONS` | `no_memory`, `memory`, `appraisal_prompting`, `full_metamo` | Systems to evaluate. |
| `SELECTOR` | `magus_with_emotion` | Answer-selection rule for MetaMo. |
| `COMPARE_SELECTORS` | `True` | Run all six selectors; `False` runs only `SELECTOR`. |
| `FEELING_SCORER` | `normalized` | Method for calculating emotion scores. |
| `FEELING_WEIGHT` | `1.0` | Feeling bonus weight in combined selection; `0` disables it. |
| `STATE_POLICY` | `independent` | Reset goals and modulators per question; `persistent` carries them forward. |
| `HISTORY_POLICY` | `full` | Show full history; alternatives are `answers_only` and `none`. |

State, history, and feeling settings apply to `full_metamo`. State reset preserves history. Comparing selectors keeps the scorer, state policy, and history policy fixed.

`SELECTORS`, `FEELING_SCORERS`, `STATE_POLICIES`, and `HISTORY_POLICIES` list the supported options. Main feeling scorers are `legacy`, `normalized`, and `prototype_distance`; the remaining variants isolate individual scoring changes.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [config.py](config.py) | Select systems, answer rules, feeling scorers, and state/history policies. |
| [steu_experiment.py](steu_experiment.py) | Run or resume experiments, manage state and history, and save results. |
| [helpers.py](helpers.py) | Build prompts, validate responses, and handle Gemini requests and retries. |
| [selection.py](selection.py) | Execute the configured answer-selection and tie-breaking rule. |
| [feeling_models.py](feeling_models.py) | Provide alternative emotion readouts and isolated scoring variants. |
| [extended_feelings.py](extended_feelings.py) | Define additional emotion templates and legacy/normalized feeling scores. |
| [steu_data.py](steu_data.py) | Prepare questions and scoring keys, and audit saved results. |
| [generate_plots.py](generate_plots.py) | Generate comparison plots, accuracy tables, and selection summaries. |
| [diagnostics.py](diagnostics.py) | Analyze emotion pairs, similar representations, and saved-input ablations offline. |

Diagnostics run when explicitly invoked and make no API requests. The state grid is finite; saved-input replays do not measure Gemini's responses to changed prompts or history.
