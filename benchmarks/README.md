# Evaluation of MetaMo Appraisal and Emotional Understanding on the STEU Benchmark

This benchmark adapts Experiment 1 of Croissant et al.’s (2024) [An appraisal-based chain-of-emotion architecture for affective language model game agents](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301033), which compared GPT-3.5 with no-memory, memory, and appraisal prompting on the Situational Test of Emotional Understanding (STEU). We intend to evaluate MetaMo’s appraisal component and emotion-informed decisions using nine ported OpenPsi feelings and 17 additional templates for emotions absent from the original feeling implementation. The appraisal-prompting baseline separately tests the paper’s approach: a brief situational appraisal followed by the LLM’s answer, with prior interactions retained (sections 4.1–4.2). It is an adaptation, not an exact prompt replication.

## Question selection

**Six event-choice questions** were excluded from the primary comparison because their answer choices describe events rather than name emotions. The emotion-only selector matches each candidate emotion to a feeling score and cannot directly score event descriptions, so including these questions would disadvantage it and make comparisons unfair. All configured baselines and MetaMo selector variants therefore use the same **36 emotion-choice questions**, answer choices, and scoring keys. The excluded questions remain in the source dataset for separate evaluation and future extensions.

## File responsibilities

| File | Responsibility |
| --- | --- |
| [steu_experiment.py](steu_experiment.py) | Prepare questions, run configured cases, save results, generate graphs, and resume interrupted experiments. |
| [helpers.py](helpers.py) | Build prompts, validate replies, resolve emotion labels, and handle Gemini requests and retries. |
| [steu_data.py](steu_data.py) | Prepare questions and scoring keys, load items, and validate saved results. |
| [generate_plots.py](generate_plots.py) | Validate saved runs and generate PNG comparison graphs, JSON metrics, and question-level CSV data offline. |
| [config.py](config.py) | Configure the feeling weight, supported and selected selectors, selector comparison, and experimental conditions. |
| [extended_feelings.py](extended_feelings.py) | Define 17 additional emotions, preserve legacy scores, and provide a normalized benchmark scorer. |
| [OpenPsi feelings](../openpsi/feelings.py) | Preserve the nine original feeling definitions and fuzzy scoring functions. |
| [tests/test_steu_experiment.py](tests/test_steu_experiment.py) | Check the main functions and workflows offline. |
| [__init__.py](__init__.py) | Identify the benchmark package. |
| [steu_author_source.txt](data/steu_author_source.txt) | Supply the author-provided questions, choices, and source scoring rules. |
| [Experiment 1 CSV](qpt6z-osfstorage-archive/experiment_1_STEU.csv) | Preserve the paper’s published predictions and scores for reference. |
| [Experiment 2 CSV](qpt6z-osfstorage-archive/experiment_2_LIWC_Output.csv), [Experiment 3 CSV](qpt6z-osfstorage-archive/experiment_3_user_study.csv) | Retain published data from the paper’s later experiments. |

Change settings in [config.py](config.py) to experiment with the implementation. COMPARE_SELECTORS=True evaluates all four selectors; False uses only SELECTOR. SELECTOR retains magus_with_emotion as the primary run; plausibility is a separate semantic classification control. CONDITIONS independently controls which systems run, including the baselines. FEELING_WEIGHT controls combined selection and defaults to 1.0. Execution defaults live in steu_experiment.py; Gemini request limits and retries live in helpers.py. The model comes from GEMINI_MODEL in .env.

Each experiment and run saves benchmark and core configuration constants, effective selector settings, question IDs, and dataset fingerprints. Code fingerprints cover the runner, helpers, and feeling definitions. State policy and feeling scorer are recorded explicitly; resuming rejects changes to either. Protocol v8 retains MetaMo history when resetting numerical state. It cannot resume v6 or v7 checkpoints; completed older runs remain available for offline analysis. Resumed experiments retain their saved settings.

## Processing flow

Each condition starts independently. No-memory answers each question alone. Memory retains prior questions and answers. Appraisal prompting retains prior questions, appraisal summaries, and answers, without numerical MetaMo state.

For MetaMo, `STATE_POLICY="independent"` resets only goals and modulators before each STEU scenario. `"persistent"` carries the numerical state forward. Both policies retain previous questions, answers, stimuli, and state/appraisal traces in prompt history. The final state is still logged for each question. Historical traces remain visible to Gemini even when the current numerical state is reset; this policy does not make prompts independent of earlier questions.

Gemini returns a four-feature stimulus and five candidate records. Each candidate includes both the legacy utility inputs and a semantic plausibility; plausibilities must sum to one. The selectors are:

- `plausibility`: choose the largest semantic likelihood directly, independently of Magus and feeling scores. This is a classification control, not evidence that MetaMo's numerical appraisal improved.
- `emotion_only`: choose the highest feeling score.
- `magus_only`: choose the highest Magus utility score.
- `magus_with_emotion`: choose Magus utility plus the centered feeling bonus.

All four rankings are logged from each MetaMo response, enabling comparisons on identical generated inputs. Separate configured selector runs still make independent model requests; do not attribute differences between those runs solely to the selector. Under both state policies, selected answers enter subsequent prompt history. With persistent state, they also affect the next question's numerical state.

`FEELING_SCORER="normalized"` averages over all defined prototype dimensions, including resolution, and removes excitement's negative prototype that duplicated anger. Every prototype has a maximum of 1. `"legacy"` preserves the upstream scores and authored extensions unchanged. Both are hypotheses about emotion recognition; neither is calibrated on STEU. Fuzzy prototypes can still tie, and stable A–E order resolves ties. Saved-result diagnostics expose these ties rather than treating them as confident distinctions. The core OpenPsi port and Magus objective are unchanged.

Offline `analysis.json` now includes `selection_diagnostics`: per-question top emotion margins and ties, predicted label frequencies, same-response Magus/emotion/plausibility picks, and counts of correct Magus answers overturned or incorrect ones rescued under both scoring keys. These measurements identify selection disagreements; they cannot establish whether a stimulus appraisal was psychologically correct.

## Reproduce the comparison

Run without subcommands or experiment flags; choose conditions and selectors in [config.py](config.py).

1. **Set up once:** activate your Python environment, install dependencies with `python -m pip install -r requirements.txt`, and set GEMINI_API_KEY and GEMINI_MODEL in .env.
2. **Configure:** in [config.py](config.py), include no_memory, memory, appraisal_prompting, and full_metamo in CONDITIONS, set COMPARE_SELECTORS=True, and keep FEELING_WEIGHT=1.0 for the seven-case comparison.
3. **Run from the repository root:**

   ```bash
   python -m benchmarks.steu_experiment
   ```

Questions are prepared and PNG graphs generated automatically. Settings, predictions, scores, and PNG/JSON/CSV analysis are saved under **benchmarks/results/&lt;timestamp&gt;/**.

For a fresh experiment, keep RESUME_DIRECTORY=None in steu_experiment.py. To resume an interrupted experiment, set it to the saved folder and run the same command.

To isolate changes, vary one setting at a time: compare `STATE_POLICY` values with legacy feeling scoring, then compare `FEELING_SCORER` values with independent state. Use separate result folders and inspect both scoring keys. The 36-question subset and unresolved source-key disagreements prevent direct comparison with the paper's 42-question percentages. No accuracy improvement follows from passing the offline tests; it requires fresh model runs.

Analyze completed results without API requests:

```bash
python -m benchmarks.generate_plots path/to/run --items path/to/steu_items.jsonl --output /tmp/steu-analysis
```

Run offline regression checks:

```bash
python -m unittest discover -s benchmarks/tests -v
```

The attached issue notes identify several plausible causes; code inspection confirms the state carryover, anger/excitement dominance, ignored resolution, and unequal score ceilings. Confusion and distress already have different effective templates, so their observed ties do not establish duplicate definitions; the new diagnostics quantify ties. The four-feature representation and uncalibrated utility/feeling fusion remain experimental limitations. This patch does not tune weights or change core appraisal coefficients on the evaluated questions.
