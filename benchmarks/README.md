# Phase 1: STEU experiment

This implements the three module configurations described in the supplied MetaMo benchmarking guideline. The downloaded `experiment_1_STEU.csv` is a table of previously recorded model answers and scores, not the original test input. Its `Output` column contains answers and must never be used as a fresh question.

The reference audit is saved in [results/reference_audit.md](results/reference_audit.md). The recorded totals are 24/42, 31/42, and 35/42. These are the paper's results, not measurements of this repository.

The experiment archive is stored in [qpt6z-osfstorage-archive/](qpt6z-osfstorage-archive/), including all three CSV files. Phase 1 reads `qpt6z-osfstorage-archive/experiment_1_STEU.csv` from this folder by default.

## What is ready

The 42 original questions and five choices have been extracted from the [author-uploaded instrument](https://www.researchgate.net/publication/233864930_MacCann_Roberts_STEU). Retrieved test text, prepared items, and provenance hashes are stored in [data/](data/). These supplement the copied CSV, which contains published answers and scores but does not contain the original questions and choices needed for a fresh experiment. No reference answers or published outputs enter model prompts or interaction memory.

| Configuration | Memory | OpenPsi appraisal | Final answer |
| --- | --- | --- | --- |
| `no_memory` | None; every question starts independently | Bypassed; no motivational state is constructed | Gemini returns one letter |
| `memory` | Previous questions, selected answers, and states | Bypassed; affect stays neutral | Gemini proposes five candidates; Magus selects a letter and goals update |
| `full_metamo` | Previous questions, selected answers, appraisals, and states | Gemini extracts a stimulus; OpenPsi computes modulator changes | Existing pseudo-bimonad runs appraisal, Magus choice, and stabilization; resulting state is blended and retained |

Every condition starts independently. Memory accumulates across the fixed numerical item order within the two memory conditions; no gold-answer feedback is given. Both use the same default goal vector and neutral modulators. All use the same selected Gemini model, temperature, seed, and welcome-gift example. One successful generation is requested per item per condition: 126 for a complete run. Each item allows up to three attempts for transient service errors, malformed/truncated JSON, or invalid answer fields. Every retry sends the same question, history, schema, and settings, and failed attempts never advance state or memory. Persistent failures stop the run; no heuristic or neutral-response fallback is scored.

Protocol v2 supplies a condition-specific JSON Schema to Gemini and validates parsed answers locally. Its output-token limit is 16,384 for all conditions, increased from v1's 8,192 to leave more space for thinking and the answer JSON. These settings are recorded in the manifest; v1 run folders are preserved.

## Actual processing pipelines
flowchart TD
    Q["Prepared question and five choices"]

    Q --> N["No memory: question + fixed example"]
    N --> NG["Gemini returns one answer letter"]

    Q --> M["Memory: question + history + current state"]
    M --> MG["Gemini proposes five candidates"]
    MG --> MD["Magus chooses using current state"]
    MD --> MU["Update goals and blend; affect stays neutral"]
    MU --> MH["Store question, selected answer and resulting state"]

    Q --> A["Appraisal + memory: question + history + current state"]
    A --> AG["Gemini proposes stimulus and five candidates"]
    AG --> OA["OpenPsi computes affect changes"]
    OA --> AD["Magus chooses using appraised state"]
    AD --> AU["Stabilize and blend the resulting state"]
    AU --> AH["Store answer, stimulus and state snapshots"]

    NG --> S["Score locally against both keys and write logs"]
    MH --> S
    AH --> S

## Run locally

Use the existing virtual environment. Put the Gemini key in the project `.env` file using `.env.example` as a template, or set `GEMINI_API_KEY` in the process environment. Do not put the key in a command argument or commit it.

Audit the downloaded results without API calls:

```bash
venv/bin/python -m benchmarks.phase1 audit
```

Try a two-item pilot across all three conditions, then the complete test:

```bash
venv/bin/python -m benchmarks.phase1 run --limit 2
venv/bin/python -m benchmarks.phase1 run
```

The default is `gemini-3-flash-preview`, matching the existing application's model setting. Select another available Gemini model with `--model MODEL_ID`. Temperature defaults to 0 and the seed to 42; these settings do not guarantee identical responses across service versions. A two-item pilot is not a 42-item STEU result.

Each new run produces `manifest.json`, `responses.jsonl`, and, if every call succeeds, `summary.json` in a new `benchmarks/results/runs/` directory. Logs include the exact prompts, raw model answers, usage metadata, model version, item order, states, candidate scores, selected letters, and both scoring variants. Response metadata also records finish reasons and failed attempts preceding a valid response. Failed runs retain completed responses, a failure manifest, and `failure.json` identifying the failed item and any rejected Gemini replies; they do not produce a success summary. Existing run directories are never overwritten.

If the original files move, pass `--reference PATH` to `audit` or `prepare`. To rebuild the prepared inputs from the locally saved instrument:

```bash
venv/bin/python -m benchmarks.phase1 prepare
```

Offline verification, using invented scenarios and synthetic model outputs:

```bash
python -m unittest benchmarks.tests.test_phase1 -v
```

These checks verify module bypasses, memory isolation, actual OpenPsi/Magus execution, answer-key exclusion, parsing, and failure logging. They do not measure emotional intelligence.

## How to interpret results

The [original study, Experiment 1](https://doi.org/10.1371/journal.pone.0301033) compared three GPT-3.5 prompting strategies, including memory and appraisal prompting. It did not test MetaMo's mathematical modules. This benchmark adapts the supplied guideline to the existing single-state OpenPsi/Magus pipeline. It does not invoke the research assistant's action vocabulary, response-generation stage, or dual-subsystem consensus.

The multiple-choice adapter asks Gemini to supply candidate goal correlations, goal updates, and answer-error risk. Magus scores these candidates; it has no existing native mapping from motivational dimensions to STEU emotions. This mapping is an experimental interface, and its quality may limit results. The no-memory condition uses a direct answer while the other two use candidate planning, and the full condition additionally requests stimulus features. Differences therefore reflect the complete configurations, including prompt and response-format differences; they do not isolate a pure causal effect of the appraisal equations.

Scoring remains provisional. The author's uploaded SPSS rules disagree with some recorded CSV scores and accept two answers for items 25 and 40. Source rules are preserved literally, without silently fixing them. Fresh runs report:

- `source_*`: scores from the literal numeric SPSS rules.
- `reference_compatible_*`: a sensitivity score using the 37 positively scored letters in the CSV, plus source-rule answers consistent with negative CSV scores for the five remaining items. This key reproduces the recorded flags but is not independently validated.

Do not compare either variant uncritically with the published 35/42. Resolve the source-key discrepancies before treating results as a definitive reproduction. A single 42-item run can suggest differences, but does not establish reliable superiority. Repeat across seeds and item orders for that question. Public test items may also occur in model training data.

The first live two-item v1 pilot completed five of six item responses, then stopped on invalid JSON. That incomplete run remains in `results/runs/`; it is not a complete pilot score. Protocol v2 addresses JSON formatting and failure diagnostics.
