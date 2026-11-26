# MetaMo–SubRep Integration

Connects MetaMo's motivational system to SubRep's controller certification and
reuse pipeline in **MO-LunarLander**. SubRep is pinned as a Git submodule under
`external/subrep`.

## How it works

1. Evaluate controllers against an idle baseline using matching seeds and settings.
2. Certify measured improvements in **Safety** and **Fuel conservation** with
   SubRep's CDS/PDS gates, then save the evidence and controller identity.
3. Convert MetaMo's motivation into objective weights. An optional trained MDN
   supplies contextual weight-region limits.
4. Screen controllers against the current requirements. Either MAGUS or SubRep
   makes the final selection, depending on the configured selection owner.
5. Execute the selected controller and feed its observed outcome into the next
   appraisal. If none is eligible, use an approved, applicable backup or stop.

Rejected controllers retain their evidence. A persistent scheduler controls
reconsideration using decaying penalties, cooldowns and retry budgets.

## Setup

Run these commands from the **repository root**, using Python 3.12 on Linux.


```bash
git submodule update --init --recursive
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install swig
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lander.txt
```

## Run

Global certification and reuse:

```bash
python -m integration.lander --output artifacts/lander-run
python -m integration.lander --output artifacts/lander-run --resume
```

Complete experiment, including MDN training, held-out validation, both selection
modes and recovery/reevaluation probes:

```bash
python -m integration.lander.complete --output artifacts/complete-run
```

Use a fresh output directory for each complete experiment. Outputs include JSON
reports, raw evaluation evidence, certificates, model files and execution traces.

## Tests

Tests live in `integration/tests/`, the folder `integration/scripts/validate_lander.py` runs.

```bash
python -m pytest integration/tests -q                  # fast, about 8 seconds, no simulator
python -m pytest integration/tests -q --run-pipeline   # also the real simulator, about 45 seconds
```

Tests marked `pipeline` are skipped unless `--run-pipeline` is passed (defined in
`integration/tests/conftest.py`). Skip reasons are always printed in the summary.

## Code layout

- `lander/coordinator.py`: screening, selection, execution and feedback.
- `lander/motives.py`, `contextual.py`: objective weights and contextual geometry.
- `lander/registry.py`, `evidence.py`: executable identities and rollout evidence.
- `lander/storage.py`, `persistence.py`: certificates and durable storage.
- `lander/scheduler.py`, `reviews.py`: bounded reevaluation and replacement review.
- `lander/recovery.py`, `deadlines.py`: backup applicability and decision watchdog.
- `scripts/`: validation and document-generation utilities. 

Certification describes measured baseline-relative evidence, not a guarantee for every trajectory.