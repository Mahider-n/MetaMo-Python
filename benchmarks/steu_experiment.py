"""Run the configured STEU comparison and generate graphs automatically."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmarks import config as benchmark_config
from benchmarks.config import FEELING_WEIGHT, SELECTOR, SELECTORS
from benchmarks.steu_data import CONDITIONS, Item, LETTERS, load_items, prepare, sha256
from benchmarks.helpers import (
    benchmark_code_hashes,
    bounded,
    config_snapshot,
    feeling_bonus,
    initial_state,
    option_feeling,
    parse_plan,
    parse_plausibility,
    parse_stimulus,
    snapshot,
    supports_emotion_selection,
    vector,
)
from benchmarks.scorers.llm import (
    MAX_OUTPUT_TOKENS,
    MAX_RESPONSE_ATTEMPTS,
    REQUEST_TIMEOUT_MS,
    GeminiJSON,
    GeminiOutputError,
    build_prompt,
    validate_model_data,
)
from category.bimonad import MetaMoPseudoBimonad
from core.config import NUM_MODULATORS
from core.state import MotivationalState
from dynamics.coherence import blend_states
from dynamics.stability import raise_boundary_caution
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from benchmarks.scorers.templates import FEELING_MODEL
from benchmarks.scorers.feelings import (
    SCORERS, READOUT_MODELS,
    score_feelings,
)
from benchmarks.scorers.quadratic import FORMULA_MODEL, formula_coefficients
from benchmarks.scorers.selection import EMOTION_SELECTORS, TIE_BREAKERS, select_answer


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
DEFAULT_REFERENCE = BENCHMARKS_DIR / "qpt6z-osfstorage-archive/experiment_1_STEU.csv"
DEFAULT_ITEMS = BENCHMARKS_DIR / "data/steu_items.jsonl"
PROTOCOL = "metamo-steu-module-ablation-v10"


EMOTION_ITEMS_ONLY = True
ITEM_LIMIT = None
MODEL = None
TEMPERATURE = 0.0
SEED = 42
REQUEST_INTERVAL = 0.0
RESULTS_DIRECTORY = BENCHMARKS_DIR / "results"
RESUME_DIRECTORY = None


class ConditionSession:
    """Keep one condition's interaction history and numerical state."""

    def __init__(self, condition: str, feeling_weight: float = FEELING_WEIGHT, selector: str = SELECTOR,
                 state_policy: str | None = None, feeling_scorer: str | None = None,
                 history_policy: str | None = None, tie_breaker: str | None = None, tie_tolerance: float | None = None):
        """Initialize only the modules enabled for this condition."""
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        if selector not in SELECTORS:
            raise ValueError(f"Unknown selector: {selector}")
        if not math.isfinite(feeling_weight) or feeling_weight < 0:
            raise ValueError("feeling-weight must be finite and nonnegative")
        self.state_policy = benchmark_config.STATE_POLICY if state_policy is None else state_policy
        self.feeling_scorer = benchmark_config.FEELING_SCORER if feeling_scorer is None else feeling_scorer
        if self.state_policy not in {"independent", "persistent"}:
            raise ValueError("Unknown state policy")
        self.history_policy = benchmark_config.HISTORY_POLICY if history_policy is None else history_policy
        self.tie_breaker = "letter" if tie_breaker is None else tie_breaker
        self.tie_tolerance = 1e-9 if tie_tolerance is None else tie_tolerance
        if self.history_policy not in benchmark_config.HISTORY_POLICIES:
            raise ValueError("Unknown history policy")
        if self.tie_breaker not in TIE_BREAKERS:
            raise ValueError("Unknown tie breaker")
        bounded(self.tie_tolerance, 0, float("inf"))
        if self.feeling_scorer not in SCORERS:
            raise ValueError("Unknown feeling scorer")
        if condition == "full_metamo" and self.feeling_scorer == "quadratic":
            formula_coefficients()
        self.selector = selector if condition == "full_metamo" else "gemini"

        self.feeling_weight = {"magus_only": 0.0, "emotion_only": 1.0}.get(selector, feeling_weight)
        self.condition = condition
        self.history = []
        self.state = initial_state() if condition == "full_metamo" else None
        self.decision = MagusDecision() if condition == "full_metamo" else None
        self.appraisal = OpenPsiAppraisal() if condition == "full_metamo" else None
        self.bimonad = MetaMoPseudoBimonad(self.appraisal, self.decision) if self.appraisal else None

    def _prepare_item(self):
        """Reset numerical state when requested, retaining interaction history."""
        if self.condition == "full_metamo" and self.state_policy == "independent":
            self.state = initial_state()

    def _prompt_history(self):
        """Return the history projection supplied to the next prompt."""
        if self.condition != "full_metamo" or self.history_policy == "full":
            return self.history
        if self.history_policy == "none":
            return []
        return [{"item": row["item"], "answer": row["answer"]} for row in self.history]

    def restore(self, item: Item, record: dict) -> None:
        """Restore a completed interaction without another API call or state update."""
        self._prepare_item()
        if record.get("condition") != self.condition or record.get("item_id") != item.id:
            raise ValueError("Saved response order does not match the requested run")
        if record.get("selector") != self.selector:
            raise ValueError("Saved response has a different selector")
        if record.get("prompt") != build_prompt(item, self.condition, self._prompt_history(), self.state):
            raise ValueError(f"Saved prompt or state differs at {self.condition} item {item.id}")
        if record.get("answer") not in list(LETTERS):
            raise ValueError("Saved response has an invalid answer letter")
        for field, accepted in (("source_correct", item.source_answers),
                                ("reference_compatible_correct", item.reference_answers)):
            if type(record.get(field)) is not int or record[field] != int(record["answer"] in accepted):
                raise ValueError(f"Saved response has inconsistent {field} at item {item.id}")
        if record.get("state_before") != snapshot(self.state) or record.get("history_items_before") != len(self._prompt_history()):
            raise ValueError("Saved response has inconsistent state or history")
        if self.condition in {"no_memory", "memory"}:
            if record.get("state_after") is not None:
                raise ValueError("Direct-answer response unexpectedly contains a retained state")
            if self.condition == "memory":
                self.history.append({"item": item.public(), "answer": record["answer"]})
            return
        saved = record["state_after"]
        modulators = saved["M"]
        if len(modulators) != NUM_MODULATORS:
            raise ValueError("Saved state has the wrong number of modulators")
        self.state = MotivationalState(
            G=vector(saved["G"], 0, 1),
            M=np.array([bounded(value, 0, 1) for value in modulators]),
        )
        self.history.append({
            "item": item.public(), "answer": record["answer"], "state": snapshot(self.state),
            "appraised_state": record["decision_state"] if self.appraisal else None,
            "stimulus": record["stimulus"] if self.appraisal else None,
        })
    def step(self, item: Item, query) -> dict:
        """Process one item and return its answer, state, and scores."""
        if self.selector in EMOTION_SELECTORS and not supports_emotion_selection(item):
            raise ValueError(f"Emotion-only cannot score item {item.id}: all choices must be emotion names. "
                             "Use the shared emotion-choice question set.")
        self._prepare_item()
        prompt = build_prompt(item, self.condition, self._prompt_history(), self.state)
        before = snapshot(self.state)
        history_length = len(self._prompt_history())
        started = time.monotonic()
        data, response_metadata = query(prompt)
        validate_model_data(data, self.condition)
        stimulus = None
        decision_state = None
        scores = None
        feeling_choice = None
        if self.condition in {"no_memory", "memory"}:
            answer = data["answer"]
            if not isinstance(answer, str) or answer not in list(LETTERS):
                raise ValueError("Expected a single answer letter A–E")
            if self.condition == "memory":
                self.history.append({"item": item.public(), "answer": answer})
        else:
            candidates = parse_plan(data)
            plausibility = parse_plausibility(data)
            stimulus = parse_stimulus(data["stimulus"])
            appraised = self.appraisal.appraise(self.state, stimulus)
            decision_state = raise_boundary_caution(appraised)
            scores = {a.id: self.decision.score_candidate(decision_state, a) for a in candidates}
            magus_pick = max(candidates, key=lambda a: scores[a.id]).id
            felt = score_feelings(decision_state, self.feeling_scorer)
            emotion_scores = {letter: felt.get(option_feeling(text)) for letter, text in item.options.items()}
            emotion_pick = max(LETTERS, key=lambda letter: emotion_scores[letter]) if all(
                value is not None for value in emotion_scores.values()) else None
            bonus = feeling_bonus(item, felt)
            combined = {a.id: scores[a.id] + self.feeling_weight * bonus[a.id] for a in candidates}
            picked, final, tie = select_answer(self.selector, scores, emotion_scores, plausibility, combined,
                                               tie_breaker=self.tie_breaker, tolerance=self.tie_tolerance)
            chosen = next(a for a in candidates if a.id == picked)
            feeling_choice = {"model": READOUT_MODELS.get(self.feeling_scorer, FEELING_MODEL),
                              "selector": self.selector,
                              "feeling_weight": self.feeling_weight, "magus_pick": magus_pick,
                              "scorer": self.feeling_scorer, "tie": tie, "plausibility_scores": plausibility,
                              "plausibility_pick": max(LETTERS, key=lambda k: plausibility[k]),
                              "emotion_pick": emotion_pick, "emotion_scores": emotion_scores,
                              "feelings": felt, "bonus": bonus, "combined_scores": combined, "final_scores": final}
            if self.selector in EMOTION_SELECTORS | {"plausibility"}:
                action, target = self.bimonad.step(self.state, stimulus, [chosen])
            else:
                action, target = self.bimonad.step(self.state, stimulus, candidates)
                if chosen.id != action.id:
                    action, target = self.bimonad.step(self.state, stimulus, [chosen])
            answer = action.id
            self.state = blend_states(self.state, target)
            self.history.append({
                "item": item.public(), "answer": answer, "state": snapshot(self.state),
                "appraised_state": snapshot(decision_state) if stimulus else None,
                "stimulus": asdict(stimulus) if stimulus else None,
            })
        return {
            "condition": self.condition, "selector": self.selector, "item_id": item.id, "answer": answer,
            "source_correct": int(answer in item.source_answers),
            "reference_compatible_correct": int(answer in item.reference_answers),
            "history_items_before": history_length, "state_before": before,
            "stimulus": asdict(stimulus) if stimulus else None,
            "decision_state": snapshot(decision_state), "candidate_scores": scores, "feeling_choice": feeling_choice,
            "state_after": snapshot(self.state), "prompt": prompt, "model_data": data,
            "response": response_metadata, "elapsed_seconds": time.monotonic() - started,
        }

def run_experiment(items: list[Item], query, output: Path, metadata: dict, *, resume: bool = False,
                   request_interval: float = 0.0, conditions: list[str] | None = None,
                   feeling_weight: float = FEELING_WEIGHT, selector: str = SELECTOR,
                   state_policy: str | None = None, feeling_scorer: str | None = None,
                   history_policy: str | None = None, tie_breaker: str | None = None,
                   tie_tolerance: float | None = None, case_id: str | None = None) -> dict:
    """Run or resume conditions and save responses and accuracy summaries."""
    manifest_path = output / "manifest.json"
    names = list(conditions or CONDITIONS)
    if not items or not names or len(names) != len(set(names)):
        raise ValueError("Expected items and distinct conditions")
    sessions = {condition: ConditionSession(condition, feeling_weight, selector, state_policy, feeling_scorer,
                                            history_policy, tie_breaker, tie_tolerance) for condition in names}
    history_policy = sessions[names[0]].history_policy
    tie_breaker = sessions[names[0]].tie_breaker
    tie_tolerance = sessions[names[0]].tie_tolerance
    state_policy = sessions[names[0]].state_policy
    feeling_scorer = sessions[names[0]].feeling_scorer
    if feeling_scorer == "quadratic":
        metadata = {**metadata, "emotion_formula_coefficients_sha256":
                    benchmark_code_hashes()["emotion_formula_coefficients_sha256"]}
    feeling_weight = sessions[names[0]].feeling_weight
    if "full_metamo" in names and selector in EMOTION_SELECTORS:
        unsupported = [i.id for i in items if not supports_emotion_selection(i)]
        if unsupported:
            raise ValueError(f"Emotion-only cannot score items {unsupported}; use the shared emotion-choice question set. "
                             "No responses have been requested.")
    expected = [(condition, item.id) for condition in names for item in items]
    results = []
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            raise ValueError("This run is already complete")
        for key in ["model", "temperature", "seed", "items_sha256", "max_output_tokens"]:
            if key not in manifest or manifest[key] != metadata.get(key):
                raise ValueError(f"Cannot resume with different {key}")
        if manifest.get("feeling_model") != FEELING_MODEL:
            raise ValueError("Cannot resume a different feeling model")
        if feeling_scorer == "quadratic":
            if manifest.get("readout_model") != FORMULA_MODEL:
                raise ValueError("Cannot resume a quadratic run from a different formula version")
            if manifest.get("emotion_formula_coefficients_sha256") != metadata["emotion_formula_coefficients_sha256"]:
                raise ValueError("Cannot resume after quadratic formula parameters change")
        if manifest.get("protocol") != PROTOCOL or manifest.get("item_order") != [i.id for i in items]:
            raise ValueError("Cannot resume a different protocol or item order")
        if manifest.get("conditions", list(CONDITIONS)) != names:
            raise ValueError("Cannot resume with different conditions")
        for key, value in (("state_policy", state_policy), ("feeling_scorer", feeling_scorer),
                           ("history_policy", history_policy), ("tie_breaker", tie_breaker),
                           ("tie_tolerance", tie_tolerance), ("case_id", case_id)):
            if manifest.get(key) != value:
                raise ValueError(f"Cannot resume with a different {key}")
        if manifest.get("selector") != selector:
            raise ValueError("Cannot resume with a different selector")
        if manifest.get("emotion_items_only", False) != metadata.get("emotion_items_only", False):
            raise ValueError("Cannot resume with a different item filter")
        if manifest.get("feeling_weight", FEELING_WEIGHT) != feeling_weight:
            raise ValueError("Cannot resume with a different feeling weight")
        results = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines() if line.strip()]
        if len(results) > len(expected):
            raise ValueError("Saved responses exceed the expected number of items")
        item_map = {item.id: item for item in items}
        for index, record in enumerate(results):
            if (record.get("condition"), record.get("item_id")) != expected[index]:
                raise ValueError("Saved responses are not a complete ordered prefix of this run")
            sessions[record["condition"]].restore(item_map[record["item_id"]], record)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        manifest.setdefault("resume_history", []).append({
            "resumed_utc": datetime.now(timezone.utc).isoformat(), "completed_before_resume": len(results),
            "previous_status": manifest.get("status"), "previous_error_type": manifest.get("error_type"),
            "benchmark_code_sha256": metadata.get("benchmark_code_sha256"),
            "benchmark_helpers_sha256": metadata.get("benchmark_helpers_sha256"),
            "request_interval_seconds": request_interval,
            "request_timeout_ms": REQUEST_TIMEOUT_MS,
            "config": config_snapshot(),
        })
        failure_path = output / "failure.json"
        if failure_path.exists():
            failure_path.rename(output / f"failure.before-resume-{stamp}.json")
        print(f"Resuming {output}: {len(results)}/{len(expected)} responses retained", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {**metadata, "protocol": PROTOCOL, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config_snapshot(),
                    "item_order": [i.id for i in items], "conditions": names, "selector": selector,
                    "state_policy": state_policy, "feeling_scorer": feeling_scorer,
                    "history_policy": history_policy, "tie_breaker": tie_breaker,
                    "tie_tolerance": tie_tolerance, "case_id": case_id,
                    "feeling_weight": feeling_weight, "feeling_model": FEELING_MODEL,
                    "readout_model": READOUT_MODELS.get(feeling_scorer, FEELING_MODEL),
                    "calls_expected": len(expected)}
    manifest.pop("error_type", None)
    manifest.update(status="running", completed_items=len(results))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed = len(results)
    last_request = None
    try:
        with (output / "responses.jsonl").open("a" if resume else "w") as handle:
            for condition in names:
                session = sessions[condition]
                for item in items:
                    if completed:
                        completed -= 1
                        continue
                    if last_request is not None:
                        remaining = request_interval - (time.monotonic() - last_request)
                        if remaining > 0:
                            time.sleep(remaining)
                    last_request = time.monotonic()
                    record = session.step(item, query)
                    results.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"{condition}: item {item.id}, answer {record['answer']}", flush=True)
    except Exception as error:
        failure = {"condition": condition, "item_id": item.id,
                   "error_type": type(error).__name__, "error": str(error)}
        if isinstance(error, GeminiOutputError):
            failure.update(prompt=build_prompt(item, condition, session._prompt_history(), session.state),
                           attempts=error.attempts)
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        manifest.update(status="failed", completed_items=len(results), error_type=type(error).__name__)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Failed run saved: {output}", file=sys.stderr)
        raise
    summary = {"kind": "fresh_metamo_module_ablation", "scoring_provisional": True,
               "selector": selector, "feeling_weight": feeling_weight,
               "scope": manifest.get("scope", "42-item benchmark" if len(items) == 42 else "partial benchmark"),
               "item_order": [i.id for i in items], "excluded_item_ids": manifest.get("excluded_item_ids", []),
               "conditions": {}}
    for condition in names:
        records = [r for r in results if r["condition"] == condition]
        summary["conditions"][condition] = {
            "selector": sessions[condition].selector,
            "total": len(records), "source_correct": sum(r["source_correct"] for r in records),
            "source_accuracy": sum(r["source_correct"] for r in records) / len(records),
            "reference_compatible_correct": sum(r["reference_compatible_correct"] for r in records),
            "reference_compatible_accuracy": sum(r["reference_compatible_correct"] for r in records) / len(records),
        }
        if condition == "full_metamo":
            changed = sum(r["answer"] != r["feeling_choice"]["magus_pick"] for r in records if r.get("feeling_choice"))
            summary["conditions"][condition]["changed_from_magus"] = changed
            if selector in {"emotion_only", "magus_with_emotion"}:
                summary["conditions"][condition]["changed_by_feelings"] = changed
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest.update(status="complete", completed_items=len(results))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def _configured_path(value) -> Path:
    """Resolve configured relative paths from the repository root."""
    return (ROOT / Path(value)).resolve()


def run_configured_experiment() -> Path:
    """Load saved or configured settings, run cases, and generate comparison plots."""
    from dotenv import load_dotenv
    from benchmarks.generate_plots import analyze_runs

    load_dotenv(ROOT / ".env")
    resuming = RESUME_DIRECTORY is not None
    if resuming:
        output = _configured_path(RESUME_DIRECTORY)
        experiment = json.loads((output / "experiment.json").read_text())
        if experiment.get("protocol") != PROTOCOL or experiment.get("feeling_model") != FEELING_MODEL:
            raise ValueError("Cannot resume a different protocol or feeling model")
        if experiment["config"]["core.config"] != config_snapshot()["core.config"]:
            raise ValueError("Restore the original core.config values before resuming")
        settings, metadata = experiment["settings"], experiment["metadata"]
        if (metadata["max_output_tokens"] != MAX_OUTPUT_TOKENS
                or metadata["max_response_attempts"] != MAX_RESPONSE_ATTEMPTS):
            raise ValueError("Restore the original output-token and retry settings before resuming")
        settings.pop("items_file", None)
        items_path = _configured_path(DEFAULT_ITEMS)
        if sha256(items_path) != metadata["items_sha256"]:
            raise ValueError("The question file has changed since this experiment started")
        by_id = {item.id: item for item in load_items(items_path)}
        items = [by_id[item_id] for item_id in experiment["item_order"]]
    else:
        names = list(benchmark_config.CONDITIONS)
        if (not names or len(set(names)) != len(names) or any(name not in CONDITIONS for name in names)):
            raise ValueError("CONDITIONS must contain distinct supported condition names")
        if benchmark_config.SELECTOR not in SELECTORS:
            raise ValueError("SELECTOR must be one of SELECTORS")
        if type(benchmark_config.COMPARE_SELECTORS) is not bool or type(EMOTION_ITEMS_ONLY) is not bool:
            raise ValueError("COMPARE_SELECTORS and EMOTION_ITEMS_ONLY must be True or False")
        limit = ITEM_LIMIT
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 42):
            raise ValueError("ITEM_LIMIT must be None or an integer from 1 to 42")
        temperature = bounded(TEMPERATURE, 0, 2)
        interval = bounded(REQUEST_INTERVAL, 0, float("inf"))
        weight = bounded(benchmark_config.FEELING_WEIGHT, 0, float("inf"))
        if type(SEED) is not int:
            raise ValueError("SEED must be an integer")
        for name, value in (("MAX_OUTPUT_TOKENS", MAX_OUTPUT_TOKENS), ("MAX_RESPONSE_ATTEMPTS", MAX_RESPONSE_ATTEMPTS)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if MODEL is not None and not isinstance(MODEL, str):
            raise ValueError("MODEL must be a model name or None to read .env")
        model = (MODEL or os.getenv("GEMINI_MODEL", "")).strip() or "gemini-3.1-flash-lite"
        items_path = _configured_path(DEFAULT_ITEMS)
        if not items_path.exists():
            prepare(BENCHMARKS_DIR / "data/steu_author_source.txt", DEFAULT_REFERENCE, items_path)
        requested = load_items(items_path)
        items = [item for item in requested if supports_emotion_selection(item)] if EMOTION_ITEMS_ONLY else requested
        excluded = [item.id for item in requested if item not in items]
        items = items[:limit] if limit is not None else items
        if not items:
            raise ValueError("No questions remain after filtering")
        selectors = [benchmark_config.SELECTOR]
        if benchmark_config.COMPARE_SELECTORS and "full_metamo" in names:
            selectors += [name for name in SELECTORS if name not in selectors]
        plan = [{"directory": "primary", "conditions": names, "selector": selectors[0]}]
        plan += [{"directory": name, "conditions": ["full_metamo"], "selector": name}
                 for name in selectors[1:]]
        if ("full_metamo" in names and any(case["selector"] in EMOTION_SELECTORS for case in plan)
                and any(not supports_emotion_selection(item) for item in items)):
            raise ValueError("Emotion selectors require EMOTION_ITEMS_ONLY=True; no API calls were made")
        ConditionSession("full_metamo")
        settings = {"model": model, "temperature": temperature,
                    "seed": SEED, "feeling_weight": weight, "request_interval": interval,
                    "state_policy": benchmark_config.STATE_POLICY, "feeling_scorer": benchmark_config.FEELING_SCORER,
                    "history_policy": benchmark_config.HISTORY_POLICY}
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            revision = None
        provenance = items_path.with_suffix(".provenance.json")
        metadata = {
            "model": model, "temperature": temperature, "seed": SEED,
            "limit": limit, "emotion_items_only": EMOTION_ITEMS_ONLY,
            "excluded_item_ids": excluded, "request_interval_seconds": interval,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "max_response_attempts": MAX_RESPONSE_ATTEMPTS,
            "request_timeout_ms": REQUEST_TIMEOUT_MS,
            "response_format": "condition-specific JSON Schema with local validation",
            "items_sha256": sha256(items_path), "git_revision": revision,
            **benchmark_code_hashes(),
            "dataset_provenance": json.loads(provenance.read_text()) if provenance.exists() else None,
            "scope": "emotion-label subset, not a full STEU score" if EMOTION_ITEMS_ONLY else (
                "42-item benchmark" if len(items) == 42 else "partial pilot, not a full STEU score"),
        }
        output = _configured_path(RESULTS_DIRECTORY) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        experiment = {"kind": "steu_comparison", "protocol": PROTOCOL, "feeling_model": FEELING_MODEL,
                      "config": config_snapshot(), "settings": settings, "metadata": metadata,
                      "item_order": [item.id for item in items], "runs": plan,
                      "started_utc": datetime.now(timezone.utc).isoformat()}
        output.mkdir(parents=True, exist_ok=False)

    metadata = {**metadata, **benchmark_code_hashes(),
                "request_timeout_ms": REQUEST_TIMEOUT_MS}
    experiment_path = output / "experiment.json"
    experiment.update(status="running", stage="responses")
    experiment.pop("error", None)
    experiment_path.write_text(json.dumps(experiment, indent=2) + "\n")
    print(f"Experiment folder: {output}", flush=True)
    print(f"Shared questions: {len(items)}; scheduled runs: {len(experiment['runs'])}", flush=True)
    planned_calls = sum(len(c["conditions"]) for c in experiment["runs"]) * len(items)
    print(f"Planned model responses before retries: {planned_calls}", flush=True)
    query = None
    last_request = None

    def paced_query(prompt):
        """Keep request spacing across the boundaries between selector runs."""
        nonlocal last_request
        if last_request is not None:
            remaining = settings["request_interval"] - (time.monotonic() - last_request)
            if remaining > 0:
                time.sleep(remaining)
        last_request = time.monotonic()
        return query(prompt)

    try:
        runs = []
        for entry in experiment["runs"]:
            run = output / entry["directory"]
            runs.append(run)
            manifest_path = run / "manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete":
                print(f"Retaining completed run: {run.name}", flush=True)
                continue
            if query is None:
                query = GeminiJSON(settings["model"], settings["temperature"], settings["seed"])
            run_experiment(items, paced_query, run, metadata, resume=manifest_path.exists(),
                           request_interval=settings["request_interval"], conditions=entry["conditions"],
                           selector=entry["selector"], feeling_weight=settings["feeling_weight"],
                           state_policy=settings["state_policy"], feeling_scorer=settings["feeling_scorer"],
                           history_policy=settings["history_policy"])
        experiment.update(stage="plots")
        experiment_path.write_text(json.dumps(experiment, indent=2) + "\n")
        plots = output / "plots"
        if plots.exists():
            plots = output / ("plots_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        analysis = analyze_runs(runs, items_path, plots)
        experiment.update(status="complete", stage="complete", plots_directory=plots.name,
                          graphs=analysis["graphs"], available_cases=analysis["available_cases"],
                          completed_utc=datetime.now(timezone.utc).isoformat())
        experiment_path.write_text(json.dumps(experiment, indent=2) + "\n")
        print(f"Saved {len(analysis['graphs'])} PNG graphs, analysis.json, and question_results.csv: {plots}")
        if analysis["missing_cases"]:
            print(f"Partial comparison; cases not configured: {', '.join(analysis['missing_cases'])}")
    except Exception as error:
        experiment.update(status="failed", error=str(error))
        experiment_path.write_text(json.dumps(experiment, indent=2) + "\n")
        print(f"To continue, set RESUME_DIRECTORY in benchmarks/steu_experiment.py to {str(output)!r} and run again.", file=sys.stderr)
        raise
    return output


def main(argv=None) -> int:
    """Use benchmarks/config.py; no subcommands or flags are required."""
    try:
        if (sys.argv[1:] if argv is None else argv):
            raise ValueError("Edit benchmarks/config.py, then run python -m benchmarks.steu_experiment without arguments")
        run_configured_experiment()
    except (ValueError, KeyError, TypeError, RuntimeError, OSError, ImportError) as error:
        print(f"STEU experiment stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
