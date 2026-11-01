"""Offline protocol checks."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from benchmarks.phase1 import ConditionSession, GeminiJSON, GeminiOutputError, build_prompt, main, parse_plan, parse_stimulus, run_experiment
from benchmarks.steu import Item, audit, load_items, prepare, sha256, write_audit


def synthetic_item(item_id=1):
    """Create an invented question with fixed scoring keys."""
    return Item(item_id, "An invented person receives welcome news. Which emotion is likely?",
                {letter: f"Invented option {letter}" for letter in "ABCDE"}, ["C"], ["C"])


def synthetic_query(prompt):
    """Return deterministic answers, candidates, and appraisal features."""
    if "previous_interactions" not in json.loads(prompt.split("\n\nDATA:\n", 1)[1]):
        return {"answer": "C"}, {"backend": "synthetic_test_fixture"}
    candidates = [{
        "id": letter, "risk_estimate": 0.1,
        "goal_correlations": [0, 0, 0.8 if letter == "C" else 0.1, 0, 0, 0, 0.8 if letter == "C" else 0.1, 0],
        "delta_g": [0, 0, 0.01, 0, 0, 0, 0, 0],
    } for letter in "ABCDE"]
    return {"candidates": candidates, "stimulus": {
        "novelty": 0.3, "conduciveness": -0.5, "risk": 0.8, "effort": 0.3,
    }}, {"backend": "synthetic_test_fixture"}


def synthetic_response(text, finish_reason="STOP"):
    """Mimic a Gemini reply with a chosen finish reason."""
    return SimpleNamespace(text=text, usage_metadata=None, model_version="synthetic-test-model",
                           candidates=[SimpleNamespace(finish_reason=finish_reason)])


def mocked_gemini(responses):
    """Configure the real wrapper with a mocked Gemini client."""
    with patch("google.genai.Client"), patch("dotenv.load_dotenv"), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-test-key"}):
        query = GeminiJSON("synthetic-test-model", 0, 42)
    query.client.models.generate_content.side_effect = responses
    return query


class ProtocolTests(unittest.TestCase):
    """Check condition isolation, data handling, and response failures offline."""

    def test_no_memory_never_constructs_or_updates_motivational_components(self):
        """Verify the baseline bypasses state, history, and motivational modules."""
        with patch("benchmarks.phase1.initial_state", side_effect=AssertionError("state accessed")), \
             patch("benchmarks.phase1.MagusDecision", side_effect=AssertionError("Magus accessed")), \
             patch("benchmarks.phase1.OpenPsiAppraisal", side_effect=AssertionError("appraisal accessed")):
            session = ConditionSession("no_memory")
            for item_id in range(1, 4):
                record = session.step(synthetic_item(item_id), synthetic_query)
                self.assertIsNone(record["state_before"])
                self.assertIsNone(record["state_after"])
                self.assertEqual(record["history_items_before"], 0)
                self.assertNotIn("previous_interactions", record["prompt"])
                self.assertNotIn("motivational_state", record["prompt"])
            self.assertEqual(session.history, [])

    def test_memory_bypasses_appraisal_and_keeps_affect_neutral(self):
        """Verify memory updates goals without enabling appraisal or affect changes."""
        with patch("benchmarks.phase1.OpenPsiAppraisal", side_effect=AssertionError("appraisal accessed")):
            session = ConditionSession("memory")
            first = session.step(synthetic_item(), synthetic_query)
            second = session.step(synthetic_item(2), synthetic_query)
        self.assertEqual(second["history_items_before"], 1)
        self.assertIsNone(second["stimulus"])
        np.testing.assert_array_equal(second["state_after"]["M"], np.full(6, 0.5))
        self.assertGreater(second["state_after"]["G"][2], first["state_before"]["G"][2])
        self.assertEqual(second["answer"], "C")

    def test_full_metamo_appraises_uses_magus_and_stores_shifted_state(self):
        """Verify appraisal affects Magus and persists the resulting state."""
        session = ConditionSession("full_metamo")
        first = session.step(synthetic_item(), synthetic_query)
        second = session.step(synthetic_item(2), synthetic_query)
        self.assertNotEqual(first["state_before"]["M"], first["decision_state"]["M"])
        self.assertNotEqual(first["state_before"]["M"], first["state_after"]["M"])
        self.assertEqual(first["answer"], max(first["candidate_scores"], key=first["candidate_scores"].get))
        self.assertEqual(second["state_before"], first["state_after"])
        self.assertEqual(second["history_items_before"], 1)
        self.assertEqual(session.history[0]["state"], first["state_after"])

    def test_answers_and_published_outputs_never_enter_prompts_or_memory(self):
        """Keep scoring keys out of prompts and interaction history."""
        item = synthetic_item()
        item = Item(item.id, item.question, item.options, ["SOURCE_KEY_SENTINEL"], ["REFERENCE_KEY_SENTINEL"])
        for condition in ["no_memory", "memory", "full_metamo"]:
            session = ConditionSession(condition)
            first = session.step(item, synthetic_query)
            followup = build_prompt(synthetic_item(2), condition, session.history, session.state)
            for value in ["SOURCE_KEY_SENTINEL", "REFERENCE_KEY_SENTINEL", "source_answers", "reference_answers"]:
                self.assertNotIn(value, first["prompt"] + followup)

    def test_invalid_outputs_fail_without_neutral_or_heuristic_fallback(self):
        """Reject invalid values and plans without changing retained state."""
        valid, _ = synthetic_query(build_prompt(synthetic_item(), "memory", [], None))
        valid["candidates"][0]["goal_correlations"][3] = float("nan")
        with self.assertRaises(ValueError):
            parse_plan(valid)
        with self.assertRaises(ValueError):
            parse_stimulus({"novelty": 0, "conduciveness": -1.2, "risk": 0, "effort": 0})
        session = ConditionSession("full_metamo")
        before = session.state.copy()
        with self.assertRaises(ValueError):
            session.step(synthetic_item(), lambda _: ({"candidates": []}, {}))
        self.assertEqual(session.history, [])
        np.testing.assert_array_equal(session.state.M, before.M)

    def test_failed_service_retains_partial_logs_and_marks_run_failed(self):
        """Preserve completed responses when a later request fails."""
        calls = 0

        def interrupted_query(prompt):
            """Simulate a service failure on the second request."""
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic service failure")
            return synthetic_query(prompt)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "run"
            with self.assertRaises(RuntimeError):
                run_experiment([synthetic_item(), synthetic_item(2)], interrupted_query, output, {"backend": "test"})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_items"], 1)
            self.assertEqual(len((output / "responses.jsonl").read_text().splitlines()), 1)
            self.assertFalse((output / "summary.json").exists())

    def test_complete_run_keeps_condition_histories_separate(self):
        """Verify each condition starts with an independent history."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "run"
            summary = run_experiment([synthetic_item(), synthetic_item(2)], synthetic_query, output, {"backend": "test"})
            records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 6)
            for condition in ["no_memory", "memory", "full_metamo"]:
                histories = [r["history_items_before"] for r in records if r["condition"] == condition]
                self.assertEqual(histories, [0, 0] if condition == "no_memory" else [0, 1])
                self.assertEqual(summary["conditions"][condition]["total"], 2)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")

    def test_source_parser_preserves_multiple_accepted_answers(self):
        """Preserve multiple source answers without copying published explanations."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("Test items\n" + "\n".join(
                f"{i}. Invented question {i}?\n(a) One\n(b) Two\n(c) Three\n(d) Four\n(e) Five"
                for i in range(1, 43)
            ) + "\nScoring the STEU in SPSS\n" + "\n".join(
                f"RECODE STEU{i:02d} (1=0) (2=1) (3=0) (4=0) (5={int(i == 25)}) INTO STEU_R{i:02d}."
                for i in range(1, 43)
            ))
            reference = root / "reference.csv"
            reference.write_text("STEU,No Memory,No Memory Score,Memory,Memory Score,Appraisal,Appraisal Score,Output\n" + "".join(
                f"{i},B,1,B,1,B,1,DO_NOT_USE_AS_QUESTION\n" for i in range(1, 43)
            ))
            destination = root / "items.jsonl"
            items = prepare(source, reference, destination)
            self.assertEqual(items[24].source_answers, ["B", "E"])
            self.assertEqual(items[24].reference_answers, ["B"])
            self.assertEqual(len(load_items(destination)), 42)
            self.assertNotIn("DO_NOT_USE_AS_QUESTION", destination.read_text())

    def test_gemini_retries_truncated_and_malformed_json_with_same_prompt_and_schema(self):
        """Retry invalid JSON without changing the prompt, seed, or schema."""
        query = mocked_gemini([
            synthetic_response('{"answer":', "MAX_TOKENS"),
            synthetic_response('{"answer": "C"'),
            synthetic_response('{"answer": "C"}'),
        ])
        prompt = build_prompt(synthetic_item(), "no_memory", [], None)
        with patch("benchmarks.phase1.time.sleep"), redirect_stdout(StringIO()):
            data, metadata = query(prompt)
        self.assertEqual(data, {"answer": "C"})
        self.assertEqual(metadata["attempts"], 3)
        self.assertEqual(len(metadata["failed_attempts"]), 2)
        self.assertEqual(metadata["failed_attempts"][0]["finish_reasons"], ["MAX_TOKENS"])
        self.assertEqual(metadata["failed_attempts"][1]["raw_text"], '{"answer": "C"')
        for call in query.client.models.generate_content.call_args_list:
            self.assertEqual(call.kwargs["contents"], prompt)
            config = call.kwargs["config"]
            self.assertEqual(config.seed, 42)
            self.assertEqual(config.response_json_schema["required"], ["answer"])
            self.assertEqual(config.response_json_schema["properties"]["answer"]["enum"], list("ABCDE"))

    def test_gemini_validates_candidates_before_committing_state_or_memory(self):
        """Reject duplicate candidates before committing one successful interaction."""
        session = ConditionSession("full_metamo")
        prompt = build_prompt(synthetic_item(), session.condition, [], session.state)
        valid, _ = synthetic_query(prompt)
        invalid = json.loads(json.dumps(valid))
        invalid["candidates"][0]["id"] = "B"  # Duplicate B and missing A, despite valid JSON.
        query = mocked_gemini([synthetic_response(json.dumps(invalid)), synthetic_response(json.dumps(valid))])
        with patch("benchmarks.phase1.time.sleep"), redirect_stdout(StringIO()):
            record = session.step(synthetic_item(), query)
        self.assertEqual(record["answer"], "C")
        self.assertEqual(record["response"]["attempts"], 2)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(record["history_items_before"], 0)
        self.assertEqual([c.kwargs["contents"] for c in query.client.models.generate_content.call_args_list], [prompt, prompt])
        schema = query.client.models.generate_content.call_args.kwargs["config"].response_json_schema
        self.assertIn("stimulus", schema["required"])

    def test_gemini_exhaustion_preserves_raw_replies_in_failure_log(self):
        """Save rejected replies and mark exhausted runs as failed."""
        query = mocked_gemini([synthetic_response('{"answer":') for _ in range(3)])
        with tempfile.TemporaryDirectory() as directory, patch("benchmarks.phase1.time.sleep"), redirect_stdout(StringIO()):
            output = Path(directory) / "run"
            with self.assertRaises(GeminiOutputError):
                run_experiment([synthetic_item()], query, output, {"backend": "synthetic_test_fixture"})
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["condition"], "no_memory")
            self.assertEqual(failure["item_id"], 1)
            self.assertEqual(len(failure["attempts"]), 3)
            self.assertTrue(all(a["raw_text"] == '{"answer":' for a in failure["attempts"]))
            self.assertEqual((output / "responses.jsonl").read_text(), "")
            self.assertFalse((output / "summary.json").exists())
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")

    def test_gemini_retries_service_errors_but_stops_on_invalid_credentials(self):
        """Retry transient errors and stop immediately on invalid credentials."""
        class FakeAPIError(Exception):
            """Mimic an API exception carrying an HTTP status code."""

            def __init__(self, code):
                """Set the simulated HTTP status code."""
                self.code = code

        prompt = build_prompt(synthetic_item(), "no_memory", [], None)
        retrying = mocked_gemini([FakeAPIError(503), synthetic_response('{"answer":"C"}')])
        with patch("benchmarks.phase1.time.sleep"), redirect_stdout(StringIO()):
            _, metadata = retrying(prompt)
        self.assertEqual(metadata["attempts"], 2)
        self.assertEqual(metadata["failed_attempts"][0]["status_code"], 503)
        unauthorized = mocked_gemini([FakeAPIError(401)])
        with self.assertRaises(FakeAPIError):
            unauthorized(prompt)
        self.assertEqual(unauthorized.client.models.generate_content.call_count, 1)

    def test_rate_limit_uses_retry_hint_and_keeps_api_diagnostics(self):
        """Honor Google's retry delay and retain quota details in response metadata."""
        from google.genai.errors import ClientError

        details = {"error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Per-minute token quota exceeded",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "45s"}],
        }}
        query = mocked_gemini([ClientError(429, details), synthetic_response('{"answer":"C"}')])
        with patch("benchmarks.phase1.time.sleep") as sleep, redirect_stdout(StringIO()):
            _, metadata = query(build_prompt(synthetic_item(), "no_memory", [], None))
        sleep.assert_called_once_with(46.0)
        self.assertEqual(metadata["failed_attempts"][0]["api_details"], details)
        self.assertIn("token quota", metadata["failed_attempts"][0]["api_message"])

    def test_rate_limit_backoff_and_exhaustion_name_http_429(self):
        """Wait through a minute-scale limit and report quota exhaustion accurately."""
        from google.genai.errors import ClientError

        query = mocked_gemini([ClientError(429, {"error": {"message": "Rate limited"}}) for _ in range(3)])
        with patch("benchmarks.phase1.time.sleep") as sleep, redirect_stdout(StringIO()):
            with self.assertRaisesRegex(GeminiOutputError, "HTTP 429") as caught:
                query(build_prompt(synthetic_item(), "no_memory", [], None))
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [30.0, 60.0])
        self.assertEqual(len(caught.exception.attempts), 3)

    def test_daily_quota_stops_without_repeated_requests(self):
        """Preserve daily-quota diagnostics without waiting for another short retry."""
        from google.genai.errors import ClientError

        details = {"error": {"message": "Daily quota exceeded", "details": [{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
        }]}}
        query = mocked_gemini([ClientError(429, details)])
        with patch("benchmarks.phase1.time.sleep") as sleep:
            with self.assertRaisesRegex(GeminiOutputError, "daily quota"):
                query(build_prompt(synthetic_item(), "no_memory", [], None))
        sleep.assert_not_called()
        self.assertEqual(query.client.models.generate_content.call_count, 1)

    def test_resume_preserves_answers_and_restores_exact_memory(self):
        """Continue only pending items with the same prompts and states as an uninterrupted run."""
        items = [synthetic_item(), synthetic_item(2)]
        metadata = {"model": "test", "temperature": 0.0, "seed": 42,
                    "items_sha256": "fixture", "max_output_tokens": 16384}
        calls = 0

        def interrupted(prompt):
            """Stop after five successful responses."""
            nonlocal calls
            calls += 1
            if calls == 6:
                raise RuntimeError("simulated quota failure")
            return synthetic_query(prompt)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            output = Path(directory) / "interrupted"
            baseline = Path(directory) / "uninterrupted"
            expected_summary = run_experiment(items, synthetic_query, baseline, metadata)
            with self.assertRaises(RuntimeError):
                run_experiment(items, interrupted, output, metadata)
            saved_bytes = (output / "responses.jsonl").read_bytes()
            query = Mock(side_effect=synthetic_query)
            summary = run_experiment(items, query, output, metadata, resume=True)
            query.assert_called_once()
            self.assertTrue((output / "responses.jsonl").read_bytes().startswith(saved_bytes))
            expected_records = [json.loads(line) for line in (baseline / "responses.jsonl").read_text().splitlines()]
            actual_records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            for expected, actual in zip(expected_records, actual_records):
                for key in ["condition", "item_id", "answer", "state_before", "state_after", "prompt", "history_items_before"]:
                    self.assertEqual(actual[key], expected[key])
            self.assertEqual(len(actual_records), 6)
            self.assertEqual(summary, expected_summary)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["resume_history"][0]["completed_before_resume"], 5)
            self.assertEqual(len(list(output.glob("failure.before-resume-*.json"))), 1)

    def test_resume_rejects_changed_inputs_and_unordered_logs_before_writing(self):
        """Reject incompatible checkpoints before changing files or querying Gemini."""
        metadata = {"model": "test", "temperature": 0.0, "seed": 42,
                    "items_sha256": "fixture", "max_output_tokens": 16384}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            output = Path(directory) / "run"
            with self.assertRaises(RuntimeError):
                run_experiment([synthetic_item()], Mock(side_effect=RuntimeError("stop")), output, metadata)
            original_manifest = (output / "manifest.json").read_bytes()
            query = Mock()
            for key, value in [("model", "different-model"), ("seed", 99), ("items_sha256", "changed")]:
                with self.assertRaisesRegex(ValueError, key):
                    run_experiment([synthetic_item()], query, output, {**metadata, key: value}, resume=True)
            invalid = {"condition": "memory", "item_id": 1}
            (output / "responses.jsonl").write_text(json.dumps(invalid) + "\n")
            with self.assertRaisesRegex(ValueError, "ordered prefix"):
                run_experiment([synthetic_item()], query, output, metadata, resume=True)
            self.assertEqual((output / "manifest.json").read_bytes(), original_manifest)
            query.assert_not_called()

    def test_resume_cli_uses_saved_settings_instead_of_environment_defaults(self):
        """Keep the original model, temperature, seed, and item count when resuming."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "manifest.json").write_text(json.dumps({
                "model": "saved-model", "temperature": 0.3, "seed": 7, "item_order": [1, 2],
            }))
            with patch.dict(os.environ, {"GEMINI_MODEL": "different-environment-model"}), \
                 patch("benchmarks.phase1.GeminiJSON") as gemini, \
                 patch("benchmarks.phase1.run_experiment", return_value={}) as run, redirect_stdout(StringIO()):
                self.assertEqual(main(["run", "--resume", str(output), "--request-interval", "6"]), 0)
            gemini.assert_called_once_with("saved-model", 0.3, 7)
            self.assertEqual(len(run.call_args.args[0]), 2)
            self.assertTrue(run.call_args.kwargs["resume"])
            self.assertEqual(run.call_args.kwargs["request_interval"], 6)

    def test_request_pacing_applies_across_condition_boundaries(self):
        """Enforce the requested start interval across all pending item calls."""
        clock = [0.0]
        starts = []

        def sleep(seconds):
            """Advance a synthetic clock without waiting."""
            clock[0] += seconds

        def query(prompt):
            """Simulate a request that takes one second."""
            starts.append(clock[0])
            clock[0] += 1
            return synthetic_query(prompt)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
             patch("benchmarks.phase1.time.monotonic", side_effect=lambda: clock[0]), \
             patch("benchmarks.phase1.time.sleep", side_effect=sleep):
            run_experiment([synthetic_item(), synthetic_item(2)], query, Path(directory) / "run", {}, request_interval=6)
        self.assertEqual(starts, [0, 6, 12, 18, 24, 30])


class ExperimentalAuditTests(unittest.TestCase):
    """Check saved-run comparisons and reject misleading input records."""

    def setUp(self):
        """Create a completed pilot whose two scoring keys disagree."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.items_path = self.root / "items.jsonl"
        items = [synthetic_item(i) for i in range(1, 43)]
        items[0] = Item(1, items[0].question, items[0].options, ["B"], ["C"])
        items[24] = Item(25, items[24].question, items[24].options, ["C", "E"], ["C"])
        self.items_path.write_text("".join(json.dumps(asdict(item)) + "\n" for item in items))
        self.run = self.root / "run"
        self.run.mkdir()
        self.manifest_path = self.run / "manifest.json"
        self.manifest_path.write_text(json.dumps({
            "status": "complete", "model": "synthetic-test-model", "item_order": [1, 2, 25],
            "items_sha256": sha256(self.items_path),
        }))
        self.records = []
        for condition, answers in {"no_memory": "CCB", "memory": "BCC", "full_metamo": "CBB"}.items():
            for item_id, answer in zip([1, 2, 25], answers):
                item = items[item_id - 1]
                self.records.append({
                    "condition": condition, "item_id": item_id, "answer": answer,
                    "source_correct": int(answer in item.source_answers),
                    "reference_compatible_correct": int(answer in item.reference_answers),
                })
        self.write_records()

    def write_records(self):
        """Save the current synthetic response records."""
        (self.run / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def test_audit_compares_both_keys_and_accounts_for_every_item(self):
        """Distinguish key-dependent improvements and retain all paired outcomes."""
        result = audit(run=self.run, items_path=self.items_path)
        self.assertEqual(result["source_vs_provisional_key_differences"], [1, 25])
        self.assertEqual(result["source_multiple_accepted_answers"], {"25": ["C", "E"]})
        self.assertEqual(result["scope"], "partial pilot, not a full STEU score")
        source = result["scoring"]["source"]
        reference = result["scoring"]["reference_compatible"]
        self.assertEqual(source["conditions"]["no_memory"]["incorrect_ids"], [1, 25])
        self.assertEqual(source["conditions"]["memory"]["correct"], 3)
        self.assertEqual(source["paired_changes"]["memory_vs_no_memory"]["improved_ids"], [1, 25])
        self.assertEqual(reference["paired_changes"]["memory_vs_no_memory"]["improved_ids"], [25])
        self.assertEqual(reference["paired_changes"]["memory_vs_no_memory"]["regressed_ids"], [1])
        for entry in result["scoring"].values():
            for comparison, changes in entry["paired_changes"].items():
                right, left = comparison.split("_vs_")
                grouped = [item_id for name, ids in changes.items() if name.endswith("_ids") for item_id in ids]
                self.assertEqual(sorted(grouped), [1, 2, 25])
                self.assertEqual(changes["net_correct_change"],
                                 entry["conditions"][right]["correct"] - entry["conditions"][left]["correct"])

    def test_shared_audit_preserves_published_results_and_report_names(self):
        """Keep the published CSV interface alongside experimental auditing."""
        reference = self.root / "reference.csv"
        reference.write_text("STEU,No Memory,No Memory Score,Memory,Memory Score,Appraisal,Appraisal Score,Output\n"
                             + "".join(f"{i},C,1,C,1,C,1,published explanation\n" for i in range(1, 43)))
        items = load_items(self.items_path)
        result = audit(reference, items)
        self.assertEqual(result["kind"], "published_reference_audit_not_a_new_metamo_run")
        self.assertEqual(result["conditions"]["no_memory"]["correct"], 42)
        self.assertEqual(result["conditions"]["memory"]["source_rule_correct"], 41)
        self.assertEqual(result["conditions"]["memory"]["source_score_disagreements"], [1])
        self.assertEqual(result["source_vs_provisional_key_differences"], [1, 25])
        self.assertEqual(result["paired_changes"]["memory_vs_no_memory"],
                         {"improved_ids": [], "regressed_ids": []})
        without_items = audit(reference)
        self.assertNotIn("source_url", without_items)
        self.assertEqual(without_items["conditions"]["memory"]["correct"], 42)
        output = self.root / "reports"
        write_audit(result, output)
        write_audit(audit(run=self.run, items_path=self.items_path), output)
        self.assertEqual(json.loads((output / "reference_audit.json").read_text()), result)
        self.assertIn("published STEU reference audit", (output / "reference_audit.md").read_text())
        self.assertTrue((output / "experiment_audit.md").exists())

    def test_shared_audit_rejects_ambiguous_inputs(self):
        """Require callers to select either published or experimental inputs."""
        for kwargs in [{}, {"run": self.run}, {"items_path": self.items_path},
                       {"reference": self.items_path, "run": self.run, "items_path": self.items_path}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit(**kwargs)

    def test_cli_writes_full_reports_without_gemini_or_changing_run_inputs(self):
        """Audit offline and display key checks and paired changes in both reports."""
        original = {path: path.read_bytes() for path in self.run.iterdir()}
        output = StringIO()
        with patch("benchmarks.phase1.GeminiJSON", side_effect=AssertionError("Gemini accessed")), redirect_stdout(output):
            self.assertEqual(main(["audit-experiment", str(self.run), "--items", str(self.items_path)]), 0)
        report = json.loads((self.run / "experiment_audit.json").read_text())
        self.assertEqual(report["kind"], "experimental_results_audit")
        markdown = (self.run / "experiment_audit.md").read_text()
        for text in ["[1, 25]", '"25": ["C", "E"]', "memory_vs_no_memory", "Unchanged correct IDs", "reference_compatible scoring"]:
            self.assertIn(text, markdown)
        self.assertIn("Experimental no_memory [source]: 1/3", output.getvalue())
        for path, contents in original.items():
            self.assertEqual(path.read_bytes(), contents)
        custom = self.root / "custom-report"
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["audit-experiment", str(self.run), "--items", str(self.items_path), "--output", str(custom)]), 0)
        self.assertTrue((custom / "experiment_audit.md").exists())

    def test_audit_rejects_changed_dataset_and_incomplete_run(self):
        """Prevent comparing a different key or treating interrupted runs as complete."""
        manifest = json.loads(self.manifest_path.read_text())
        self.manifest_path.write_text(json.dumps({**manifest, "status": "failed"}))
        with self.assertRaisesRegex(ValueError, "completed run"):
            audit(run=self.run, items_path=self.items_path)
        self.manifest_path.write_text(json.dumps(manifest))
        self.items_path.write_text(self.items_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "items_sha256"):
            audit(run=self.run, items_path=self.items_path)

    def test_audit_rejects_duplicate_missing_and_inconsistent_responses(self):
        """Reject records that would inflate counts or misstate correctness."""
        original = json.loads(json.dumps(self.records))
        for label, records, message in [
            ("duplicate", original + [original[0]], "Duplicate response"),
            ("missing", original[:-1], "missing responses"),
            ("inconsistent", [{**original[0], "source_correct": 1}] + original[1:], "inconsistent source_correct"),
        ]:
            with self.subTest(label=label):
                self.records = records
                self.write_records()
                with self.assertRaisesRegex(ValueError, message):
                    audit(run=self.run, items_path=self.items_path)

    def test_cli_failure_does_not_create_a_report(self):
        """Return an actionable error without writing a misleading audit."""
        self.records.pop()
        self.write_records()
        errors = StringIO()
        with redirect_stderr(errors):
            self.assertEqual(main(["audit-experiment", str(self.run), "--items", str(self.items_path)]), 1)
        self.assertIn("missing responses", errors.getvalue())
        self.assertFalse((self.run / "experiment_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
