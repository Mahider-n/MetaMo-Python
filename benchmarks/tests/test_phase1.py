"""Offline protocol checks."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.phase1 import ConditionSession, GeminiJSON, GeminiOutputError, build_prompt, parse_plan, parse_stimulus, run_experiment
from benchmarks.steu import Item, load_items, prepare


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


if __name__ == "__main__":
    unittest.main()
