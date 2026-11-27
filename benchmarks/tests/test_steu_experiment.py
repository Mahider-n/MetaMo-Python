"""Focused offline tests for the main STEU experiment workflows."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmarks import config, steu_experiment as runner
from benchmarks.helpers import initial_state, parse_plan, parse_stimulus
from benchmarks.scorers.llm import GeminiJSON, build_prompt
from benchmarks.scorers.selection import SELECTORS
from benchmarks.steu_data import CONDITIONS, Item, audit, load_items

METADATA = {"model": "offline", "temperature": 0, "seed": 42,
            "items_sha256": "fixture", "max_output_tokens": 16384}


def synthetic_item(item_id=1):
    """Create an emotion-choice question with C as its accepted answer."""
    return Item(item_id, "Which emotion is likely?",
                dict(zip("ABCDE", ("Joy", "Sad", "Hope", "Angry", "Scared"))), ["C"], ["C"])


def synthetic_query(prompt):
    """Return valid model responses without making API requests."""
    condition = json.loads(prompt.split("\n\nDATA:\n", 1)[1]).get("condition")
    if condition != "full_metamo":
        return {"answer": "C"}, {}
    candidates = [{"id": letter, "risk_estimate": 0.1,
                   "plausibility": 0.6 if letter == "C" else 0.1,
                   "goal_correlations": [0, 0, 0.8 if letter == "C" else 0.1, 0, 0, 0, 0, 0],
                   "delta_g": [0, 0, 0.01, 0, 0, 0, 0, 0]} for letter in "ABCDE"]
    return {"candidates": candidates,
            "stimulus": {"novelty": 0.3, "conduciveness": -0.5, "risk": 0.8, "effort": 0.3}}, {}


class ExperimentTests(unittest.TestCase):
    def test_prompts_include_history_without_answer_keys(self):
        item = synthetic_item()
        item = Item(item.id, item.question, item.options, ["SOURCE_KEY"], ["REFERENCE_KEY"])
        history = [{"item": item.public(), "answer": "C"}]
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                prompt = build_prompt(item, condition, history, initial_state())
                payload = json.loads(prompt.split("\n\nDATA:\n", 1)[1])
                self.assertNotIn("SOURCE_KEY", prompt)
                self.assertNotIn("REFERENCE_KEY", prompt)
                self.assertEqual(payload.get("previous_interactions", []),
                                 [] if condition == "no_memory" else history)

    def test_candidate_and_stimulus_parsing(self):
        with self.assertRaisesRegex(ValueError, "Unknown condition"):
            runner.ConditionSession("appraisal_prompting")
        data, _ = synthetic_query(build_prompt(synthetic_item(), "full_metamo", [], initial_state()))
        self.assertEqual([action.id for action in parse_plan(data)], list("ABCDE"))
        self.assertEqual(parse_stimulus(data["stimulus"]).risk, 0.8)
        data["candidates"][-1]["id"] = "A"
        with self.assertRaises(ValueError):
            parse_plan(data)
        with self.assertRaises(ValueError):
            parse_stimulus({**data["stimulus"], "conduciveness": 2})

    def test_sessions_manage_state_history_and_emotion_selection(self):
        for condition in CONDITIONS:
            policies = config.STATE_POLICIES if condition == "full_metamo" else ("independent",)
            for policy in policies:
                with self.subTest(condition=condition, policy=policy):
                    session = runner.ConditionSession(condition, selector="emotion_only",
                        feeling_scorer="quadratic", state_policy=policy, history_policy="full")
                    first = session.step(synthetic_item(1), synthetic_query)
                    second = session.step(synthetic_item(2), synthetic_query)
                    self.assertEqual(second["history_items_before"], 0 if condition == "no_memory" else 1)
                    if condition == "full_metamo":
                        expected = first["state_before"] if policy == "independent" else first["state_after"]
                        self.assertEqual(second["state_before"], expected)
                        scores = second["feeling_choice"]["emotion_scores"]
                        self.assertEqual(second["answer"], max(scores, key=scores.get))
                    else:
                        self.assertEqual(second["answer"], "C")
                        self.assertIsNone(second["state_after"])

    def test_gemini_retries_invalid_answers(self):
        responses = [SimpleNamespace(text=json.dumps({"answer": answer}), candidates=[],
                                    usage_metadata=None, model_version="offline") for answer in ("F", "C")]
        with patch("google.genai.Client") as client, patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"}), \
             patch("benchmarks.scorers.llm.time.sleep"), redirect_stdout(StringIO()):
            client.return_value.models.generate_content.side_effect = responses
            prompt = build_prompt(synthetic_item(), "memory", [], None)
            data, metadata = GeminiJSON("offline", 0, 42)(prompt)
        self.assertEqual(data["answer"], "C")
        self.assertEqual(metadata["attempts"], 2)
        self.assertIn("Expected a single answer letter", metadata["retry_prompt"])
        self.assertTrue(metadata["retry_prompt"].endswith(prompt))

    def test_resume_keeps_completed_answers_and_restores_state(self):
        items = [synthetic_item(1), synthetic_item(2)]
        first = synthetic_query(build_prompt(items[0], "full_metamo", [], initial_state()))
        settings = dict(conditions=["full_metamo"], selector="emotion_only", feeling_scorer="quadratic",
                        state_policy="persistent", history_policy="full")
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            output = Path(temporary) / "run"
            with self.assertRaises(RuntimeError):
                runner.run_experiment(items, Mock(side_effect=[first, RuntimeError("stop")]),
                                      output, METADATA, **settings)
            retained = (output / "responses.jsonl").read_bytes()
            pending = Mock(side_effect=synthetic_query)
            runner.run_experiment(items, pending, output, METADATA, resume=True, **settings)
            pending.assert_called_once()
            self.assertTrue((output / "responses.jsonl").read_bytes().startswith(retained))
            records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(records[1]["state_before"], records[0]["state_after"])
            self.assertEqual(records[1]["history_items_before"], 1)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")

    def test_configured_run_prepares_data_audits_results_and_generates_plots(self):
        query = Mock(side_effect=synthetic_query)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()), \
             patch("dotenv.load_dotenv"), patch("benchmarks.steu_experiment.GeminiJSON", return_value=query):
            root = Path(temporary)
            with patch.multiple(config, CONDITIONS=tuple(CONDITIONS), COMPARE_SELECTORS=True,
                                SELECTOR="magus_with_emotion", FEELING_SCORER="normalized",
                                FEELING_WEIGHT=1.0, STATE_POLICY="independent", HISTORY_POLICY="full"), \
                 patch.multiple(runner, DEFAULT_ITEMS=root / "items.jsonl", RESULTS_DIRECTORY=root / "results",
                                MODEL="offline", ITEM_LIMIT=2,
                                EMOTION_ITEMS_ONLY=True, REQUEST_INTERVAL=0.0):
                output = runner.run_configured_experiment()
            experiment = json.loads((output / "experiment.json").read_text())
            self.assertEqual(experiment["status"], "complete")
            self.assertEqual(len(load_items(root / "items.jsonl")), 42)
            self.assertEqual(query.call_count, 2 * (len(CONDITIONS) + len(SELECTORS) - 1))
            self.assertEqual(audit(run=output / "primary", items_path=root / "items.jsonl")["item_count"], 2)
            self.assertEqual(len(experiment["graphs"]), 3)
            self.assertTrue(all((output / "plots" / (name + ".png")).is_file() for name in experiment["graphs"]))
            self.assertTrue((output / "plots/question_results.csv").is_file())


if __name__ == "__main__":
    unittest.main()
