"""Focused offline checks for the main STEU benchmark functions."""

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

from benchmarks.config import FEELING_WEIGHT, SELECTORS
from benchmarks.extended_feelings import FEELING_NAMES, all_feelings
from benchmarks.steu_experiment import (
    ConditionSession, GeminiJSON, build_prompt, feeling_bonus, initial_state,
    main, option_feeling, parse_plan, parse_stimulus, response_schema, run_experiment,
)
from benchmarks.steu import CONDITIONS, Item, audit, load_items, prepare, sha256, write_audit
from openpsi.feelings import feelings as native_feelings


def synthetic_item(item_id=1):
    """Invent a question whose expected answer is C."""
    return Item(item_id, "Which emotion is likely?",
                {letter: f"Invented option {letter}" for letter in "ABCDE"}, ["C"], ["C"])


def synthetic_query(prompt):
    """Provide predictable responses without contacting Gemini."""
    if json.loads(prompt.split("\n\nDATA:\n", 1)[1]).get("condition") != "full_metamo":
        return {"answer": "C"}, {}
    candidates = [{
        "id": letter, "risk_estimate": 0.1,
        "goal_correlations": [0, 0, 0.8 if letter == "C" else 0.1, 0, 0, 0, 0, 0],
        "delta_g": [0, 0, 0.01, 0, 0, 0, 0, 0],
    } for letter in "ABCDE"]
    return {"candidates": candidates, "stimulus": {
        "novelty": 0.3, "conduciveness": -0.5, "risk": 0.8, "effort": 0.3,
    }}, {}


class MainFunctionTests(unittest.TestCase):
    def test_build_prompt_adds_only_history_and_excludes_answer_keys(self):
        item = synthetic_item()
        item = Item(item.id, item.question, item.options, ["SOURCE_KEY"], ["REFERENCE_KEY"])
        baseline = build_prompt(item, "no_memory", [], None)
        self.assertEqual(build_prompt(item, "memory", [], None), baseline)
        self.assertEqual(response_schema("memory"), response_schema("no_memory"))
        history = [{"item": item.public(), "answer": "C"}]
        instruction, data = build_prompt(item, "memory", history, None).split("\n\nDATA:\n")
        baseline_instruction, baseline_data = baseline.split("\n\nDATA:\n")
        payload = json.loads(data)
        self.assertEqual(payload.pop("previous_interactions"), history)
        self.assertEqual(payload, json.loads(baseline_data))
        self.assertEqual(instruction, baseline_instruction)
        for condition in CONDITIONS:
            prompt = build_prompt(item, condition, history, initial_state())
            self.assertNotIn("SOURCE_KEY", prompt)
            self.assertNotIn("REFERENCE_KEY", prompt)

    def test_parse_plan_and_stimulus_validate_model_data(self):
        data, _ = synthetic_query(build_prompt(synthetic_item(), "full_metamo", [], initial_state()))
        self.assertEqual([action.id for action in parse_plan(data)], list("ABCDE"))
        self.assertEqual(parse_stimulus(data["stimulus"]).risk, 0.8)
        data["candidates"][-1]["id"] = "A"
        with self.assertRaises(ValueError):
            parse_plan(data)
        with self.assertRaises(ValueError):
            parse_stimulus({**data["stimulus"], "conduciveness": 2})

    def test_option_feeling_preserves_variants_and_distinct_emotions(self):
        expected = {" Angry ": "anger", "anger": "anger", "frustrated": "frustration",
                    "hopeful": "hope", "relieved": "relief", "guilty": "guilt",
                    "joy": "happiness", "scared": "fear", "contemptuous": "contempt"}
        for label, name in expected.items():
            self.assertEqual(option_feeling(label), name)
        for name in FEELING_NAMES:
            self.assertEqual(option_feeling(name), name)
        self.assertIsNone(option_feeling("An event happens"))

    def test_feeling_bonus_centres_known_options(self):
        item = Item(1, "Which emotion?", dict(zip("ABCDE",
                    ["Hope", "Relief", "Unknown", "Angry", "Bored"])), ["A"], ["A"])
        bonus = feeling_bonus(item, {"hope": 0.6, "relief": 0.3, "anger": 0.0})
        self.assertAlmostEqual(bonus["A"], 0.3)
        self.assertAlmostEqual(bonus["D"], -0.3)
        self.assertAlmostEqual(sum(bonus.values()), 0)
        self.assertEqual((bonus["C"], bonus["E"]), (0, 0))

    def test_all_feelings_preserves_native_scores(self):
        state = initial_state()
        before = state.copy()
        native, combined = native_feelings(state), all_feelings(state)
        self.assertEqual(len(combined), 26)
        self.assertEqual({name: combined[name] for name in native}, native)
        self.assertTrue(all(np.isfinite(value) and 0 <= value <= 1 for value in combined.values()))
        np.testing.assert_array_equal(state.G, before.G)
        np.testing.assert_array_equal(state.M, before.M)

    def test_direct_conditions_use_gemini_and_keep_history_only_for_memory(self):
        with patch("benchmarks.steu_experiment.initial_state", side_effect=AssertionError("state used")), \
             patch("benchmarks.steu_experiment.MagusDecision", side_effect=AssertionError("Magus used")), \
             patch("benchmarks.steu_experiment.OpenPsiAppraisal", side_effect=AssertionError("appraisal used")):
            for condition in ["no_memory", "memory"]:
                session = ConditionSession(condition)
                records = [session.step(synthetic_item(i), synthetic_query) for i in [1, 2]]
                self.assertEqual([r["answer"] for r in records], ["C", "C"])
                self.assertIsNone(session.state)
                self.assertEqual([r["history_items_before"] for r in records],
                                 [0, 1] if condition == "memory" else [0, 0])
                self.assertEqual(len(session.history), 2 if condition == "memory" else 0)

    def test_full_metamo_appraises_and_applies_the_feeling_weight(self):
        item = Item(1, "Which emotion?", dict(zip("ABCDE",
                    ["Joy", "Sad", "Hope", "Angry", "Scared"])), ["C"], ["C"])

        def query(prompt):
            data, metadata = synthetic_query(prompt)
            data["candidates"][0]["goal_correlations"][2] = 0.7
            return data, metadata

        # A is a compromise, B has the strongest emotion, C has the best Magus score.
        felt = dict(zip(["happiness", "sadness", "hope", "anger", "fear"], [0.79, 0.8, 0, 0, 0]))
        prompts = []
        for selector, answer in [("emotion_only", "B"), ("magus_only", "C"), ("magus_with_emotion", "A")]:
            session = ConditionSession("full_metamo", feeling_weight=1, selector=selector)
            with patch("benchmarks.steu_experiment.feelings", return_value=felt):
                record = session.step(item, query)
            prompts.append(record["prompt"])
            self.assertEqual(record["answer"], answer)
            self.assertEqual(record["selector"], selector)
            self.assertEqual(record["feeling_choice"]["magus_pick"], "C")
            self.assertEqual(record["feeling_choice"]["emotion_pick"], "B")
            self.assertEqual(session.history[0]["answer"], answer)
            self.assertNotEqual(record["state_before"]["M"], record["state_after"]["M"])
        self.assertEqual(len(set(prompts)), 1)
        without_bonus = ConditionSession("full_metamo", feeling_weight=0).step(item, synthetic_query)
        self.assertEqual(without_bonus["answer"], without_bonus["feeling_choice"]["magus_pick"])
        unsupported_query = Mock()
        with self.assertRaisesRegex(ValueError, "emotion names"):
            ConditionSession("full_metamo", selector="emotion_only").step(synthetic_item(), unsupported_query)
        unsupported_query.assert_not_called()

    def test_gemini_json_retries_an_invalid_answer_without_changing_the_prompt(self):
        def response(text):
            return SimpleNamespace(text=text, candidates=[], usage_metadata=None, model_version="offline")
        with patch("google.genai.Client") as client, patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"}), \
             patch("benchmarks.steu_experiment.time.sleep"), redirect_stdout(StringIO()):
            query = GeminiJSON("offline-model", 0, 42)
            client.return_value.models.generate_content.side_effect = [
                response('{"answer":"F"}'), response('{"answer":"C"}'),
            ]
            prompt = build_prompt(synthetic_item(), "memory", [], None)
            data, metadata = query(prompt)
        self.assertEqual(data, {"answer": "C"})
        self.assertEqual(metadata["attempts"], 2)
        for call in client.return_value.models.generate_content.call_args_list:
            self.assertEqual(call.kwargs["contents"], prompt)
            self.assertEqual(call.kwargs["config"].response_json_schema, response_schema("memory"))

    def test_prepare_run_and_audit_work_together(self):
        benchmarks = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            items_path, output = root / "items.jsonl", root / "run"
            items = prepare(benchmarks / "data/steu_author_source.txt",
                            benchmarks / "qpt6z-osfstorage-archive/experiment_1_STEU.csv", items_path)
            self.assertEqual(len(items), 42)
            self.assertEqual(load_items(items_path), items)
            for item in items:
                names = [option_feeling(text) for text in item.options.values() if len(text.split()) == 1]
                self.assertNotIn(None, names)
                self.assertEqual(len(names), len(set(names)), f"Category collision: item {item.id}")
            summary = run_experiment(items[:2], synthetic_query, output, {"items_sha256": sha256(items_path)})
            report = audit(run=output, items_path=items_path)
            write_audit(report, output)
            self.assertTrue((output / "experiment_audit.md").exists())
            self.assertEqual(len((output / "responses.jsonl").read_text().splitlines()), 6)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")
            self.assertEqual(set(report["scoring"]), {"source", "reference_compatible"})
            for condition in CONDITIONS:
                self.assertEqual(summary["conditions"][condition]["total"], 2)
            shared = [item for item in items if all(option_feeling(text) for text in item.options.values())]
            excluded = [item.id for item in items if item not in shared]
            self.assertEqual(len(shared), 36)
            for selector in SELECTORS:
                selected_run = root / selector
                selected = run_experiment(shared, synthetic_query, selected_run,
                                          {"items_sha256": sha256(items_path), "emotion_items_only": True,
                                           "excluded_item_ids": excluded, "scope": "emotion-label subset"},
                                          conditions=["full_metamo"], selector=selector)
                checked = audit(run=selected_run, items_path=items_path)
                write_audit(checked, selected_run)
                self.assertEqual(selected["selector"], selector)
                self.assertEqual(checked["selector"], selector)
                self.assertEqual(checked["item_order"], [item.id for item in shared])
                self.assertEqual(checked["excluded_item_ids"], [5, 10, 19, 20, 29, 41])
                self.assertEqual(selected["conditions"]["full_metamo"]["total"], 36)
                self.assertEqual(set(checked["scoring"]["source"]["conditions"]), {"full_metamo"})
                self.assertIn(selector, (selected_run / "experiment_audit.md").read_text())

    def test_run_experiment_resumes_with_the_same_history_and_state(self):
        items = [synthetic_item(1), synthetic_item(2)]
        metadata = {"model": "offline", "temperature": 0, "seed": 42,
                    "items_sha256": "fixture", "max_output_tokens": 16384}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(directory)
            output, baseline = root / "interrupted", root / "baseline"
            expected = run_experiment(items, synthetic_query, baseline, metadata)
            interrupted = Mock(side_effect=[({"answer": "C"}, {})] * 3 + [RuntimeError("interrupted")])
            with self.assertRaises(RuntimeError):
                run_experiment(items, interrupted, output, metadata)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")
            saved = (output / "responses.jsonl").read_bytes()
            pending = Mock(side_effect=synthetic_query)
            with self.assertRaisesRegex(ValueError, "selector"):
                run_experiment(items, pending, output, metadata, resume=True, selector="magus_only")
            pending.assert_not_called()
            actual = run_experiment(items, pending, output, metadata, resume=True)
            self.assertEqual(pending.call_count, 3)
            self.assertEqual(actual, expected)
            self.assertTrue((output / "responses.jsonl").read_bytes().startswith(saved))
            records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            expected_records = [json.loads(line) for line in (baseline / "responses.jsonl").read_text().splitlines()]
            self.assertEqual([(r["prompt"], r["state_after"]) for r in records],
                             [(r["prompt"], r["state_after"]) for r in expected_records])

    def test_main_uses_configured_weight_and_accepts_a_cli_override(self):
        with tempfile.TemporaryDirectory() as directory, patch("dotenv.load_dotenv"), \
             patch("benchmarks.steu_experiment.GeminiJSON"), \
             patch("benchmarks.steu_experiment.run_experiment", return_value={}) as run, redirect_stdout(StringIO()):
            items_path = Path(directory) / "items.jsonl"
            items_path.write_text("".join(json.dumps(asdict(synthetic_item(i))) + "\n" for i in range(1, 43)))
            for override, expected in [([], FEELING_WEIGHT), (["--feeling-weight", "0"], 0.0)]:
                self.assertEqual(main(["run", "--items", str(items_path), "--limit", "1", *override]), 0)
                self.assertEqual(run.call_args.kwargs["feeling_weight"], expected)
            options = dict(zip("ABCDE", ["Joy", "Sad", "Hope", "Angry", "Scared"]))
            items = [Item(i, "Which emotion?", options, ["C"], ["C"]) for i in range(1, 43)]
            items[4] = synthetic_item(5)  # Event choices have no emotion-name mapping.
            items_path.write_text("".join(json.dumps(asdict(item)) + "\n" for item in items))
            for selector in SELECTORS:
                self.assertEqual(main(["run", "--items", str(items_path), "--only", "full_metamo",
                                       "--selector", selector, "--emotion-items-only"]), 0)
                self.assertEqual(run.call_args.kwargs["selector"], selector)
                self.assertEqual(len(run.call_args.args[0]), 41)
                self.assertEqual(run.call_args.args[3]["excluded_item_ids"], [5])
            (Path(directory) / "manifest.json").write_text(json.dumps({
                "model": "offline", "temperature": 0, "seed": 42, "limit": 42,
                "item_order": [i.id for i in items if i.id != 5], "selector": "emotion_only",
                "emotion_items_only": True, "conditions": ["full_metamo"], "feeling_weight": 1,
            }))
            self.assertEqual(main(["run", "--items", str(items_path), "--resume", directory]), 0)
            self.assertEqual(run.call_args.kwargs["selector"], "emotion_only")
            self.assertEqual(len(run.call_args.args[0]), 41)


if __name__ == "__main__":
    unittest.main()
