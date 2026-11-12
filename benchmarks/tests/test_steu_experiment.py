"""Focused offline checks for the main STEU benchmark functions."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from benchmarks import config as benchmark_config
from benchmarks import steu_experiment as experiment_runner
from benchmarks.config import FEELING_WEIGHT, SELECTORS
from benchmarks.extended_feelings import FEELING_NAMES, all_feelings
from benchmarks.steu_experiment import ConditionSession, main, run_experiment
from benchmarks.helpers import (
    GeminiJSON, GeminiOutputError, api_retry_delay, build_prompt, feeling_bonus, initial_state,
    option_feeling, parse_plan, parse_stimulus, response_schema,
)
from benchmarks.steu_data import CONDITIONS, Item, audit, load_items, prepare, sha256, write_audit
from core import config as core_config
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
             patch("benchmarks.helpers.time.sleep"), redirect_stdout(StringIO()):
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

    def test_gemini_server_errors_back_off_preserve_the_prompt_and_report_http_status(self):
        from google.genai.errors import ServerError

        def unavailable():
            return ServerError(503, {"error": {"message": "Model experiencing high demand"}})

        with patch("google.genai.Client") as client, patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"}), \
             patch("benchmarks.helpers.time.sleep") as sleep, redirect_stdout(StringIO()):
            query = GeminiJSON("offline-model", 0, 42)
            generate = client.return_value.models.generate_content
            generate.side_effect = [unavailable(), unavailable(), SimpleNamespace(
                text='{"answer":"C"}', candidates=[], usage_metadata=None, model_version="offline")]
            prompt = build_prompt(synthetic_item(), "no_memory", [], None)
            data, metadata = query(prompt)
            self.assertEqual(data, {"answer": "C"})
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [30.0, 60.0])
            self.assertEqual([entry["status_code"] for entry in metadata["failed_attempts"]], [503, 503])
            self.assertTrue(all(call.kwargs["contents"] == prompt for call in generate.call_args_list))
            sleep.reset_mock()
            generate.side_effect = [unavailable() for _ in range(3)]
            with self.assertRaisesRegex(GeminiOutputError, "HTTP 503"):
                query(prompt)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [30.0, 60.0])
            hinted = unavailable()
            hinted.response = SimpleNamespace(headers={"Retry-After": "45"})
            self.assertEqual(api_retry_delay(hinted, 0), 46.0)

    def test_network_failures_retry_without_advancing_memory_or_changing_the_prompt(self):
        from importlib import import_module

        with patch("google.genai.Client") as client, patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"}), \
             patch("benchmarks.helpers.time.sleep") as sleep, redirect_stdout(StringIO()):
            query = GeminiJSON("offline-model", 0, 42)
            self.assertEqual(client.call_args.kwargs["http_options"].timeout, experiment_runner.REQUEST_TIMEOUT_MS)
            transport = import_module(query.retryable_network_errors[0].__module__.split(".")[0])
            generate = client.return_value.models.generate_content
            for error_class in (transport.ReadTimeout, transport.ConnectError, transport.RemoteProtocolError):
                with self.subTest(error=error_class.__name__):
                    session = ConditionSession("memory")
                    session.step(synthetic_item(), synthetic_query)
                    generate.reset_mock()
                    sleep.reset_mock()
                    generate.side_effect = [error_class("temporary network failure"), error_class("temporary network failure"),
                                            SimpleNamespace(text='{"answer":"C"}', candidates=[], usage_metadata=None,
                                                            model_version="offline")]
                    record = session.step(synthetic_item(2), query)
                    self.assertEqual(record["answer"], "C")
                    self.assertEqual(len(session.history), 2)
                    self.assertEqual(record["response"]["attempts"], 3)
                    self.assertTrue(all(entry["network_error"] for entry in record["response"]["failed_attempts"]))
                    self.assertEqual([call.kwargs["contents"] for call in generate.call_args_list], [record["prompt"]] * 3)
                    self.assertEqual([call.args[0] for call in sleep.call_args_list], [30.0, 60.0])
            generate.reset_mock()
            sleep.reset_mock()
            generate.side_effect = transport.UnsupportedProtocol("unsupported URL scheme")
            with self.assertRaises(transport.UnsupportedProtocol):
                query(record["prompt"])
            generate.assert_called_once()
            sleep.assert_not_called()

    def test_exhausted_network_retries_save_diagnostics_and_completed_answers(self):
        from importlib import import_module

        with tempfile.TemporaryDirectory() as directory, patch("google.genai.Client") as client, \
             patch("dotenv.load_dotenv"), patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"}), \
             patch("benchmarks.helpers.time.sleep"), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            query = GeminiJSON("offline-model", 0, 42)
            transport = import_module(query.retryable_network_errors[0].__module__.split(".")[0])
            client.return_value.models.generate_content.side_effect = [
                SimpleNamespace(text='{"answer":"C"}', candidates=[], usage_metadata=None, model_version="offline"),
                *[transport.ReadTimeout("The read operation timed out") for _ in range(3)],
            ]
            output = Path(directory) / "run"
            with self.assertRaisesRegex(GeminiOutputError, "network error.*ReadTimeout"):
                run_experiment([synthetic_item(), synthetic_item(2)], query, output, {}, conditions=["memory"])
            manifest = json.loads((output / "manifest.json").read_text())
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_items"], 1)
            self.assertEqual(failure["item_id"], 2)
            self.assertEqual(len(failure["attempts"]), 3)
            self.assertTrue(all(entry["error_type"] == "ReadTimeout" and entry["network_error"] for entry in failure["attempts"]))
            records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            self.assertEqual([(record["item_id"], record["answer"]) for record in records], [(1, "C")])

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
            with patch.object(benchmark_config, "EXPERIMENT_TAG", "config-snapshot-test", create=True):
                summary = run_experiment(items[:2], synthetic_query, output, {"items_sha256": sha256(items_path)})
            report = audit(run=output, items_path=items_path)
            write_audit(report, output)
            self.assertTrue((output / "experiment_audit.md").exists())
            self.assertEqual(len((output / "responses.jsonl").read_text().splitlines()), 6)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["config"]["benchmarks.config"]["EXPERIMENT_TAG"], "config-snapshot-test")
            for module in [benchmark_config, core_config]:
                expected = {name: value for name, value in vars(module).items()
                            if name.isupper() and not name.startswith("_")}
                saved = {name: value for name, value in manifest["config"][module.__name__].items()
                         if name != "EXPERIMENT_TAG"}
                self.assertEqual(saved, json.loads(json.dumps(expected)))
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
                saved = json.loads((selected_run / "manifest.json").read_text())
                self.assertEqual(saved["config"]["benchmarks.config"]["FEELING_WEIGHT"], FEELING_WEIGHT)
                self.assertEqual(saved["feeling_weight"],
                                 {"magus_only": 0.0, "emotion_only": 1.0}.get(selector, FEELING_WEIGHT))
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
            original_config = json.loads((output / "manifest.json").read_text())["config"]
            pending = Mock(side_effect=synthetic_query)
            with self.assertRaisesRegex(ValueError, "selector"):
                run_experiment(items, pending, output, metadata, resume=True, selector="magus_only")
            pending.assert_not_called()
            with patch.object(benchmark_config, "FEELING_WEIGHT", FEELING_WEIGHT + 1):
                actual = run_experiment(items, pending, output, metadata, resume=True)
            resumed = json.loads((output / "manifest.json").read_text())
            self.assertEqual(resumed["config"], original_config)
            self.assertEqual(resumed["resume_history"][-1]["config"]["benchmarks.config"]["FEELING_WEIGHT"],
                             FEELING_WEIGHT + 1)
            self.assertEqual(pending.call_count, 3)
            self.assertEqual(actual, expected)
            self.assertTrue((output / "responses.jsonl").read_bytes().startswith(saved))
            records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            expected_records = [json.loads(line) for line in (baseline / "responses.jsonl").read_text().splitlines()]
            self.assertEqual([(r["prompt"], r["state_after"]) for r in records],
                             [(r["prompt"], r["state_after"]) for r in expected_records])

    def test_main_prepares_runs_all_five_cases_and_generates_graphs_without_arguments(self):
        with tempfile.TemporaryDirectory() as directory, patch("dotenv.load_dotenv"), \
             patch("benchmarks.steu_experiment.GeminiJSON", return_value=Mock(side_effect=synthetic_query)) as gemini, \
             redirect_stdout(StringIO()):
            root = Path(directory)
            with patch.multiple(experiment_runner, DEFAULT_ITEMS=root / "data/items.jsonl",
                                RESULTS_DIRECTORY=str(root / "results"), RESUME_DIRECTORY=None,
                                MODEL="offline-config-model", ITEM_LIMIT=2, EMOTION_ITEMS_ONLY=True,
                                TEMPERATURE=0.2, SEED=7, REQUEST_INTERVAL=0.0), \
                 patch.multiple(benchmark_config, COMPARE_SELECTORS=True, SELECTOR="magus_with_emotion",
                                CONDITIONS=tuple(CONDITIONS), FEELING_WEIGHT=1.0):
                self.assertEqual(main([]), 0)
            gemini.assert_called_once_with("offline-config-model", 0.2, 7)
            self.assertEqual(gemini.return_value.call_count, 10)
            output, = (root / "results").iterdir()
            experiment = json.loads((output / "experiment.json").read_text())
            self.assertEqual(experiment["status"], "complete")
            self.assertNotIn("items_file", experiment["settings"])
            self.assertEqual(experiment["item_order"], [1, 2])
            self.assertEqual(experiment["available_cases"],
                             ["no_memory", "memory", "emotion_only", "magus_only", "magus_with_emotion"])
            self.assertEqual(experiment["metadata"]["excluded_item_ids"], [5, 10, 19, 20, 29, 41])
            self.assertEqual(experiment["graphs"], ["accuracy_all", "accuracy_selectors", "question_correctness"])
            plots = output / "plots"
            for graph in experiment["graphs"]:
                self.assertTrue((plots / f"{graph}.png").read_bytes().startswith(b"\x89PNG"))
            data = json.loads((plots / "analysis.json").read_text())
            self.assertEqual(data["conditions"]["no_memory"]["scoring"]["source"]["total"], 2)
            self.assertEqual({path.stem for path in plots.glob("*.png")}, set(experiment["graphs"]))
            self.assertEqual({path.name for path in plots.iterdir()},
                             {f"{graph}.png" for graph in experiment["graphs"]} | {"analysis.json", "question_results.csv"})
            for name in ["analysis.json", "question_results.csv"]:
                self.assertTrue((plots / name).is_file())
            for entry in experiment["runs"]:
                manifest = json.loads((output / entry["directory"] / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["config"]["benchmarks.config"]["COMPARE_SELECTORS"], True)

    def test_configured_resume_retains_completed_cases_and_then_plots(self):
        from benchmarks.generate_plots import load_comparison

        def analyze(runs, items, output):
            _, data = load_comparison(runs, items)
            return {**data, "graphs": []}

        with tempfile.TemporaryDirectory() as directory, patch("dotenv.load_dotenv"), \
             patch("benchmarks.generate_plots.analyze_runs", side_effect=analyze) as plots, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(directory)
            calls = 0

            def query(prompt):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("synthetic service interruption")
                return synthetic_query(prompt)

            with patch.multiple(experiment_runner, DEFAULT_ITEMS=root / "items.jsonl",
                                RESULTS_DIRECTORY=str(root / "results"), RESUME_DIRECTORY=None,
                                MODEL="offline", ITEM_LIMIT=1, EMOTION_ITEMS_ONLY=True, REQUEST_INTERVAL=0.0), \
                 patch.multiple(benchmark_config, COMPARE_SELECTORS=True, SELECTOR="magus_with_emotion",
                                CONDITIONS=tuple(CONDITIONS), FEELING_WEIGHT=1.0), \
                 patch("benchmarks.steu_experiment.GeminiJSON", return_value=query):
                self.assertEqual(main([]), 1)
            plots.assert_not_called()
            output, = (root / "results").iterdir()
            completed = output / "primary/responses.jsonl"
            saved = completed.read_bytes()
            # Older checkpoints with this field still resume and drop it on save.
            checkpoint = json.loads((output / "experiment.json").read_text())
            checkpoint["settings"]["items_file"] = str(root / "items.jsonl")
            (output / "experiment.json").write_text(json.dumps(checkpoint))
            pending = Mock(side_effect=synthetic_query)
            with patch.multiple(experiment_runner, RESUME_DIRECTORY=str(output), DEFAULT_ITEMS=root / "items.jsonl"), \
                 patch.multiple(benchmark_config, SELECTOR="emotion_only", FEELING_WEIGHT=9.0), \
                 patch("benchmarks.steu_experiment.GeminiJSON", return_value=pending):
                self.assertEqual(main([]), 0)
            self.assertEqual(pending.call_count, 2)
            self.assertEqual(completed.read_bytes(), saved)
            experiment = json.loads((output / "experiment.json").read_text())
            self.assertEqual(experiment["settings"]["feeling_weight"], 1.0)
            self.assertEqual(experiment["status"], "complete")
            self.assertNotIn("items_file", experiment["settings"])
            resumed_manifest = json.loads((output / "emotion_only/manifest.json").read_text())
            self.assertEqual(resumed_manifest["resume_history"][-1]["request_timeout_ms"], experiment_runner.REQUEST_TIMEOUT_MS)
            new_manifest = json.loads((output / "magus_only/manifest.json").read_text())
            self.assertEqual(new_manifest["request_timeout_ms"], experiment_runner.REQUEST_TIMEOUT_MS)
            plots.assert_called_once()
            # Reopening a finished experiment only regenerates its plots.
            with patch.multiple(experiment_runner, RESUME_DIRECTORY=str(output), DEFAULT_ITEMS=root / "items.jsonl"), \
                 patch("benchmarks.steu_experiment.GeminiJSON") as gemini:
                self.assertEqual(main([]), 0)
            gemini.assert_not_called()

    def test_single_selector_configuration_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch("dotenv.load_dotenv"), \
             patch("benchmarks.steu_experiment.GeminiJSON", return_value=Mock(side_effect=synthetic_query)) as gemini, \
             patch("benchmarks.generate_plots.analyze_runs", return_value={"graphs": [], "available_cases": ["magus_only"],
                                                                   "missing_cases": ["no_memory", "memory", "emotion_only", "magus_with_emotion"]}), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(directory)
            with patch.multiple(experiment_runner, DEFAULT_ITEMS=root / "items.jsonl",
                                RESULTS_DIRECTORY=str(root / "results"), RESUME_DIRECTORY=None,
                                MODEL="offline", ITEM_LIMIT=1, EMOTION_ITEMS_ONLY=True, REQUEST_INTERVAL=0.0), \
                 patch.multiple(benchmark_config, COMPARE_SELECTORS=False,
                                SELECTOR="magus_only", CONDITIONS=("full_metamo",)):
                self.assertEqual(main([]), 0)
                output, = (root / "results").iterdir()
                experiment = json.loads((output / "experiment.json").read_text())
                self.assertEqual(len(experiment["runs"]), 1)
                self.assertEqual(experiment["runs"][0]["selector"], "magus_only")
                self.assertEqual(gemini.return_value.call_count, 1)
                gemini.reset_mock()
                with patch.object(experiment_runner, "ITEM_LIMIT", 0):
                    self.assertEqual(main([]), 1)
                self.assertEqual(main(["--selector", "emotion_only"]), 1)
                gemini.assert_not_called()


if __name__ == "__main__":
    unittest.main()
