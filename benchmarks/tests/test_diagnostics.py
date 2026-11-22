"""Focused offline checks for scorers, selectors, and diagnostic reports."""

from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks import steu_experiment as runner
from benchmarks.diagnostics import analyze_run, dominance_certificate, pairwise_audit, representation_audit
from benchmarks.generate_plots import analyze_runs, load_comparison
from benchmarks.helpers import initial_state, option_feeling, snapshot
from benchmarks.scorers.feelings import SCORERS, score_feelings, score_values, state_values
from benchmarks.scorers.selection import SELECTORS, select_answer
from benchmarks.scorers.templates import all_feelings, normalized_feelings
from benchmarks.steu_data import Item, prepare, sha256
from benchmarks.tests.test_steu_experiment import METADATA, synthetic_query


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.items_path = Path(temporary.name) / "items.jsonl"
        cls.items = prepare(runner.BENCHMARKS_DIR / "data/steu_author_source.txt",
                            runner.DEFAULT_REFERENCE, cls.items_path)
        cls.emotion_items = [item for item in cls.items if all(option_feeling(v) for v in item.options.values())]

    def test_registered_scorers_preserve_reference_scores_and_input_dimensions(self):
        state = initial_state()
        state.M = np.array([.2, .8, .3, .1, .7, .9])
        for scorer in SCORERS:
            with self.subTest(scorer=scorer):
                values = state_values(state, scorer)
                self.assertEqual(len(values), 6 if scorer == "quadratic" else 5)
                actual = score_feelings(state, scorer)
                vectorized = score_values(values, scorer)
                self.assertEqual(len(actual), 26)
                for name, value in actual.items():
                    self.assertTrue(np.isfinite(value))
                    self.assertAlmostEqual(value, float(vectorized[name]))
        for scorer, expected in (("legacy", all_feelings(state)), ("normalized", normalized_feelings(state))):
            for name, value in score_feelings(state, scorer).items():
                self.assertAlmostEqual(value, expected[name])

    def test_all_selection_rules_choose_the_expected_answer(self):
        magus = dict(zip("ABCDE", [0, .1, .9, .2, 0]))
        emotion = dict(zip("ABCDE", [.2, .8, .1, .3, .8]))
        plausibility = dict(zip("ABCDE", [.05, .1, .1, .05, .7]))
        combined = dict(zip("ABCDE", [.3, .5, .9, 1, .4]))
        expected = {"emotion_only": "B", "magus_only": "C", "magus_with_emotion": "D",
                    "plausibility": "E", "emotion_then_plausibility": "E", "rank_fusion": "E"}
        self.assertEqual(set(expected), set(SELECTORS))
        for selector, answer in expected.items():
            with self.subTest(selector=selector):
                self.assertEqual(select_answer(selector, magus, emotion, plausibility, combined)[0], answer)

    def test_pairwise_audit_covers_every_competing_emotion_pair(self):
        expected = {pair for item in self.emotion_items
                    for pair in combinations(sorted({option_feeling(v) for v in item.options.values()}), 2)}
        result = pairwise_audit(self.items, [0., 1.])
        for scorer in SCORERS:
            rows = [row for row in result["pairs"] if row["scorer"] == scorer]
            self.assertEqual({(row["emotion_a"], row["emotion_b"]) for row in rows}, expected)
            for row in rows:
                self.assertEqual(row["sample_count"], 2 ** (6 if scorer == "quadratic" else 5))
                self.assertEqual(row["a_wins"] + row["b_wins"] + row["ties"], row["sample_count"])
        self.assertTrue(dominance_certificate("legacy", "excitement", "anger")["certified"])
        self.assertFalse(dominance_certificate("normalized", "excitement", "anger")["certified"])

    def test_representation_audit_reports_identical_inputs_with_different_targets(self):
        options = dict(zip("ABCDE", ("Angry", "Guilty", "Happy", "Scared", "Surprised")))
        items = {1: Item(1, "Unfair treatment", options, ["A"], ["A"]),
                 2: Item(2, "Harmed a friend", options, ["B"], ["B"])}
        records = [{"item_id": i, "stimulus": dict(novelty=.5, conduciveness=-.5, risk=.5, effort=.5),
                    "decision_state": snapshot(initial_state())} for i in items]
        result = representation_audit(records, items, .05)
        self.assertEqual(result["different_label_near_inputs"]["source"], 1)
        self.assertEqual(result["pairs"][0]["modulator_rms_distance"], 0)

    def test_analyze_run_saves_offline_reports_and_state_comparisons(self):
        metadata = {**METADATA, "items_sha256": sha256(self.items_path)}
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            runner.run_experiment(self.emotion_items[:2], synthetic_query, root / "run", metadata,
                                  conditions=["full_metamo"], selector="emotion_only", feeling_scorer="quadratic")
            with patch("benchmarks.steu_experiment.GeminiJSON", side_effect=AssertionError("API used")):
                report = analyze_run(root / "run", self.items_path, root / "diagnostics", grid=[0., 1.])
            self.assertEqual(len(report["comparisons"]["state_replay"]), len(SCORERS) * len(SELECTORS) * 2)
            self.assertEqual(len(report["comparisons"]["fixed_states"]), len(SCORERS) * len(SELECTORS))
            for name in ("diagnostics.json", "pairwise.csv", "comparison_metrics.csv", "error_evidence.csv"):
                self.assertTrue((root / "diagnostics" / name).is_file())

    def test_plot_comparison_distinguishes_variants_and_rejects_duplicates(self):
        variants = [("normalized", "independent", "full", 1, "emotion_only"),
                    ("quadratic", "independent", "full", 1, "emotion_only"),
                    ("normalized", "persistent", "full", 1, "emotion_only"),
                    ("normalized", "independent", "none", 1, "emotion_only"),
                    ("normalized", "independent", "full", 1, "magus_with_emotion"),
                    ("normalized", "independent", "full", 2, "magus_with_emotion")]
        metadata = {**METADATA, "items_sha256": sha256(self.items_path)}
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            runs = [root / str(i) for i in range(len(variants))]
            for run, (scorer, state, history, weight, selector) in zip(runs, variants):
                runner.run_experiment(self.emotion_items[:2], synthetic_query, run, metadata,
                    conditions=["full_metamo"], feeling_scorer=scorer, state_policy=state,
                    history_policy=history, feeling_weight=weight, selector=selector)
            report = analyze_runs(runs, self.items_path, root / "plots")
            self.assertEqual(len(report["available_cases"]), len(variants))
            self.assertEqual(len({row["label"] for row in report["conditions"].values()}), len(variants))
            self.assertTrue({"feeling_scorer", "state_policy", "history_policy", "feeling_weight"}
                            <= set(report["varying_settings"]))
            self.assertTrue((root / "plots/question_results.csv").is_file())
            self.assertEqual(list(load_comparison(runs[:1], self.items_path)[0]), ["emotion_only"])
            with self.assertRaisesRegex(ValueError, "Duplicate configuration"):
                load_comparison([runs[0], runs[0]], self.items_path)
            runner.run_experiment(self.emotion_items[:2], synthetic_query, root / "alias", metadata,
                conditions=["full_metamo"], selector="magus_with_emotion", feeling_scorer="normalized",
                state_policy="independent", history_policy="full", feeling_weight=1.0, case_id="alias")
            with self.assertRaisesRegex(ValueError, "Duplicate configuration"):
                load_comparison([runs[4], root / "alias"], self.items_path)


if __name__ == "__main__":
    unittest.main()
