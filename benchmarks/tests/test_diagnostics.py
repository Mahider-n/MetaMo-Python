"""Independent checks for experimental readouts and offline controls."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

from benchmarks import config
from benchmarks import steu_experiment as runner
from benchmarks.diagnostics import (pairwise_audit, competing_pairs, dominance_certificate,
                                    representation_audit, analyze_run)
from benchmarks.extended_feelings import all_feelings, normalized_feelings
from benchmarks.feeling_models import SCORERS, score_feelings, score_values, state_values
from benchmarks.helpers import initial_state, snapshot, option_feeling
from benchmarks.selection import select_answer, SELECTORS
from benchmarks.steu_data import Item, prepare, sha256
from benchmarks.tests.test_steu_experiment import synthetic_query
from core.config import M_RESOLUTION

ROOT=Path(__file__).resolve().parents[2]


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.items_path=Path(cls.temp.name)/'items.jsonl'
        cls.items=prepare(ROOT/'benchmarks/data/steu_author_source.txt',
                          ROOT/'benchmarks/qpt6z-osfstorage-archive/experiment_1_STEU.csv',cls.items_path)
        cls.emotion_items=[i for i in cls.items if all(option_feeling(t) for t in i.options.values())]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_legacy_and_normalized_numeric_compatibility(self):
        rng=np.random.default_rng(42)
        for _ in range(30):
            state=initial_state();state.M=rng.random(6)
            for name,expected in [('legacy',all_feelings(state)),('normalized',normalized_feelings(state))]:
                actual=score_feelings(state,name)
                vectorized=score_values(state_values(state),name)
                for key in actual:self.assertAlmostEqual(actual[key],float(vectorized[key]),places=13)
                self.assertEqual(actual.keys(),expected.keys())
                for key in actual:self.assertAlmostEqual(actual[key],expected[key],places=13)

    def test_resolution_and_excitement_are_separate_controls(self):
        state=initial_state();state.M[M_RESOLUTION]=.1
        low={s:score_feelings(state,s) for s in SCORERS}
        state.M[M_RESOLUTION]=.9
        high={s:score_feelings(state,s) for s in SCORERS}
        for s in ('legacy','legacy_fixed_excitement','normalized_no_resolution'):
            self.assertEqual(low[s],high[s])
        self.assertGreater(low['normalized']['anger'],high['normalized']['anger'])
        self.assertGreater(low['legacy_with_resolution']['anger'],high['legacy_with_resolution']['anger'])
        self.assertTrue(dominance_certificate('legacy','excitement','anger')['certified'])
        self.assertFalse(dominance_certificate('normalized','excitement','anger')['certified'])
        # Defined region centers reach one under the distance readout as well.
        values=np.array([[.9,.1,.5,.1,.05],[.9,.1,.5,.1,.95]])
        scores=score_values(values,'prototype_distance')
        self.assertGreater(scores['anger'][0],scores['excitement'][0])
        self.assertGreater(scores['excitement'][1],scores['anger'][1])
        self.assertTrue(all(((v>=0)&(v<=1)).all() for v in scores.values()))

    def test_every_competing_pair_is_reported_for_every_scorer(self):
        expected={pair for item in self.emotion_items
                  for pair in combinations(sorted({option_feeling(t) for t in item.options.values()}),2)}
        result=pairwise_audit(self.items,[0.,.5,1.])
        self.assertEqual(result['pair_count'],len(expected))
        self.assertEqual(set(result['excluded_nonemotion_item_ids']),{5,10,19,20,29,41})
        for scorer in SCORERS:
            rows=[r for r in result['pairs'] if r['scorer']==scorer]
            self.assertEqual({(r['emotion_a'],r['emotion_b']) for r in rows},expected)
            self.assertEqual(len(rows),len(expected))
            for row in rows:
                self.assertEqual(row['a_wins']+row['b_wins']+row['ties'],3**5)
                if row['a_witness_values'] is not None:
                    score=score_values(np.array(row['a_witness_values']),scorer)
                    self.assertGreater(score[row['emotion_a']],score[row['emotion_b']])
                if row['b_dominates_a']['certified']:
                    self.assertEqual(row['a_wins'],0)
                if row['a_dominates_b']['certified']:
                    self.assertEqual(row['b_wins'],0)

    def test_tie_break_uses_semantics_only_for_tied_options(self):
        emotion=dict(zip('ABCDE',[.1,.8,.2,.3,.8]))
        probabilities=dict(zip('ABCDE',[.1,.1,.1,.1,.6]))
        magus={k:100 if k=='C' else 0 for k in 'ABCDE'}
        pick,_,tie=select_answer('emotion_then_plausibility',magus,emotion,probabilities,magus)
        self.assertEqual(pick,'E');self.assertTrue(tie['resolved_by_plausibility'])
        self.assertEqual(select_answer('emotion_only',magus,emotion,probabilities,magus)[0],'B')
        emotion['B']=.81
        self.assertEqual(select_answer('emotion_then_plausibility',magus,emotion,probabilities,magus)[0],'B')

    def test_rank_fusion_does_not_depend_on_magus_or_raw_magnitude(self):
        e=dict(zip('ABCDE',[.1,.2,.3,.4,.5]));p=dict(zip('ABCDE',[.6,.1,.05,.15,.1]))
        magus={k:0 for k in 'ABCDE'}
        before=select_answer('rank_fusion',magus,e,p,magus)
        after=select_answer('rank_fusion',{k:1e6 for k in 'ABCDE'},
                            {k:17*v-20 for k,v in e.items()},p,magus)
        self.assertEqual(before[:2],after[:2])

    def test_history_projection_does_not_erase_audit_history_or_change_state_policy(self):
        for policy in config.STATE_POLICIES:
            for history in config.HISTORY_POLICIES:
                with self.subTest(policy=policy,history=history):
                    session=runner.ConditionSession('full_metamo',state_policy=policy,history_policy=history)
                    first=session.step(self.emotion_items[0],synthetic_query)
                    second=session.step(self.emotion_items[1],synthetic_query)
                    payload=json.loads(second['prompt'].split('\n\nDATA:\n')[1])
                    self.assertEqual(len(session.history),2)
                    self.assertEqual(second['history_items_before'],0 if history=='none' else 1)
                    self.assertEqual(second['state_before'],snapshot(initial_state()) if policy=='independent' else first['state_after'])
                    if history=='answers_only':self.assertEqual(set(payload['previous_interactions'][0]),{'item','answer'})
                    if history=='full':self.assertIn('appraised_state',payload['previous_interactions'][0])

    def test_resume_reproduces_each_history_projection(self):
        metadata={'model':'offline','temperature':0,'seed':42,'items_sha256':sha256(self.items_path),'max_output_tokens':16384}
        for policy in config.STATE_POLICIES:
            for history in config.HISTORY_POLICIES:
                with self.subTest(policy=policy,history=history), tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()),redirect_stderr(StringIO()):
                    root=Path(tmp);count=0
                    def interrupted(prompt):
                        nonlocal count
                        count+=1
                        if count==2:raise RuntimeError('interrupted')
                        return synthetic_query(prompt)
                    kwargs=dict(conditions=['full_metamo'],state_policy=policy,history_policy=history,
                                feeling_scorer='prototype_distance',selector='emotion_then_plausibility')
                    with self.assertRaises(RuntimeError):runner.run_experiment(self.emotion_items[:3],interrupted,root/'resume',metadata,**kwargs)
                    runner.run_experiment(self.emotion_items[:3],synthetic_query,root/'resume',metadata,resume=True,**kwargs)
                    runner.run_experiment(self.emotion_items[:3],synthetic_query,root/'fresh',metadata,**kwargs)
                    def load(name):return [{k:v for k,v in json.loads(l).items() if k!='elapsed_seconds'} for l in (root/name/'responses.jsonl').read_text().splitlines()]
                    self.assertEqual(load('fresh'),load('resume'))


    def test_identical_features_with_different_target_emotions_are_reported(self):
        options=dict(zip('ABCDE',['Angry','Guilty','Happy','Scared','Surprised']))
        items={1:Item(1,'Unfair treatment',options,['A'],['A']),2:Item(2,'Harmed a friend',options,['B'],['B'])}
        records=[{'item_id':i,'stimulus':dict(novelty=.5,conduciveness=-.5,risk=.5,effort=.5),
                  'decision_state':snapshot(initial_state())} for i in items]
        result=representation_audit(records,items,.05)
        self.assertEqual(result['different_label_near_inputs']['source'],1)
        self.assertEqual(result['pairs'][0]['input_rms_distance'],0)
        self.assertEqual(result['guilt_vs_anger']['source'],'evaluable')


    def test_real_diagnostic_report_and_frozen_state_comparisons(self):
        metadata={'model':'offline','temperature':0,'seed':42,'items_sha256':sha256(self.items_path),'max_output_tokens':16384}
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
            root=Path(tmp)
            runner.run_experiment(self.emotion_items[:2],synthetic_query,root/'run',metadata,conditions=['full_metamo'])
            with patch('benchmarks.steu_experiment.GeminiJSON',side_effect=AssertionError('API used')):
                result=analyze_run(root/'run',self.items_path,root/'diagnostics',grid=[0.,1.])
            rows=result['comparisons']['state_replay']
            self.assertEqual(len(rows),len(SCORERS)*len(SELECTORS)*2)
            self.assertTrue((root/'diagnostics/pairwise.csv').is_file())
            self.assertNotIn('modulator_traces',result)
            self.assertNotIn('behavior_checks',result)
            self.assertFalse((root/'diagnostics/modulator_changes.csv').exists())
            self.assertTrue((root/'diagnostics/error_evidence.csv').is_file())
            self.assertTrue((root/'diagnostics/score_scales.csv').is_file())
            # State reset vs persistence sees the same generated data, but different starting states.
            pair=[r for r in rows if r['scorer']=='normalized' and r['selector']=='magus_only']
            reset=next(r for r in pair if r['state_policy']=='independent')
            persistent=next(r for r in pair if r['state_policy']=='persistent')
            self.assertEqual(reset['items'][1]['state_before'],snapshot(initial_state()))
            self.assertEqual(persistent['items'][1]['state_before'],persistent['items'][0]['state_after'])
            self.assertNotEqual(reset['items'][1]['state_before'],persistent['items'][1]['state_before'])
            # Tampered saved numeric scores must not be presented as a controlled comparison.
            path=root/'run/responses.jsonl';records=[json.loads(l) for l in path.read_text().splitlines()]
            records[0]['candidate_scores']['A']+=1
            path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
            with self.assertRaisesRegex(ValueError,'MAGUS'):
                analyze_run(root/'run',self.items_path,root/'bad',grid=[0.,1.])
            self.assertFalse((root/'bad').exists())

    def test_resume_rejects_changed_history_or_tie_settings_before_query(self):
        metadata={'model':'offline','temperature':0,'seed':42,'items_sha256':sha256(self.items_path),'max_output_tokens':16384}
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            root=Path(tmp);calls=0
            def query(prompt):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('stop')
                return synthetic_query(prompt)
            with self.assertRaises(RuntimeError):
                runner.run_experiment(self.emotion_items[:2],query,root/'run',metadata,conditions=['full_metamo'])
            for change in [{'history_policy':'none'},{'tie_breaker':'plausibility'},{'tie_tolerance':.001}]:
                client=Mock()
                with self.assertRaisesRegex(ValueError,'Cannot resume'):
                    runner.run_experiment(self.emotion_items[:2],client,root/'run',metadata,conditions=['full_metamo'],resume=True,**change)
                client.assert_not_called()


    def test_compare_selectors_uses_one_configured_scorer_and_state_policy(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            root=Path(tmp)
            with patch.multiple(config,COMPARE_SELECTORS=True,CONDITIONS=('full_metamo',),
                                FEELING_SCORER='prototype_distance',STATE_POLICY='persistent',HISTORY_POLICY='full'), \
                 patch.multiple(runner,DEFAULT_ITEMS=self.items_path,RESULTS_DIRECTORY=root,RESUME_DIRECTORY=None,
                                MODEL='offline',ITEM_LIMIT=1), \
                 patch('benchmarks.steu_experiment.GeminiJSON',return_value=Mock(side_effect=synthetic_query)) as client:
                self.assertEqual(runner.main([]),0)
            self.assertEqual(client.return_value.call_count,len(SELECTORS))
            output,=root.iterdir()
            experiment=json.loads((output/'experiment.json').read_text())
            self.assertEqual(len(experiment['runs']),len(SELECTORS))
            self.assertEqual({r['selector'] for r in experiment['runs']},set(SELECTORS))
            for entry in experiment['runs']:
                manifest=json.loads((output/entry['directory']/'manifest.json').read_text())
                self.assertEqual(manifest['feeling_scorer'],'prototype_distance')
                self.assertEqual(manifest['state_policy'],'persistent')
                self.assertEqual(manifest['history_policy'],'full')

    def test_single_run_does_not_execute_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            root=Path(tmp)
            with patch.multiple(config,COMPARE_SELECTORS=False,CONDITIONS=('full_metamo',),
                                SELECTOR='emotion_then_plausibility',FEELING_SCORER='prototype_distance'), \
                 patch.multiple(runner,DEFAULT_ITEMS=self.items_path,RESULTS_DIRECTORY=root,RESUME_DIRECTORY=None,
                                MODEL='offline',ITEM_LIMIT=2), \
                 patch('benchmarks.steu_experiment.GeminiJSON',return_value=Mock(side_effect=synthetic_query)) as client, \
                 patch('benchmarks.diagnostics.analyze_run',side_effect=AssertionError('Offline analysis ran')) as diagnostics:
                self.assertEqual(runner.main([]),0)
            diagnostics.assert_not_called()
            self.assertEqual(client.return_value.call_count,2)
            output,=root.iterdir()
            experiment=json.loads((output/'experiment.json').read_text())
            self.assertEqual(len(experiment['runs']),1)
            self.assertNotIn('diagnostics_directory',experiment)
            self.assertFalse((output/'diagnostics').exists())
            records=[json.loads(l) for l in (output/'primary/responses.jsonl').read_text().splitlines()]
            self.assertTrue(all('appraisal_trace' not in r for r in records))

    def test_invalid_single_case_settings_are_rejected_before_query(self):
        for changes in [{'HISTORY_POLICY':'invalid'},{'FEELING_SCORER':'invalid'},
                        {'COMPARE_SELECTORS':1}]:
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as tmp,redirect_stderr(StringIO()):
                with patch.multiple(config,**changes), \
                     patch.multiple(runner,DEFAULT_ITEMS=self.items_path,RESULTS_DIRECTORY=Path(tmp),
                                    RESUME_DIRECTORY=None,MODEL='offline',ITEM_LIMIT=1), \
                     patch('benchmarks.steu_experiment.GeminiJSON') as client:
                    self.assertEqual(runner.main([]),1)
                client.assert_not_called()

if __name__=='__main__':unittest.main()
