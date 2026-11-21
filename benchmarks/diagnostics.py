"""Offline STEU discrimination, representation, and fixed-input state comparisons. """
import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path

import numpy as np

from benchmarks.scorers.quadratic import EMOTIONS, formula_coefficients
from benchmarks.scorers.feelings import (
    SCORERS,
    model_spec,
    score_feelings,
    score_values,
    state_values,
)
from benchmarks.helpers import option_feeling, parse_plan, parse_stimulus
from benchmarks.scorers.llm import validate_model_data
from benchmarks.scorers.selection import SELECTORS, select_answer
from benchmarks.steu_data import LETTERS, load_items, sha256
from core.state import MotivationalState
from dynamics.stability import raise_boundary_caution
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from openpsi.feelings import LEVELS, U

DEFAULT_GRID = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)
DEFAULT_REPRESENTATION_TOLERANCE = 0.05  # the maximum distance at which two representations count as similar.


def validate_settings(grid, tolerance):
    """Validate the sampled state grid and representation distance tolerance."""
    if (not isinstance(grid, (tuple, list)) or not 2 <= len(grid) <= 11 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in grid)
            or len(set(grid)) != len(grid)):
        raise ValueError("Diagnostic grid must have 2–11 distinct finite values in [0,1]")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Representation tolerance must be in [0,1]")


def _region_contains(a, b):
    """Check whether level a's matching region contains level b's region."""
    ka, ta = LEVELS[a]
    kb, tb = LEVELS[b]
    return ka == kb and (ta >= tb if ka == "lt" else ta <= tb if ka == "gt" else ta == tb)

def dominance_certificate(scorer, stronger, weaker):
    """Prove global weak score dominance when prototype regions and weights permit.

    Failure to find a certificate does not rule out dominance.
    """
    if scorer == "quadratic":
        weights = formula_coefficients()
        certified = np.array_equal(weights[EMOTIONS.index(stronger)], weights[EMOTIONS.index(weaker)])
        return {"certified": certified, "method": "identical quadratic coefficients" if certified else
                "distinct coefficients admit opposite rankings on the full modulator cube"}
    templates, resolution, normalize = model_spec(scorer)
    def terms(proto):
        """Extract the scored dimensions, target levels, and weights."""
        levels, divisor = proto
        defined = [(i, level) for i, level in enumerate(levels) if level != U and (resolution or i != 1)]
        weight = 1 / (len(defined) if normalize else divisor)
        return {i: (level, weight) for i, level in defined}
    def covers(a, b):
        """Check whether prototype a guarantees a score at least as high as b."""
        if scorer == "prototype_distance" and a.keys() != b.keys():
            return False
        for i, (level, weight) in b.items():
            if i not in a or not _region_contains(a[i][0], level):
                return False
            if scorer == "prototype_distance":
                if not math.isclose(a[i][1], weight):
                    return False
            elif a[i][1] < weight - 1e-12:
                return False
        return True
    certified = all(any(covers(terms(a), terms(b)) for a in templates[stronger]) for b in templates[weaker])
    return {"certified": certified, "method": "prototype term/region containment" if certified else "no certificate found"}


def competing_pairs(items):
    """Find emotion pairs offered together and exclude unsupported items."""
    pairs = {}
    excluded = []
    for item in items:
        names = [option_feeling(label) for label in item.options.values()]
        if any(name is None for name in names):
            excluded.append(item.id)
            continue
        for pair in combinations(sorted(set(names)), 2):
            pairs.setdefault(pair, []).append(item.id)
    return pairs, excluded


def pairwise_audit(items, grid):
    """Measure pairwise wins, ties, score distributions, and dominance evidence."""
    pairs, excluded = competing_pairs(items)
    grids = {}
    rows, stats, sample_counts, value_orders = [], {}, {}, {}
    names = sorted({name for pair in pairs for name in pair})
    for scorer in SCORERS:
        dimensions = 6 if scorer == "quadratic" else 5
        if dimensions not in grids:
            grids[dimensions] = np.array(list(product(grid, repeat=dimensions)))
        values = grids[dimensions]
        sample_counts[scorer] = len(values)
        value_orders[scorer] = (["valence", "arousal", "approach", "resolution", "threshold", "securing"]
                                if scorer == "quadratic" else
                                ["activation", "resolution", "securing (inverted from MetaMo)", "selection", "pleasure"])
        scores = score_values(values, scorer)
        stats[scorer] = {name: {"min": float(scores[name].min()), "max": float(scores[name].max()),
                               "mean": float(scores[name].mean()), "std": float(scores[name].std()),
                               "quantiles_10_50_90": np.quantile(scores[name], [.1, .5, .9]).tolist()}
                         for name in names}
        for (a, b), ids in pairs.items():
            diff = scores[a] - scores[b]
            aw = int((diff > 1e-9).sum()); bw = int((diff < -1e-9).sum())
            ai, bi = int(diff.argmax()), int(diff.argmin())
            rows.append({"scorer": scorer, "emotion_a": a, "emotion_b": b, "item_ids": ids,
                         "sample_count": len(values), "a_wins": aw, "b_wins": bw,
                         "ties": int((abs(diff) <= 1e-9).sum()),
                         "sampled_never_wins": [name for name, wins in [(a, aw), (b, bw)] if not wins],
                         "max_a_minus_b": float(diff[ai]), "max_b_minus_a": float(-diff[bi]),
                         "a_witness_values": values[ai].tolist() if aw else None,
                         "b_witness_values": values[bi].tolist() if bw else None,
                         "a_dominates_b": dominance_certificate(scorer, a, b),
                         "b_dominates_a": dominance_certificate(scorer, b, a)})
    return {"pair_count": len(pairs), "scorers": list(SCORERS), "grid": list(grid),
            "states_per_scorer": sample_counts, "excluded_nonemotion_item_ids": excluded,
            "grid_note": "Finite-grid ties/never-wins are sampled evidence; only certificates assert global weak dominance.",
            "value_order": value_orders,
            "score_distributions": stats, "pairs": rows}


def _state(saved):
    """Restore a motivational state from saved goals and modulators."""
    return MotivationalState(G=np.array(saved["G"]), M=np.array(saved["M"]))


def representation_audit(records, items, tolerance):
    """Find similar appraisal inputs (stimulus) and states with different accepted emotions."""
    rows = []
    inputs, states = {}, {}
    for record in records:
        stimulus = record['stimulus']
        inputs[record['item_id']] = np.array([stimulus['novelty'], (stimulus['conduciveness'] + 1) / 2,
                                             stimulus['risk'], stimulus['effort']])
        states[record['item_id']] = np.array(record['decision_state']['M'])
    for a, b in combinations(records, 2):
        stimulus_distance = float(np.sqrt(np.mean((inputs[a['item_id']] - inputs[b['item_id']]) ** 2)))
        state_distance = float(np.sqrt(np.mean((states[a['item_id']] - states[b['item_id']]) ** 2)))
        left, right = items[a['item_id']], items[b['item_id']]
        distinct = {}
        for key in ('source', 'reference'):
            ln = {option_feeling(left.options[k]) for k in getattr(left, key + '_answers')}
            rn = {option_feeling(right.options[k]) for k in getattr(right, key + '_answers')}
            distinct[key] = ln.isdisjoint(rn)
        rows.append({'item_a':left.id, 'item_b':right.id, 'input_rms_distance':stimulus_distance,
                     'modulator_rms_distance':state_distance, 'near_inputs':stimulus_distance <= tolerance,
                     'near_modulators':state_distance <= tolerance, 'different_accepted_emotions':distinct})
    rows.sort(key=lambda r:r['input_rms_distance'])
    targets = {key: sorted({option_feeling(items[r['item_id']].options[k]) for r in records
                           for k in getattr(items[r['item_id']],key+'_answers')}) for key in ('source','reference')}
    return {'tolerance':tolerance, 'distance':'RMS; conducive input scaled to [0,1]',
            'pairs':rows, 'different_label_near_inputs':{key:sum(r['near_inputs'] and r['different_accepted_emotions'][key] for r in rows) for key in targets},
            'available_target_emotions':targets,
            'guilt_vs_anger':{key:'evaluable' if {'guilt','anger'} <= set(names) else 'not evaluable: both target emotions are not present' for key,names in targets.items()},
            'interpretation':'Similar representations cannot reveal omitted agency/blame. A plausibility tie-break uses a separate semantic signal, not recovered appraisal information.'}


def _model_data(record):
    """Load saved model output and verify its stimulus and plausibility scores."""
    data = record.get("model_data")
    if data is None:
        raw = (record.get("response") or {}).get("raw_text")
        if raw is None:
            raise ValueError("Saved model_data or response.raw_text is required for replay")
        data = json.loads(raw)
    try:
        validate_model_data(data, "full_metamo")
    except (KeyError, TypeError) as error:
        raise ValueError("Saved candidate data must include valid plausibilities for all replay methods") from error
    logged = (record.get("feeling_choice") or {}).get("plausibility_scores")
    if logged and any(not math.isclose(c["plausibility"], logged[c["id"]], abs_tol=1e-12)
                      for c in data["candidates"]):
        raise ValueError("Saved raw and logged plausibilities differ")
    if data["stimulus"] != record["stimulus"]:
        raise ValueError("Saved stimulus and raw model data differ")
    return data


def fixed_input_comparisons(records, items, tie_breaker, tie_tolerance, feeling_weight):
    """Compare readouts, selectors, and state policies using saved model inputs.

    Replay does not measure how changed history would affect model output.
    """
    from benchmarks.steu_experiment import ConditionSession
    replay_rows = []
    base = records[0]['state_before']
    saved_inputs = [(items[record['item_id']], _model_data(record)) for record in records]
    for scorer, selector in product(SCORERS, SELECTORS):
        for policy in ('independent','persistent'):
            session = ConditionSession('full_metamo', selector=selector, feeling_scorer=scorer,
                                       state_policy=policy, history_policy='none',
                                       tie_breaker=tie_breaker, tie_tolerance=tie_tolerance, feeling_weight=feeling_weight)
            session.state = _state(base)

            outcomes=[]
            for item, data in saved_inputs:
                r=session.step(item,lambda prompt, data=data:(data,{'source':'frozen saved response'}))
                outcomes.append({'item_id':item.id,'answer':r['answer'],'state_before':r['state_before'],
                                 'state_after':r['state_after'],'source_correct':r['source_correct'],
                                 'reference_correct':r['reference_compatible_correct']})
            replay_rows.append({'kind':'frozen_input_state_replay','state_policy':policy,'scorer':scorer,'selector':selector,
                                'source_correct':sum(r['source_correct'] for r in outcomes),
                                'reference_correct':sum(r['reference_correct'] for r in outcomes),'total':len(outcomes),
                                'items':outcomes})
    fixed=[]
    for scorer, selector in product(SCORERS, SELECTORS):
        outcomes=[]
        for saved in records:
            item=items[saved['item_id']]; state=_state(saved['decision_state'])
            felt=score_feelings(state,scorer)
            emotion={k:felt[option_feeling(v)] for k,v in item.options.items()}
            probs=saved['feeling_choice'].get('plausibility_scores')
            if not probs:
                raise ValueError('Saved plausibilities are required for these comparisons')
            magus=saved['candidate_scores']; mean=sum(emotion.values())/5
            combined={k:magus[k]+feeling_weight*(emotion[k]-mean) for k in LETTERS}
            answer,_,tie=select_answer(selector,magus,emotion,probs,combined,tie_breaker=tie_breaker,tolerance=tie_tolerance)
            outcomes.append({'item_id':item.id,'answer':answer,'tie':tie,
                             'source_correct':int(answer in item.source_answers),'reference_correct':int(answer in item.reference_answers)})
        fixed.append({'kind':'fixed_recorded_states','scorer':scorer,'selector':selector,
                      'source_correct':sum(r['source_correct'] for r in outcomes),
                      'reference_correct':sum(r['reference_correct'] for r in outcomes),'total':len(outcomes),'items':outcomes})
    return {'scope':'Frozen extraction and candidate data, no API requests. History is not varied in replay. Selector-driven state updates are recomputed for trajectory comparisons.',
            'state_replay':replay_rows,'fixed_states':fixed}


def analyze_run(run, items_path, output, *, grid=None, tolerance=None):
    """Validate a completed run and save offline diagnostic reports."""
    if output.exists():
        raise FileExistsError(f'Diagnostic output already exists: {output}')
    grid = DEFAULT_GRID if grid is None else grid
    tolerance = DEFAULT_REPRESENTATION_TOLERANCE if tolerance is None else tolerance
    validate_settings(grid,tolerance)
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('items_sha256')!=sha256(items_path):
        raise ValueError('Diagnostics require a completed run and its matching prepared item file')
    items={i.id:i for i in load_items(items_path)}
    records=[json.loads(l) for l in (run/'responses.jsonl').read_text().splitlines() if l.strip()]
    records=[r for r in records if r['condition']=='full_metamo']
    if not records:
        raise ValueError('This run has no MetaMo records to diagnose')
    if [r['item_id'] for r in records]!=manifest['item_order']:
        raise ValueError('Saved MetaMo records must match the complete item order')
    selected=[items[r['item_id']] for r in records]
    if any(option_feeling(v) is None for item in selected for v in item.options.values()):
        raise ValueError('Diagnostics currently require the emotion-choice subset')
    engine=OpenPsiAppraisal(); decision_engine=MagusDecision()
    for r in records:
        item=items[r['item_id']]
        for field,key in [('source_correct','source_answers'),('reference_compatible_correct','reference_answers')]:
            if r[field]!=int(r['answer'] in getattr(item,key)):
                raise ValueError('Saved correctness flags do not match the answer keys')
        if json.loads(r['prompt'].split('\n\nDATA:\n',1)[1])['current_item']!=item.public():
            raise ValueError('Saved question does not match the prepared item')
        before=_state(r['state_before']); stimulus=parse_stimulus(r['stimulus']); appraised=engine.appraise(before,stimulus)
        decision=raise_boundary_caution(appraised)
        if (not np.allclose(decision.M,r['decision_state']['M'],rtol=0,atol=1e-10)
                or not np.allclose(decision.G,r['decision_state']['G'],rtol=0,atol=1e-10)):
            raise ValueError('Current appraisal does not reproduce saved decision states; use original core implementation')
        candidates=parse_plan(_model_data(r))
        scores={c.id:decision_engine.score_candidate(decision,c) for c in candidates}
        if any(not math.isclose(scores[k],r['candidate_scores'][k],rel_tol=0,abs_tol=1e-10) for k in LETTERS):
            raise ValueError('Current MAGUS does not reproduce saved scores; use original core implementation')
    pairs=pairwise_audit(selected,grid)
    pairs['recorded_state_distributions']={}
    for scorer in SCORERS:
        observed_values=np.array([state_values(_state(r['decision_state']),scorer) for r in records])
        scores=score_values(observed_values,scorer)
        pairs['recorded_state_distributions'][scorer]={name:{'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),
            'std':float(v.std()),'quantiles_10_50_90':np.quantile(v,[.1,.5,.9]).tolist()} for name,v in scores.items()}
        for row in (r for r in pairs['pairs'] if r['scorer']==scorer):
            indices=[i for i,r in enumerate(records) if r['item_id'] in row['item_ids']]
            diff=scores[row['emotion_a']][indices]-scores[row['emotion_b']][indices]
            row['recorded_comparisons']={'total':len(indices),'a_wins':int((diff>1e-9).sum()),
                                         'b_wins':int((diff< -1e-9).sum()),'ties':int((abs(diff)<=1e-9).sum()),
                                         'mean_a_minus_b':float(diff.mean())}

    weight = manifest.get('feeling_weight',1.0) if manifest['selector']=='magus_with_emotion' else manifest.get('config',{}).get('benchmarks.config',{}).get('FEELING_WEIGHT',1.0)
    comparisons=fixed_input_comparisons(records,items,manifest.get('tie_breaker','letter'),manifest.get('tie_tolerance',1e-9),weight)
    representations=representation_audit(records,items,tolerance)
    diagnostics={'kind':'offline_steu_diagnostics','created_utc':datetime.now(timezone.utc).isoformat(),
                 'source_run':str(run.resolve()),'source_protocol':manifest['protocol'],
                 'manifest_sha256':sha256(run/'manifest.json'),'responses_sha256':sha256(run/'responses.jsonl'),
                 'items_sha256':sha256(items_path),'item_order':manifest['item_order'],
                 'diagnostics_code_sha256':sha256(Path(__file__)),
                 'appraisal_code_sha256':sha256(Path(__file__).resolve().parents[1]/'openpsi/appraisal.py'),
                 'magus_code_sha256':sha256(Path(__file__).resolve().parents[1]/'magus/decision.py'),
                 'comparison_feeling_weight':weight,
                 'scoring_keys':{'source':'literal author scoring rules','reference':'provisional CSV-compatible key'},
                 'pairwise':pairs,'representations':representations,'comparisons':comparisons}

    distributions={}
    for scorer in SCORERS:
        margins=[]; predictions=Counter()
        for r,item in zip(records,selected):
            felt=score_feelings(_state(r['decision_state']),scorer)
            scores={k:felt[option_feeling(v)] for k,v in item.options.items()}
            vals=sorted(scores.values(),reverse=True); pick=max(LETTERS,key=lambda k:scores[k])
            predictions[option_feeling(item.options[pick])]+=1
            margins.append({'item_id':item.id,'margin':vals[0]-vals[1],
                            'all_option_spread':vals[0]-vals[-1],
                            'source_correct':int(pick in item.source_answers),'reference_correct':int(pick in item.reference_answers)})
        targets={key:dict(Counter(option_feeling(item.options[k]) for item in selected for k in getattr(item,key+'_answers'))) for key in ('source','reference')}
        groups={}; associations={}
        for key in ('source','reference'):
            groups[key]={}
            for name,predicate in [('tied',lambda m:m<=1e-9),('margin_le_0.05',lambda m:m<=.05),('margin_gt_0.05',lambda m:m>.05)]:
                chosen=[r for r in margins if predicate(r['margin'])]
                groups[key][name]={'total':len(chosen),'correct':sum(r[key+'_correct'] for r in chosen)}
            values=np.array([r['margin'] for r in margins]); correct=np.array([r[key+'_correct'] for r in margins])
            associations[key]={'mean_margin_correct':float(values[correct==1].mean()) if (correct==1).any() else None,
                               'mean_margin_incorrect':float(values[correct==0].mean()) if (correct==0).any() else None,
                               'margin_correctness_correlation':float(np.corrcoef(values,correct)[0,1]) if values.std()>1e-12 and correct.std()>1e-12 else None}
        distributions[scorer]={'predicted_labels':dict(predictions),'accepted_label_counts':targets,
                               'note':'Source counts may exceed item count where multiple labels are accepted; selections here use stable letter ties.',
                               'items':margins,'margin_groups':groups,'margin_association':associations,
                               'nearly_uniform_options':{'spread_le_1e-9':sum(r['all_option_spread']<=1e-9 for r in margins),
                                                         'spread_le_0.05':sum(r['all_option_spread']<=.05 for r in margins)}}
    diagnostics['score_diagnostics']=distributions
    encoded=json.dumps(diagnostics,indent=2,allow_nan=False)+'\n'
    output.mkdir(parents=True,exist_ok=False)
    (output/'diagnostics.json').write_text(encoded)
    def write_csv(name,rows):
        """Save report rows as CSV, encoding nested values as JSON."""
        if not rows:return
        with (output/name).open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader()
            w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    write_csv('pairwise.csv',pairs['pairs'])
    write_csv('representation_pairs.csv',representations['pairs'])
    write_csv('comparison_metrics.csv',[{k:v for k,v in row.items() if k!='items'} for kind in ['state_replay','fixed_states'] for row in comparisons[kind]])
    error_rows=[]; scale_rows=[]
    for r,item in zip(records,selected):
        scorer=r['feeling_choice'].get('scorer',manifest.get('feeling_scorer','legacy'))
        felt=r['feeling_choice'].get('feelings')
        if felt is None:
            felt=score_feelings(_state(r['decision_state']),scorer)
        es={k:felt[option_feeling(v)] for k,v in item.options.items()}
        probs=r['feeling_choice']['plausibility_scores']; ms=r['candidate_scores']
        mp=max(LETTERS,key=lambda k:ms[k]); ep=max(LETTERS,key=lambda k:es[k]); pp=max(LETTERS,key=lambda k:probs[k])
        evidence={'item_id':item.id,'selector':r['selector'],'scorer':scorer,'question':item.question,
                  'selected':r['answer'],'magus_pick':mp,'emotion_pick_letter_ties':ep,'plausibility_pick':pp,
                  'emotion_top_ties':[k for k in LETTERS if max(es.values())-es[k]<=1e-9],
                  'near_different_label_items':sorted({pair['item_b'] if pair['item_a']==item.id else pair['item_a']
                      for pair in representations['pairs'] if item.id in (pair['item_a'],pair['item_b'])
                      and pair['near_inputs'] and any(pair['different_accepted_emotions'].values())}),
                  'appraisal_quality':'not independently adjudicated','causal_attribution':'requires review/controlled comparison'}
        for key in ('source','reference'):
            accepted=getattr(item,key+'_answers')
            evidence[key+'_accepted']=accepted
            evidence[key+'_semantic_correct_numerical_wrong']=pp in accepted and mp not in accepted
            evidence[key+'_correct_magus_overturned']=mp in accepted and r['answer'] not in accepted
            evidence[key+'_selected_correct']=r['answer'] in accepted
        error_rows.append(evidence)
        magus_span=max(ms.values())-min(ms.values());emotion_span=max(es.values())-min(es.values())
        scale_rows.append({'item_id':item.id,'scorer':scorer,'magus_span':magus_span,
                           'emotion_span':emotion_span,'weighted_emotion_span':weight*emotion_span,
                           'weighted_span_ratio':weight*emotion_span/magus_span if magus_span>1e-12 else None})
    write_csv('error_evidence.csv',error_rows)
    write_csv('score_scales.csv',scale_rows)

    lines=['# Offline STEU diagnostics','',f"Source: {run.resolve()} (archived protocol {manifest['protocol']}).",'',f"Items: {len(records)}; competing pairs: {pairs['pair_count']}; readouts: {len(SCORERS)}.",
           '', 'All comparisons below reuse saved inputs. They are not fresh model experiments and do not calibrate any parameters.',
           'Quadratic scores use coefficients previously fitted to STEU answers; performance on those questions is development evidence.',
           'Error evidence uses archived emotion scores when available; comparison tables use the current scorers.',
           '', '## Fixed recorded states', '', '| Readout | Selector | Source | CSV-compatible |','| --- | --- | ---: | ---: |']
    lines += [f"| {r['scorer']} | {r['selector']} | {r['source_correct']}/{r['total']} | {r['reference_correct']}/{r['total']} |" for r in comparisons['fixed_states']]
    lines += ['', '## Reset versus persistent state: frozen-input replay', '',
              '| Readout | Selector | State policy | Source | CSV-compatible |',
              '| --- | --- | --- | ---: | ---: |']
    lines += [f"| {r['scorer']} | {r['selector']} | {r['state_policy']} | {r['source_correct']}/{r['total']} | {r['reference_correct']}/{r['total']} |" for r in comparisons['state_replay']]
    lines += ['', '## Interpretation', '',
              'Pair coverage is exhaustive for co-occurring labels in these items. State-grid coverage is finite; global dominance is asserted only with a structural certificate.',
              'State replay holds generated stimuli/candidates fixed. For fresh state comparisons, run the experiment separately with each STATE_POLICY, keeping the other settings fixed.',
              'A semantic tie-break is an additional information source; it cannot repair an appraisal representation that omits agency or responsibility.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')

    return diagnostics


def main(argv=None):
    """Run offline diagnostics from command-line arguments."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path,help='One completed run folder containing MetaMo records, e.g. primary')
    p.add_argument('--items',type=Path,default=Path('benchmarks/data/steu_items.jsonl'))
    p.add_argument('--output',required=True,type=Path,help='New output directory')
    args=p.parse_args(argv)
    analyze_run(args.run,args.items,args.output)
    print(f'Diagnostics saved: {args.output}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
