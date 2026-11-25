"""Measured training contexts and explicit-denominator integration evaluation."""
from dataclasses import asdict, replace
import time
import numpy as np
from category.bimonad import MetaMoPseudoBimonad
from core.state import MotivationalState, Stimulus
from dynamics.coherence import check_self_model_drift
from dynamics.stability import is_in_safe_region, apply_homeostatic_damping, check_contractive_update_law
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from .contracts import digest
from .coordinator import Coordinator
from .contextual import semantics
from .evidence import collect, matched_improvements, rollout
from .motives import project


def training_rows(env, registry, contract, seeds, contexts_per_seed=16):
    rng = np.random.default_rng(17)
    baseline_candidate = next(c for c in registry.candidates if c.loader == "noop")
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
    rows, raw_evidence = [], {}
    for seed in seeds:
        protocol = replace(contract, seeds=(seed,))
        baseline = collect(env, registry, baseline_candidate, protocol)
        records = [collect(env, registry, candidate, protocol) for candidate in registry.candidates]
        measurements = []
        for candidate, record in zip(registry.candidates, records):
            dr, dn = matched_improvements(baseline, record, protocol)
            measurements.append({"skill_id": candidate.skill_id, "delta_r": dr, "delta_n": dn.tolist(), "action": candidate.metadata()["action"]})
        raw_evidence[str(seed)] = {"baseline": baseline, "candidates": records}
        evidence_id = digest(raw_evidence[str(seed)])
        eps = [r["episodes"][0] for r in records]
        risk = sum(e["crashed"] for e in eps) / len(eps)
        success = sum(e["landing_success"] for e in eps) / len(eps)
        effort = float(np.clip(np.mean([-e["objectives"][1] / e["steps"] for e in eps]), 0, 1))
        dispersion = float(np.std([e["payoff"] for e in eps]))
        # A declared, reproducible target rule, not a fitted safety guarantee.
        slack = .10 + .10 * dispersion / (1 + dispersion)
        for _ in range(contexts_per_seed):
            stimulus = Stimulus(float(rng.uniform()), success, risk, effort)
            a = MotivationalState(rng.uniform(.1, .9, 8), rng.uniform(.1, .9, 6))
            b = MotivationalState(rng.uniform(.1, .9, 8), rng.uniform(.1, .9, 6))
            ca, cb = bimonad.decision_context(a, stimulus), bimonad.decision_context(b, stimulus)
            context = bimonad.parallel_merge(ca, cb)
            geo = project(context, stimulus)
            rows.append({"seed": seed, "context": list(geo.context), "weights": list(geo.weights),
                         "support_values": np.minimum(1, np.array(geo.weights) + slack).tolist(),
                         "semantics": semantics(contract), "evidence_digest": evidence_id, "candidates": measurements,
                         "context_a": [*ca.G.tolist(), *ca.M.tolist()], "context_b": [*cb.G.tolist(), *cb.M.tolist()]})
    return rows, raw_evidence


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def evaluate_runtime(repo, evaluation_env, operational_env, model, selector, seeds, trace_path, backup_approval=None):
    host = Coordinator(repo, epsilon=5., selector=selector, contextual_model=model,
                       backup_approval=backup_approval, trace_path=trace_path)
    actions = [c.action for c in repo.registry.candidates]
    baseline_candidate = next(c for c in repo.registry.candidates if c.loader == "noop")
    records, no_selection, backup, successes, crashes, negative = [], 0, 0, 0, 0, 0
    queries_before = model.queries if model else 0
    infeasible_before = model.invalid_support_events if model else 0
    switches, variation, srv, projected, coherence_failures, contraction_failures = 0, [], 0, 0, 0, 0
    recovery_latencies, adaptation_latencies = [], []
    comparable = 0
    full_diagnostics = []
    subsystem_variation = {"a": [], "b": []}
    lifts, previous_skill = [], None
    stimulus = Stimulus(.2, .5, .5, .2)
    for seed in seeds:
        obs, _ = operational_env.reset(seed=seed)
        before = host.state_a.copy()
        before_b = host.state_b.copy()
        previous_stimulus = stimulus
        result = host.select(obs, stimulus, actions)
        if result.selection:
            action = repo.registry.resolve(next(a for a in actions if a.id == result.selection.action_id)).action
            _, _, _, _, context_a, context_b = host._pending
            for decision_context in (context_a, context_b):
                proposed = decision_context.G + apply_homeostatic_damping(decision_context, action.delta_g)
                projected += int(proposed[0] < .3 or np.linalg.norm(np.clip(proposed, 0, 1)) > 2)
            probe = context_a.copy()
            probe.G = np.clip(probe.G + .001, 0, 1)
            probe.M = np.clip(probe.M + .001, 0, 1)
            first = host.bimonad.transition_selected(context_a, action)
            second = host.bimonad.transition_selected(probe, action)
            contraction_failures += int(first.distance_to(second) > .9 * context_a.distance_to(probe) + .05)
            outcome, stimulus = host.execute(result.selection, operational_env, obs, seed)
            for name, state in (("a", before), ("b", before_b)):
                probe = state.copy()
                probe.G = np.clip(probe.G + .001, 0, 1)
                full_diagnostics.append({"seed": seed, "subsystem": name,
                                         "lax_distributive_law": bool(host.bimonad.check_lax_distributive_law(state, previous_stimulus, result.eligible_actions)),
                                         "contractive_update_law": bool(check_contractive_update_law(host.bimonad, state, probe, previous_stimulus, result.eligible_actions))})
        else:
            no_selection += 1
            recovery_latencies.append(result.recovery.latency_seconds)
            if result.recovery.outcome != "backup":
                records.append({"seed": seed, "recovery": asdict(result.recovery)})
                continue
            backup += 1
            outcome, stimulus = host.execute_recovery(result.recovery, operational_env, obs, seed)
        skill = outcome["skill_id"]
        switches += int(previous_skill is not None and skill != previous_skill)
        previous_skill = skill
        successes += int(outcome["outcome"]["landing_success"])
        crashes += int(outcome["outcome"]["crashed"])
        for name, old, current in (("a", before, host.state_a), ("b", before_b, host.state_b)):
            variation.append(current.distance_to(old))
            subsystem_variation[name].append({"G": float(np.linalg.norm(current.G - old.G)), "M": float(np.linalg.norm(current.M - old.M))})
            srv += int(not is_in_safe_region(current))
            coherence_failures += int(not check_self_model_drift(old, current))
        # Comparator measurements occur only in an independent evaluation env.
        baseline = rollout(evaluation_env, repo.registry.policy(baseline_candidate, seed), repo.contract, seed)
        selected = repo.registry.by_skill(skill)
        matched = rollout(evaluation_env, repo.registry.policy(selected, seed), repo.contract, seed)
        comparator = next((c for c in repo.registry.candidates if c.loader == "ppo_deterministic"), baseline_candidate)
        comparison = rollout(evaluation_env, repo.registry.policy(comparator, seed), repo.contract, seed)
        dr = matched["payoff"] - baseline["payoff"]
        dn = np.array(matched["objectives"]) - baseline["objectives"]
        w = result.selection.geometry.weights if result.selection else (.5, .5)
        negative += int(dr + np.dot(w, dn) < -host.epsilon)
        comparable += 1
        lifts.append(matched["payoff"] - comparison["payoff"])
        records.append({"seed": seed, "execution": outcome, "matched_baseline": baseline,
                        "matched_selected": matched, "comparator": comparator.skill_id,
                        "matched_comparator": comparison, "measured_delta_r": dr, "measured_delta_n": dn.tolist()})
    # Fixed observation/evidence, explicit motive shift; no policy/model update.
    obs, _ = operational_env.reset(seed=seeds[0])
    shifts = []
    for (ind, trans, self_goal), override in zip(((.9, .9, .1), (.3, .1, .9)), ((.8, .2), (.2, .8))):
        for state in (host.state_a, host.state_b):
            state.G[0], state.G[1], state.G[5] = ind, trans, self_goal
        host.weight_override = override
        started = time.monotonic()
        result = host.select(obs, Stimulus(.2, .5, .5, .2), actions)
        adaptation_latencies.append(time.monotonic() - started)
        shifts.append({"eligible_ids": [a.id for a in result.eligible_actions],
                       "selection": asdict(result.selection) if result.selection else None,
                       "recovery": asdict(result.recovery) if result.recovery else None})
    survival = len(set(shifts[0]["eligible_ids"]) & set(shifts[1]["eligible_ids"]))
    host.flush_traces()
    return {"selector": selector, "geometry": "MDN_WX" if model else "FULL_SIMPLEX", "records": records,
            "no_selection_rate": ratio(no_selection, len(seeds)), "backup_use_rate": ratio(backup, len(seeds)),
            "success_rate": ratio(successes, comparable), "crash_rate": ratio(crashes, comparable),
            "negative_transfer_rate": ratio(negative, comparable), "negative_transfer_rule": "delta_r + w.dot(delta_n) < -current_epsilon",
            "selection_lift": {"comparator": "bundled deterministic PPO", "count": len(lifts),
                               "mean_payoff_difference": float(np.mean(lifts)) if lifts else None},
            "certificate_survival": ratio(survival, len(shifts[0]["eligible_ids"])), "motive_shift": shifts,
            "adaptation_latency_seconds": adaptation_latencies, "skill_switches": ratio(switches, max(0, len(seeds) - 1)),
            "mean_goal_modulator_variation": float(np.mean(variation)) if variation else None,
            "safe_region_violations": ratio(srv, 2 * comparable), "projection_frequency": ratio(projected, 2 * (comparable - backup)),
            "coherence_failures": ratio(coherence_failures, 2 * comparable),
            "motivational_sampling": "both subsystems at every atomic decision boundary; motivations fixed during policy rollout",
            "subsystem_variation": subsystem_variation, "full_bimonad_diagnostics": full_diagnostics,
            "sampled_contractivity_failures": ratio(contraction_failures, comparable - backup),
            "recovery_latency_seconds": recovery_latencies,
            "support_infeasibility": ratio(model.invalid_support_events - infeasible_before, model.queries - queries_before) if model else None}
