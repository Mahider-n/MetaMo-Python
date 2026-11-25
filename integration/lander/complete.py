"""Complete document workflow: python -m integration.lander.complete --output ..."""
import argparse
from dataclasses import asdict
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
from core.state import MotivationalState, Stimulus
import numpy as np
from .contracts import BoundaryError, MeasurementContract, file_digest
from .contextual import ContextualModel, train_checkpoint
from .coordinator import Coordinator
from .evidence import collect, make_environment
from .experiments import evaluate_runtime, ratio, training_rows
from .motives import project
from .recovery import approve_backup
from .registry import controlled_registry
from .storage import CertifiedRepository
from .persistence import write_json
from .vendor import SUBREP, source_revision


def reproduce(output):
    """Native SubRep demo, isolated outputs; historical/stub mode is labeled."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    model_link = output / "models/pilot_ppo.pt"
    if not model_link.exists():
        model_link.symlink_to(SUBREP / "models/pilot_ppo.pt")
    script = (
        "import json; from demo import run_full_pipeline as d; "
        "d.NUM_EPISODES=10; d.MAX_STEPS=200; d.BASELINE_EPISODES=20; "
        "result=d.run_pipeline(); "
        "open('result.json','w').write(json.dumps(result,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))"
    )
    import os
    environment = {**os.environ, "PYTHONPATH": str(SUBREP)}
    with (output / "pipeline.log").open("w") as log:
        subprocess.run([sys.executable, "-c", script], cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    from certification.metta_storage import CertificateStore
    store = CertificateStore()
    store.load_from_file(output / "data/certificates.metta")
    library = json.loads((output / "data/library.json").read_text())
    ids = sorted(c.skill_id for c in store.load_all())
    if ids != sorted(library["skills"]):
        raise BoundaryError("PERSISTENCE_CONFLICT", "upstream reproduction IDs differ")
    result = json.loads((output / "result.json").read_text())
    return {"configured_attempts": 10, "baseline_episodes": 20, "candidate_horizon": 200,
            "baseline_horizon": "environment termination (native demo behavior)",
            "baseline_seeds": list(range(42, 62)), "gamma": .99, "epsilon": 5,
            "mdn_source": "upstream explicit stub fallback; not the trained integration model",
            "identity_sets_equal": True, "certificate_ids": ids, "native_result": result,
            "comparison_note": "Separate historical-style reproduction; integration below uses matched baseline/candidate seeds and horizons"}


def populate(repo, env, geometry=None):
    registry = repo.registry
    baseline = next(c for c in registry.candidates if c.loader == "noop")
    repo.set_baseline(collect(env, registry, baseline, repo.contract))
    decisions = []
    for candidate in registry.candidates:
        try:
            decisions.append(repo.admit(candidate, collect(env, registry, candidate, repo.contract), 5., geometry))
        except BoundaryError as exc:
            repo.rejections.record(candidate, repo.contract, exc.code, detail=str(exc))
            decisions.append({"skill_id": candidate.skill_id, "admitted": False, "reason": exc.code})
    repo.save()
    repo.load()
    return decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/complete-integration"))
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--skip-reproduction", action="store_true", help="explicitly omit the separate historical-style demo")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if (output / "report.json").exists():
        parser.error("use a fresh output directory; completed evidence is immutable")
    output.mkdir(parents=True, exist_ok=True)
    registry = controlled_registry(extended=True)
    contract = MeasurementContract(horizon=args.horizon)
    eval_env = make_environment(contract)
    operational = make_environment(contract)
    try:
        reproduction = None if args.skip_reproduction else reproduce(output / "reproduction")
        train_seeds, heldout_seeds = tuple(range(100, 108)), tuple(range(108, 112))
        rows, raw = training_rows(eval_env, registry, contract, train_seeds + heldout_seeds)
        write_json(output / "training_evidence.json", raw)
        write_json(output / "dataset.json", rows)
        metadata = train_checkpoint(rows, contract, output / "mdn.pt", train_seeds, heldout_seeds, args.epochs)
        write_json(output / "model_validation.json", metadata)
        # Fail closed: a merely present checkpoint cannot activate itself.
        model = ContextualModel(output / "mdn.pt", contract, weight_owner="metamo", fallback="error")
        global_repo = CertifiedRepository(output / "global", registry, contract)
        contextual_repo = CertifiedRepository(output / "contextual", registry, contract)
        global_decisions = populate(global_repo, eval_env)
        state = MotivationalState(np.full(8, .5), np.full(6, .5))
        geometry = model.geometry(project(state, Stimulus(.2, .5, .5, .2), 5.))
        contextual_decisions = populate(contextual_repo, eval_env, geometry)
        backup_path = output / "backup_approval.json"
        try:
            backup_result = approve_backup(global_repo, "lander.ppo_deterministic.v1", backup_path)
        except BoundaryError as exc:
            backup_result = {"approved": False, "reason": str(exc)}
        evaluations = []
        for repo, active_model, name in ((global_repo, None, "global"), (contextual_repo, model, "contextual")):
            for selector in ("subrep_filter_then_magus", "subrep_selects"):
                evaluations.append(evaluate_runtime(repo, eval_env, operational, active_model, selector,
                                                    (1046, 1047, 1048), output / f"{name}-{selector}.jsonl"))
        host = Coordinator(global_repo, backup_approval=backup_path if backup_path.exists() else None,
                           trace_path=output / "recovery.jsonl")
        obs, _ = operational.reset(seed=contract.seeds[0])
        empty = host.select(obs, Stimulus(0, 0, 0, 0), [])
        recovery_probe = {"decision": asdict(empty.recovery)}
        if empty.recovery.outcome == "backup":
            recovery_probe["execution"] = host.execute_recovery(empty.recovery, operational, obs, contract.seeds[0])[0]
        out_of_scope = host.select(np.full(8, 100.), Stimulus(0, 0, 0, 0), [])
        recovery_probe["outside_applicability"] = asdict(out_of_scope.recovery)
        budget_host = Coordinator(global_repo, decision_budget_seconds=1e-12)
        exhausted = budget_host.select(obs, Stimulus(0, 0, 0, 0), [])
        recovery_probe["budget_exhaustion_probe"] = asdict(exhausted.recovery)
        host.flush_traces()
        budget_host.flush_traces()
        # Deterministic clock audit advances scheduling time without delaying
        # the operational simulator. Actual rollouts still run independently.
        entries = contextual_repo.scheduler.state["entries"]
        reevaluation = None
        if entries:
            entry = next(v for v in entries.values() if "random" in v["skill_id"])
            scheduler = contextual_repo.scheduler
            simulated_now = entry["last_failure"] + scheduler.config.cooldown_seconds + 1
            scheduler.clock = lambda: simulated_now
            candidate = registry.by_skill(entry["skill_id"])
            changed = model.geometry(project(state, Stimulus(.9, .2, .8, .7), 5.))
            request = scheduler.request(entry["key"], candidate, "context_change", changed.context_id, changed.context_id)
            dispatched = scheduler.dispatch(contextual_repo, eval_env, {changed.context_id: changed}, 5.)
            result = dispatched["result"]
            penalty_before = scheduler.penalty(entry)
            simulated_now += scheduler.config.half_life_seconds
            penalty_after = scheduler.penalty(entry)
            reevaluation = {"request": request, "result": result, "clock_mode": "deterministic scheduling-time audit",
                            "penalty_before": penalty_before, "penalty_after_one_half_life": penalty_after,
                            "admission_rate": ratio(int(result["admitted"]), 1)}
            contextual_repo.load()
            reevaluation["restart_attempts"] = contextual_repo.scheduler.state["entries"][entry["key"]]["attempts"]
        report = {"implementation": "audited_document_v2", "contract": asdict(contract),
                  "audit_fixes": ["exact_factory_identity", "suspension_all_paths", "decision_process_watchdog",
                                  "durable_retry_epochs", "decay_priority_dispatch", "exact_evaluated_backup_starts",
                                  "feedback_replacement_workflow", "expanded_validation_contracts"],
                  "source_revisions": {"subrep": source_revision(SUBREP),
                                       "host_base": source_revision(Path(__file__).resolve().parents[2])},
                  "python": sys.version, "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
                  "reproduction": reproduction, "model_validation": metadata, "model_digest": file_digest(output / "mdn.pt"),
                  "weight_owner": "metamo", "model_fallback": "error", "global_admissions": global_decisions,
                  "contextual_admissions": contextual_decisions,
                  "global_admission_rate": ratio(sum(d["admitted"] for d in global_decisions), len(global_decisions)),
                  "contextual_admission_rate": ratio(sum(d["admitted"] for d in contextual_decisions), len(contextual_decisions)),
                  "evaluations": evaluations, "backup_approval": backup_result,
                  "recovery_probe": recovery_probe, "reevaluation": reevaluation,
                  "recovery_metrics": {"backup_use_rate": ratio(int(empty.recovery.outcome == "backup"), 2),
                                       "decision_budget_exhaustion_rate": ratio(int(exhausted.recovery.reason == "DECISION_BUDGET_EXHAUSTED"), 1),
                                       "probe_denominator_note": "two applicability probes and one separate forced deadline probe"},
                  "scheduler_metrics": {
                      "completed_reevaluations": sum(e["event"] == "attempt_completed" for e in contextual_repo.scheduler.state["events"]),
                      "retry_budget_exhaustion": ratio(sum(e["event"] == "budget_exhausted" for e in contextual_repo.scheduler.state["events"]),
                                                       sum(e["event"] == "attempt_completed" for e in contextual_repo.scheduler.state["events"]))},
                  "global_ready_ids": global_repo.verify_ready(), "contextual_ready_ids": contextual_repo.verify_ready(),
                  "limitations": ["Finite-seed functional evaluation, not universal safety or performance evidence",
                                  "MDN distills a declared motive/support rule; no claim of candidate-conditioned outcome learning",
                                  "Two measured objectives do not calibrate eight predicted goal effects",
                                  "Linux process watchdog bounds decision waiting; OS scheduling/IPC overhead is measured, not hard real time",
                                  "Backup restricted to exact evaluated seed/start pairs; no midflight or expanded-envelope approval",
                                  "Motivations held fixed inside atomic rollouts; both subsystems measured at each boundary",
                                  "Single writer per repository; policy rollout cancellation remains cooperative"]}
        write_json(output / "report.json", report)
        print(json.dumps({"report": str(output / "report.json"), "model_approved": metadata["approved"],
                          "global_certificates": len(report["global_ready_ids"]), "contextual_certificates": len(report["contextual_ready_ids"])}, indent=2))
        return report
    finally:
        eval_env.close()
        operational.close()


if __name__ == "__main__":
    main()
