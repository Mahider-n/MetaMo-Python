"""Run matched-evidence admission, verified restart, and operational cycles."""
import argparse
from dataclasses import asdict
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
import uuid
from core.state import Stimulus
from .contracts import BoundaryError, MeasurementContract, digest
from .coordinator import Coordinator
from .evidence import collect, make_environment
from .registry import controlled_registry
from .storage import CertifiedRepository, write_json
from .vendor import SUBREP


def source_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lander-phase1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.cycles <= 0:
        parser.error("--cycles must be positive")
    contract = MeasurementContract(seeds=tuple(args.seeds), horizon=args.horizon, gamma=args.gamma)
    registry = controlled_registry()
    repo = CertifiedRepository(args.output / "repository", registry, contract)
    if (repo.root / "CURRENT").exists() and not args.resume:
        parser.error("output already has committed evidence; use --resume or a new --output")
    env = make_environment(contract)
    decisions = []
    try:
        if args.resume:
            repo.load()
        else:
            baseline_candidate = next(c for c in registry.candidates if c.loader == "noop")
            repo.set_baseline(collect(env, registry, baseline_candidate, contract))
            for candidate in registry.candidates:
                try:
                    evidence = collect(env, registry, candidate, contract)
                    decisions.append(repo.admit(candidate, evidence, args.epsilon))
                except BoundaryError as exc:
                    repo.rejections.record(candidate, contract, exc.code, detail=str(exc))
                    raise
            repo.save()  # Also commit a valid empty library if every candidate failed.
            # Exercise real persistence and executable reattachment before use.
            repo = CertifiedRepository(repo.root, registry, contract)
            repo.load()
        ids = repo.verify_ready()
        coordinator = Coordinator(repo, epsilon=args.epsilon, trace_path=args.output / "execution.jsonl")
        stimulus = Stimulus(.2, .5, .5, .2)
        outcomes, stops, geometries = [], [], []
        for cycle in range(args.cycles):
            seed = max(contract.seeds) + 1000 + cycle
            obs, _ = env.reset(seed=seed)  # Host begins a new operational episode.
            result = coordinator.select(obs, stimulus, [c.action for c in registry.candidates])
            if result.selection is None:
                stops.append(asdict(result.recovery))
                continue
            geometries.append(asdict(result.selection.geometry))
            outcome, stimulus = coordinator.execute(result.selection, env, obs, seed)
            outcomes.append(outcome)
        # Same observation, policies and evidence; different motives, no training.
        shift = []
        obs, _ = env.reset(seed=max(contract.seeds) + 2000)
        for ind, trans, self_goal in ((.9, .9, .1), (.3, .1, .9)):
            for state in (coordinator.state_a, coordinator.state_b):
                state.G[0], state.G[1], state.G[5] = ind, trans, self_goal
            result = coordinator.select(obs, Stimulus(.2, .5, .5, .2), [c.action for c in registry.candidates])
            shift.append({"eligible_ids": [a.id for a in result.eligible_actions],
                          "selection": asdict(result.selection) if result.selection else None,
                          "recovery": asdict(result.recovery) if result.recovery else None})
        # An explicit empty proposal tests stop behavior with the actual host.
        empty = coordinator.select(obs, Stimulus(0, 0, 0, 0), [])
        report = {
            "phase": 1, "protocol": "matched-seed bridge validation; not historical demo reproduction",
            "contract": asdict(contract), "contract_id": contract.identity,
            "selection_owner": "subrep_filter_then_magus", "geometry": "FULL_SIMPLEX",
            "policy_source": "SubRep bundled PPO checkpoint and controlled fixed/random policy equivalents",
            "mdn_source": "none; Phase 2 deferred", "training_performed": False,
            "source_revisions": {"host_base": source_revision(Path(__file__).resolve().parents[2]),
                                 "subrep": source_revision(SUBREP)},
            "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
            "python": sys.version, "admissions_this_run": decisions,
            "committed_skill_ids": ids, "identity_sets_verified": True,
            "rejected_attempts_total": len(repo.rejections.records),
            "operational_boundaries": args.cycles, "normal_executions": len(outcomes),
            "no_selection_count": len(stops), "no_selection_rate": len(stops) / args.cycles,
            "backup_use_rate": 0.0, "recovery_policy": "stop_only_v1",
            "successes": sum(o["outcome"]["landing_success"] for o in outcomes),
            "crashes": sum(o["outcome"]["crashed"] for o in outcomes),
            "outcomes": outcomes, "stops": stops, "motive_shift_screenings": shift,
            "empty_set_probe": asdict(empty.recovery),
            "limitations": ["Small functional validation, not a performance benchmark",
                             "Certificates describe matched-seed mean evidence, not trajectory guarantees",
                             "Atomic policies run up to the configured horizon; no mid-rollout reselection",
                             "Stop-only recovery; no operational backup controller approved",
                             "No trained MDN, reevaluation scheduler or eight-goal outcome calibration"],
        }
        if decisions:
            report["admission_rate"] = sum(d["admitted"] for d in decisions) / len(decisions)
        (args.output / "runs").mkdir(exist_ok=True)
        write_json(args.output / "runs" / ("run-" + uuid.uuid4().hex + ".json"), report)
        write_json(args.output / "report.json", report)
        print(json.dumps({"report": str(args.output / "report.json"), "certified": ids,
                          "executions": len(outcomes), "stops": len(stops)}, indent=2))
        return report
    finally:
        env.close()


if __name__ == "__main__":
    main()
