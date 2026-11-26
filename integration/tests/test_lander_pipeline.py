"""Full pipeline on the real MO-LunarLander simulator and SubRep's real pilot policy.

    environment-bound, so it is skipped unless you pass the flag:
    PYTHONPATH=external/subrep python -m pytest integration/tests --run-pipeline -q

Needs the pinned stack (mo-gymnasium 1.3.1, gymnasium 1.4.0, box2d-py 2.3.5, torch, hyperon).
Missing packages skip the module; a wrong version fails loudly, because the
contract refuses to measure on a different simulator.
"""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.pipeline

for package in ("hyperon", "torch"):  # explicit, so the skip never depends on import order
    pytest.importorskip(package)
try:
    from integration.lander.evidence import collect, make_environment
    from integration.lander.storage import CertifiedRepository
except (RuntimeError, ImportError) as exc:
    pytest.skip(f"simulator stack unavailable: {exc}", allow_module_level=True)

import json

import numpy as np

from core.state import MotivationalState, Stimulus
from integration.lander import __main__ as phase1
from integration.lander import complete
from integration.lander.complete import populate
from integration.lander.contextual import ContextualModel, train_checkpoint
from integration.lander.contracts import MeasurementContract, digest, file_digest
from integration.lander.coordinator import Coordinator
from integration.lander.experiments import training_rows
from integration.lander.motives import project
from integration.lander.recovery import approve_backup
from integration.lander.registry import controlled_registry

CONTRACT = MeasurementContract()
STIM = Stimulus(0.2, 0.5, 0.5, 0.2)
PILOT, IDLE = "lander.ppo_deterministic.v1", "lander.noop.v1"


@pytest.fixture(scope="module")
def sim():
    try:
        return make_environment(CONTRACT), make_environment(CONTRACT)  # evaluation, operational
    except ImportError as exc:
        pytest.skip(f"simulator unavailable: {exc}")


@pytest.fixture(scope="module")
def world(sim, tmp_path_factory):
    registry = controlled_registry()
    root = tmp_path_factory.mktemp("pipeline") / "repo"
    repo = CertifiedRepository(root, registry, CONTRACT)
    decisions = populate(repo, sim[0])  # baseline, admit every controller, save, reload
    return SimpleNamespace(repo=repo, registry=registry, root=root, decisions={d["skill_id"]: d for d in decisions},
                           actions={c.loader: c.action for c in registry.candidates})


def idle(world):
    return next(c for c in world.registry.candidates if c.loader == "noop")


def test_measurement_is_deterministic_and_idle_always_crashes(world, sim):
    first, second = collect(sim[0], world.registry, idle(world), CONTRACT), collect(sim[0], world.registry, idle(world), CONTRACT)
    assert digest(first) == digest(second)
    assert all(e["crashed"] and not e["landing_success"] for e in first["episodes"])


def test_only_the_pilot_improves_on_idle(world):
    admitted = {s for s, d in world.decisions.items() if d["admitted"]}
    assert admitted == {IDLE, PILOT}  # idle trivially matches itself
    for loader in ("left_engine", "main_engine", "right_engine", "random"):
        assert world.decisions[f"lander.{loader}.v1"]["reason"] == "GATE_REJECTED"
    assert world.repo.verify_ready() == sorted(admitted)


def test_certified_state_reloads_and_pilot_evidence_replays(world, sim):
    restored = CertifiedRepository(world.root, world.registry, CONTRACT)
    assert restored.load() == sorted([IDLE, PILOT])
    pilot = world.registry.by_skill(PILOT)
    assert digest(collect(sim[0], world.registry, pilot, CONTRACT)) == digest(restored.evidence[PILOT])  # fresh policy, same result


def test_select_and_execute_on_the_simulator(world, sim):
    host = Coordinator(world.repo, epsilon=5.0)
    obs, _ = sim[1].reset(seed=42)
    result = host.select(obs, STIM, list(world.actions.values()))
    assert result.selection.skill_id in {IDLE, PILOT}
    assert {r for _, r in result.rejected} == {"NOT_CERTIFIED"} and len(result.rejected) == 4
    executed, feedback = host.execute(result.selection, sim[1], obs, 42)
    assert executed["skill_id"] == result.selection.skill_id and executed["outcome"]["seed"] == 42
    assert 0 <= feedback.risk <= 1


def test_backup_runs_at_an_evaluated_start_and_is_refused_elsewhere(world, sim, tmp_path):
    approval = tmp_path / "backup.json"
    approve_backup(world.repo, PILOT, approval)  # pilot: 0 crashes, enough landings
    host = Coordinator(world.repo, epsilon=5.0, backup_approval=approval)
    nothing_certified = [world.actions["main_engine"]]

    obs, _ = sim[1].reset(seed=42)  # exactly an evaluated start
    stopped = host.select(obs, STIM, nothing_certified)
    assert stopped.selection is None and stopped.recovery.outcome == "backup"
    executed, _ = host.execute_recovery(stopped.recovery, sim[1], obs, 42)
    assert executed["mode"] == "backup" and executed["skill_id"] == PILOT

    obs, _ = sim[1].reset(seed=99)  # never evaluated
    refused = host.select(obs, STIM, nothing_certified)
    assert refused.recovery.outcome == "stop" and refused.recovery.reason == "BACKUP_NOT_APPLICABLE"


# ------------------------------------------------------- MDN contextual path

TRAIN_SEEDS, HELDOUT_SEEDS = tuple(range(100, 108)), tuple(range(108, 112))


@pytest.fixture(scope="module")
def mdn_world(sim, tmp_path_factory):
    """MDN trained on real measurements (as complete.py does), then both repositories populated."""
    registry = controlled_registry(extended=True)
    root = tmp_path_factory.mktemp("mdn_pipeline")
    rows, _ = training_rows(sim[0], registry, CONTRACT, TRAIN_SEEDS + HELDOUT_SEEDS)
    metadata = train_checkpoint(rows, CONTRACT, root / "mdn.pt", TRAIN_SEEDS, HELDOUT_SEEDS, epochs=600)
    model = ContextualModel(root / "mdn.pt", CONTRACT, weight_owner="metamo")
    geometry = model.geometry(project(MotivationalState(np.full(8, 0.5), np.full(6, 0.5)), STIM, 5.0))
    global_repo = CertifiedRepository(root / "global", registry, CONTRACT)
    contextual_repo = CertifiedRepository(root / "contextual", registry, CONTRACT)
    return SimpleNamespace(metadata=metadata, model=model, geometry=geometry, registry=registry,
                           global_decisions=populate(global_repo, sim[0]),
                           contextual_decisions=populate(contextual_repo, sim[0], geometry),
                           global_repo=global_repo, contextual_repo=contextual_repo)


def admitted(decisions):
    return {d["skill_id"] for d in decisions if d["admitted"]}


def test_mdn_trained_on_real_measurements_passes_held_out_acceptance(mdn_world):
    m = mdn_world.metadata
    assert m["approved"] and not set(m["train_seeds"]) & set(m["validation_seeds"])
    assert mdn_world.geometry.region == "MDN_WX" and mdn_world.geometry.model_digest == mdn_world.model.sha


def test_contextual_certification_keeps_everything_global_certification_admits(mdn_world):
    glob, contextual = admitted(mdn_world.global_decisions), admitted(mdn_world.contextual_decisions)
    assert PILOT in glob and glob <= contextual  # a smaller weight region can only widen admission
    certs = [mdn_world.contextual_repo.verify_entry(s)[1] for s in contextual]
    assert {c.weight_region_type for c in certs} == {"MDN_WX"}
    assert {c.weight_region_type for c in (mdn_world.global_repo.verify_entry(s)[1] for s in glob)} == {"FULL_SIMPLEX"}


def test_contextual_host_selects_and_executes_on_the_simulator(mdn_world, sim):
    host = Coordinator(mdn_world.contextual_repo, epsilon=5.0, contextual_model=mdn_world.model)
    obs, _ = sim[1].reset(seed=42)
    actions = [c.action for c in mdn_world.registry.candidates]
    result = host.select(obs, STIM, actions)
    assert result.selection.geometry.region == "MDN_WX"
    assert result.selection.skill_id in admitted(mdn_world.contextual_decisions)
    executed, _ = host.execute(result.selection, sim[1], obs, 42)
    assert executed["skill_id"] == result.selection.skill_id
    blind = Coordinator(mdn_world.contextual_repo, epsilon=5.0).select(obs, STIM, actions)  # no model: global screening only
    assert blind.selection is None and blind.recovery.reason == "EMPTY_ELIGIBLE_SET"


# ----------------------------------------------- phase-1 runner (python -m integration.lander)

@pytest.fixture(scope="module")
def phase1_run(sim, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase1") / "run"
    return out, phase1.main(["--output", str(out)])


def test_phase1_run_certifies_operates_and_reports(phase1_run):
    out, report = phase1_run
    assert report["committed_skill_ids"] == sorted([IDLE, PILOT]) and report["identity_sets_verified"]
    assert report["training_performed"] is False and report["admission_rate"] == pytest.approx(2 / 6)
    assert report["normal_executions"] + report["no_selection_count"] == report["operational_boundaries"] == 2
    assert report["successes"] + report["crashes"] <= report["normal_executions"]
    assert len(report["motive_shift_screenings"]) == 2
    assert report["empty_set_probe"]["reason"] == "EMPTY_ELIGIBLE_SET"
    saved = json.loads((out / "report.json").read_text())
    assert saved["contract_id"] == report["contract_id"] and saved["committed_skill_ids"] == report["committed_skill_ids"]
    assert list((out / "runs").glob("run-*.json")) and (out / "execution.jsonl").read_text().strip()


def test_phase1_refuses_committed_output_but_resumes_it(phase1_run):
    out, first = phase1_run
    with pytest.raises(SystemExit) as e:
        phase1.main(["--output", str(out)])
    assert e.value.code == 2
    resumed = phase1.main(["--output", str(out), "--resume"])
    assert resumed["committed_skill_ids"] == first["committed_skill_ids"] and resumed["admissions_this_run"] == []


def test_phase1_rejects_non_positive_cycles(tmp_path):
    with pytest.raises(SystemExit) as e:
        phase1.main(["--output", str(tmp_path / "x"), "--cycles", "0"])
    assert e.value.code == 2


# ------------------------------- complete runner (python -m integration.lander.complete)

@pytest.fixture(scope="module")
def complete_run(sim, tmp_path_factory):
    out = tmp_path_factory.mktemp("complete") / "run"
    return out, complete.main(["--output", str(out), "--skip-reproduction", "--epochs", "300"])


def test_complete_run_trains_certifies_both_geometries_and_pins_the_model(complete_run):
    out, r = complete_run
    assert r["model_validation"]["approved"] and r["reproduction"] is None
    assert r["model_digest"] == file_digest(out / "mdn.pt")
    expected = {IDLE, PILOT, "lander.ppo_stochastic.v1", "lander.ppo_noisy.v1"}
    assert set(r["global_ready_ids"]) == set(r["contextual_ready_ids"]) == expected
    assert r["global_admission_rate"]["numerator"] == r["contextual_admission_rate"]["numerator"] == 4
    for name in ("dataset.json", "model_validation.json", "training_evidence.json", "backup_approval.json", "recovery.jsonl"):
        assert (out / name).stat().st_size > 0


def test_complete_run_evaluates_every_selector_in_both_geometries(complete_run):
    _, r = complete_run
    assert len(r["evaluations"]) == 4
    assert {e["selector"] for e in r["evaluations"]} == {"subrep_filter_then_magus", "subrep_selects"}
    assert {e["geometry"] for e in r["evaluations"]} == {"FULL_SIMPLEX", "MDN_WX"}


def test_complete_run_recovery_probes(complete_run):
    _, r = complete_run
    assert r["backup_approval"]["approval"]["skill_id"] == PILOT
    probe = r["recovery_probe"]
    assert probe["decision"]["outcome"] == "backup" and probe["decision"]["reason"] == "APPLICABLE_VALIDATED_BACKUP"
    assert "execution" in probe
    assert (probe["outside_applicability"]["outcome"], probe["outside_applicability"]["reason"]) == ("stop", "BACKUP_NOT_APPLICABLE")
    assert (probe["budget_exhaustion_probe"]["outcome"], probe["budget_exhaustion_probe"]["reason"]) == ("stop", "DECISION_BUDGET_EXHAUSTED")
    assert r["recovery_metrics"]["backup_use_rate"]["value"] == 0.5


def test_complete_run_reevaluation_survives_restart(complete_run):
    _, r = complete_run
    re = r["reevaluation"]
    assert re["restart_attempts"] == 1 and r["scheduler_metrics"]["completed_reevaluations"] == 1
    assert re["penalty_after_one_half_life"] == pytest.approx(re["penalty_before"] / 2)


def test_complete_refuses_an_output_that_already_has_a_report(complete_run):
    out, _ = complete_run
    with pytest.raises(SystemExit) as e:
        complete.main(["--output", str(out), "--skip-reproduction"])
    assert e.value.code == 2


def test_reproduction_of_the_upstream_demo_matches_identity_sets(sim, tmp_path):
    result = complete.reproduce(tmp_path / "repro")
    assert result["identity_sets_equal"] and result["certificate_ids"] == ["skill_001_ppo_deterministic", "skill_005_noop"]
