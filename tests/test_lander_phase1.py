"""Bridge acceptance cases; synthetic evidence is deliberately not a benchmark."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import pytest
from core.state import Action, MotivationalState, Stimulus
from integration.lander.contracts import BoundaryError, MeasurementContract, digest, file_digest, weights
from integration.lander.motives import project
from integration.lander.registry import Registry, controlled_registry
from integration.lander.evidence import matched_improvements, summarize, validate_evidence
from integration.lander.storage import CertifiedRepository, RejectionMemory, write_json
from integration.lander.coordinator import Coordinator


@pytest.fixture
def setup(tmp_path):
    registry = controlled_registry()
    contract = MeasurementContract(seeds=(42,), horizon=1)
    repo = CertifiedRepository(tmp_path, registry, contract)
    by_name = {c.loader: c for c in registry.candidates}
    repo.set_baseline(evidence(by_name["noop"], contract, 0, 0))
    return repo, by_name


def evidence(candidate, contract, safety, fuel):
    episodes = [{"seed": seed, "initial_observation": [0.] * 8,
                 "payoff": safety + fuel, "objectives": [safety, fuel],
                 "terminated": False, "truncated": False, "steps": 1,
                 "stop_reason": "max_steps", "landing_success": False, "crashed": False,
                 "raw_rewards": [[0., safety, fuel, 0.]]} for seed in contract.seeds]
    return {"contract": asdict(contract), "behavior": candidate.metadata(), "episodes": episodes, **summarize(episodes)}


def admit(repo, candidate, safety=3., fuel=-1., epsilon=0):
    return repo.admit(candidate, evidence(candidate, repo.contract, safety, fuel), epsilon)


def state():
    return MotivationalState(np.full(8, .5), np.full(6, .5))


def test_projection_formula_and_monotonicity():
    neutral = project(state(), Stimulus(.5, .5, .5, .5))
    assert neutral.weights == pytest.approx((.5, .5))
    cautious = state()
    cautious.G[0] = .9
    cautious.G[6] = .9
    cautious.M[5] = .9
    assert project(cautious, Stimulus(.5, .5, .9, .5)).weights[0] > neutral.weights[0]
    conserving = state()
    conserving.G[5], conserving.G[1], conserving.M[3] = .9, .1, .9
    assert project(conserving, Stimulus(.5, .5, .5, .9)).weights[1] > neutral.weights[1]
    zero = MotivationalState(np.zeros(8), np.zeros(6))
    zero.G[1] = 1
    fallback = project(zero, Stimulus(0, 0, 0, 0))
    assert fallback.weights == (.5, .5) and fallback.zero_default_used


@pytest.mark.parametrize("bad", [[1, 1], [-.1, 1.1], [float('nan'), 0], [[.5, .5]], [1, 0, 0]])
def test_weights_are_rejected_without_repair(bad):
    with pytest.raises(BoundaryError):
        weights(bad)


@pytest.mark.parametrize("field,value", [("G", np.zeros((8, 1))), ("M", np.full(6, np.inf))])
def test_native_state_rank_and_finiteness(field, value):
    s = state()
    setattr(s, field, value)
    with pytest.raises(BoundaryError, match="INVALID_CONTEXT"):
        project(s, Stimulus(0, 0, 0, 0))


def test_registry_unknown_ambiguous_and_changed_artifacts(setup):
    repo, candidates = setup
    unknown = Action("safe_answer", np.zeros(8), 0, np.zeros(8))
    with pytest.raises(BoundaryError, match="UNKNOWN_BEHAVIOR"):
        repo.registry.resolve(unknown)
    c = candidates["noop"]
    with pytest.raises(BoundaryError, match="IDENTITY_CONFLICT"):
        Registry([c, c])
    with pytest.raises(BoundaryError, match="ARTIFACT_CHANGED"):
        Registry([replace(c, artifact_digest="changed")])


def test_evidence_contract_and_raw_reward_checks(setup):
    repo, c = setup
    e = evidence(c["main_engine"], repo.contract, 3., -1.)
    dr, dn = matched_improvements(repo.baseline, e, repo.contract)
    assert dr == 2 and dn.tolist() == [3, -1]
    e["contract"]["objective_names"] = ["fuel", "safety"]
    with pytest.raises(BoundaryError, match="SCHEMA_MISMATCH"):
        matched_improvements(repo.baseline, e, repo.contract)
    e = evidence(c["main_engine"], repo.contract, 3., -1.)
    e["episodes"][0]["raw_rewards"][0][2] = 1
    with pytest.raises(BoundaryError, match="SCHEMA_MISMATCH"):
        validate_evidence(e, repo.contract)


def test_evidence_rejects_mismatched_reset_seed_and_start(setup):
    repo, c = setup
    e = evidence(c["main_engine"], repo.contract, 3., -1.)
    e["episodes"][0]["seed"] = 99
    with pytest.raises(BoundaryError, match="SCHEMA_MISMATCH"):
        matched_improvements(repo.baseline, e, repo.contract)
    e["episodes"][0]["seed"] = 42
    e["episodes"][0]["initial_observation"][0] = .1
    with pytest.raises(BoundaryError, match="SCHEMA_MISMATCH"):
        matched_improvements(repo.baseline, e, repo.contract)


def test_cds_zero_pds_exact_boundary_and_rejection_persistence(setup):
    repo, c = setup
    assert admit(repo, c["noop"], 0, 0)["certificate"]["admission_margin"] == 0
    # dr=-.5; min(dn)=-1 => robust=-1.5, exactly admissible with epsilon 1.5.
    pds = admit(repo, c["random"], .5, -1., 1.5)
    assert pds["certificate"]["gate_type"] == "PDS"
    assert pds["certificate"]["admission_margin"] == 0
    rejected = admit(repo, c["left_engine"], .49, -1., 1.5)
    assert not rejected["admitted"]
    assert not repo.store.contains(c["left_engine"].skill_id)
    restored = RejectionMemory(repo.root / "rejections.json")
    assert len(restored.records) == 1
    assert restored.records[0]["evidence"]["mean_payoff"] == pytest.approx(-.51)


def test_restart_reverifies_evidence_and_reattaches_policy(setup):
    repo, c = setup
    admit(repo, c["main_engine"])
    restored = CertifiedRepository(repo.root, repo.registry, repo.contract)
    assert restored.load() == [c["main_engine"].skill_id]
    entry = restored.library.get_skill(c["main_engine"].skill_id)
    assert callable(entry.policy) and entry.policy(np.zeros(8)) == 2
    assert restored.store.get_certificate(entry.skill_id).to_dict() == entry.certificate.to_dict()


def test_restart_rejects_legacy_schema_and_partial_writes(setup):
    repo, c = setup
    admit(repo, c["main_engine"])
    pointer = json.loads((repo.root / "CURRENT").read_text())
    directory = repo.root / pointer["generation"]
    # Orphan generation never becomes operational.
    (repo.root / "generation-orphan.pending").mkdir()
    assert CertifiedRepository(repo.root, repo.registry, repo.contract).load()
    wrong_contract = replace(repo.contract, gamma=.9)
    with pytest.raises(BoundaryError, match="SCHEMA_MISMATCH"):
        CertifiedRepository(repo.root, repo.registry, wrong_contract).load()
    (directory / "library.json").write_text('{}')
    with pytest.raises(BoundaryError, match="PERSISTENCE_CONFLICT"):
        CertifiedRepository(repo.root, repo.registry, repo.contract).load()


@pytest.mark.parametrize("raise_error", [False, True])
def test_insertion_rollback_on_false_and_exception(setup, monkeypatch, raise_error):
    repo, c = setup
    admit(repo, c["noop"], 0, 0)
    def broken(*args, **kwargs):
        if raise_error:
            raise OSError("injected library failure")
        return False
    monkeypatch.setattr(repo.library, "add_skill", broken)
    with pytest.raises((OSError, BoundaryError)):
        admit(repo, c["main_engine"])
    assert repo.verify_ready() == [c["noop"].skill_id]
    assert CertifiedRepository(repo.root, repo.registry, repo.contract).load() == [c["noop"].skill_id]


def test_failed_disk_commit_preserves_previous_generation(setup, monkeypatch):
    repo, c = setup
    admit(repo, c["noop"], 0, 0)
    original = Path.replace
    def fail_pointer(path, target):
        if Path(target).name == "CURRENT":
            raise OSError("injected pointer failure")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", fail_pointer)
    with pytest.raises(OSError):
        admit(repo, c["main_engine"])
    assert repo.verify_ready() == [c["noop"].skill_id]
    restored = CertifiedRepository(repo.root, repo.registry, repo.contract)
    assert restored.load() == [c["noop"].skill_id]


def test_screening_preserves_objects_order_and_every_decision_path(setup, monkeypatch):
    repo, c = setup
    admit(repo, c["noop"], 0, 0)
    admit(repo, c["main_engine"])
    host = Coordinator(repo)
    seen = []
    scored = []
    decide = host.bimonad.decision.decide
    def inspect(state, actions):
        seen.append(tuple(a.id for a in actions))
        return decide(state, actions)
    monkeypatch.setattr(host.bimonad.decision, "decide", inspect)
    original_score = host.bimonad.decision.score_candidate
    def inspect_score(state, action):
        scored.append(action.id)
        return original_score(state, action)
    monkeypatch.setattr(host.bimonad.decision, "score_candidate", inspect_score)
    proposals = [c["main_engine"].action, c["random"].action, c["noop"].action]
    result = host.select(np.zeros(8), Stimulus(.2, .5, .5, .2), proposals)
    assert result.eligible_actions[0] is proposals[0] and result.eligible_actions[1] is proposals[2]
    assert all(ids == (proposals[0].id, proposals[2].id) for ids in seen)
    assert set(scored) == {proposals[0].id, proposals[2].id}
    assert result.selection.action_id in result.selection.eligible_ids
    assert result.selection.owner == "subrep_filter_then_magus"


def test_appraisal_once_per_subsystem_and_stricter_epsilon(setup, monkeypatch):
    repo, c = setup
    admit(repo, c["random"], .5, -1., 2)
    host = Coordinator(repo, epsilon=2)
    calls = []
    original = host.bimonad.appraisal.appraise
    def count(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(host.bimonad.appraisal, "appraise", count)
    args = (np.zeros(8), Stimulus(.2, .5, .5, .2), [c["random"].action])
    assert host.select(*args).selection is not None
    assert len(calls) == 2
    host.epsilon = 0
    result = host.select(*args)
    assert result.no_selection and result.recovery.outcome == "stop"
    assert result.rejected == ((c["random"].action.id, "CURRENT_BUDGET_EXCEEDED"),)
    assert repo.store.count() == 1  # Motive/budget changes never rewrite evidence.


def test_empty_set_does_not_invoke_magus_or_change_state(setup, monkeypatch):
    repo, c = setup
    host = Coordinator(repo, trace_path=repo.root / "trace.jsonl")
    old = host.state_a.copy()
    def forbidden(*args):
        pytest.fail("MAGUS must not receive an empty eligible set")
    monkeypatch.setattr(host.bimonad.decision, "decide", forbidden)
    started = time.monotonic()
    result = host.select(np.zeros(8), Stimulus(0, 0, 0, 0), [c["main_engine"].action])
    assert result.no_selection and result.recovery.outcome == "stop"
    assert time.monotonic() - started < host.decision_budget_seconds
    np.testing.assert_array_equal(host.state_a.G, old.G)
    assert [json.loads(line)["event"] for line in host.trace_path.read_text().splitlines()] == ["no_selection", "recovery"]


def test_stale_snapshot_rejected_before_execution(setup):
    repo, c = setup
    admit(repo, c["noop"], 0, 0)
    host = Coordinator(repo)
    obs = np.zeros(8)
    selection = host.select(obs, Stimulus(0, 0, 0, 0), [c["noop"].action]).selection
    host.state_a.G[0] += .1
    with pytest.raises(BoundaryError, match="STALE_SNAPSHOT"):
        host.execute(selection, None, obs, 42)


def test_execution_feedback_applies_only_chosen_prediction_once(setup):
    repo, c = setup
    # Change descriptors before collecting evidence; same executable identity.
    c["main_engine"].action.delta_g[2] = .05
    admit(repo, c["main_engine"])
    host = Coordinator(repo)
    obs = np.zeros(8)
    result = host.select(obs, Stimulus(0, 0, 0, 0), [c["main_engine"].action])
    old_g = host.state_a.G.copy()
    class Env:
        def step(self, action):
            assert action == 2
            return obs, np.array([3., -1.]), False, False, {"raw_rewards": [0, 3, -1, 0]}
        def reset(self, *args, **kwargs):
            pytest.fail("operational execution must not reset the environment")
    outcome, feedback = host.execute(result.selection, Env(), obs, 42)
    assert outcome["cycle_id"] == result.selection.cycle_id
    assert outcome["certificate_digest"] == result.selection.certificate_digest
    assert outcome["outcome"]["stop_reason"] == "max_steps"
    assert host.state_a.G[2] > old_g[2]
    assert feedback.effort == 1
    with pytest.raises(BoundaryError, match="STALE_SNAPSHOT"):
        host.execute(result.selection, Env(), obs, 42)


def test_execution_failure_keeps_state_unchanged(setup):
    repo, c = setup
    admit(repo, c["noop"], 0, 0)
    host = Coordinator(repo, trace_path=repo.root / "trace.jsonl")
    obs = np.zeros(8)
    result = host.select(obs, Stimulus(0, 0, 0, 0), [c["noop"].action])
    before = host.state_a.copy()
    class Broken:
        def step(self, action):
            raise RuntimeError("controller failure")
    with pytest.raises(RuntimeError):
        host.execute(result.selection, Broken(), obs, 42)
    np.testing.assert_array_equal(host.state_a.G, before.G)
    assert json.loads(host.trace_path.read_text().splitlines()[-1])["event"] == "execution_failure"


def test_budget_exhaustion_returns_stop_before_magus(setup, monkeypatch):
    repo, c = setup
    admit(repo, c["noop"], 0, 0)
    host = Coordinator(repo, decision_budget_seconds=1e-12)
    def forbidden(*args):
        pytest.fail("deadline exhausted before MAGUS")
    monkeypatch.setattr(host.bimonad.decision, "decide", forbidden)
    result = host.select(np.zeros(8), Stimulus(0, 0, 0, 0), [c["noop"].action])
    assert result.recovery.reason == "DECISION_BUDGET_EXHAUSTED"


def test_global_motive_reuse_preserves_certificate_and_evidence(setup):
    repo, c = setup
    admit(repo, c["main_engine"])
    before = digest(repo.store.get_certificate(c["main_engine"].skill_id).to_dict())
    host = Coordinator(repo)
    proposals = [c["main_engine"].action]
    first = host.select(np.zeros(8), Stimulus(.5, .5, .5, .5), proposals)
    host.state_a.G[0], host.state_b.G[0] = .9, .9
    second = host.select(np.zeros(8), Stimulus(.5, .5, .5, .5), proposals)
    assert first.selection.geometry.weights != second.selection.geometry.weights
    assert first.selection.eligible_ids == second.selection.eligible_ids
    assert digest(repo.store.get_certificate(c["main_engine"].skill_id).to_dict()) == before


def test_identical_admission_retry_is_idempotent_but_conflicting_evidence_fails(setup):
    repo, c = setup
    first = admit(repo, c["main_engine"])
    assert admit(repo, c["main_engine"]) == first
    assert repo.store.count() == 1
    with pytest.raises(BoundaryError, match="IDENTITY_CONFLICT"):
        admit(repo, c["main_engine"], 4, -1)


def test_semantic_manifest_tamper_is_rejected_even_with_updated_file_hashes(setup):
    repo, c = setup
    admit(repo, c["main_engine"])
    pointer = json.loads((repo.root / "CURRENT").read_text())
    directory = repo.root / pointer["generation"]
    data = json.loads((directory / "manifest.json").read_text())
    data["manifests"][c["main_engine"].skill_id]["behavior"]["artifact_digest"] = "wrong"
    write_json(directory / "manifest.json", data)
    index = json.loads((directory / "commit.json").read_text())
    index["manifest.json"] = file_digest(directory / "manifest.json")
    write_json(directory / "commit.json", index)
    pointer["commit_digest"] = digest(index)
    write_json(repo.root / "CURRENT", pointer)
    with pytest.raises(BoundaryError, match="PERSISTENCE_CONFLICT"):
        CertifiedRepository(repo.root, repo.registry, repo.contract).load()


def test_stateful_random_policy_has_fresh_seeded_lifecycle(setup):
    repo, c = setup
    a = repo.registry.policy(c["random"], 42)
    b = repo.registry.policy(c["random"], 42)
    assert [a(None) for _ in range(50)] == [b(None) for _ in range(50)]
