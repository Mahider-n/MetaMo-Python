"""Real CertifiedRepository + Registry + SubRep gates, with synthetic evidence.

No simulator is run: evidence is built to satisfy `validate_evidence`, and
`rollout` / `collect` are patched. Needs the SubRep submodule, hyperon and
torch (see integration/README.md); the module skips itself otherwise.
Run:  PYTHONPATH=external/subrep python -m pytest integration/tests -q
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

for package in ("hyperon", "torch"):  # explicit, so the skip never depends on import order
    pytest.importorskip(package)
try:
    from integration.lander.storage import CertifiedRepository
except (RuntimeError, ImportError) as exc:  # submodule, hyperon or torch missing
    pytest.skip(f"SubRep stack unavailable: {exc}", allow_module_level=True)

from dataclasses import asdict

from core.state import Action, MotivationalState, Stimulus
from integration.lander import coordinator as coord_mod
from integration.lander import evidence as evidence_mod
from integration.lander.contracts import BoundaryError, MeasurementContract
from integration.lander.coordinator import Coordinator
from integration.lander.motives import project
from integration.lander.recovery import approve_backup
from integration.lander.evidence import summarize
from integration.lander.registry import CandidateSkill, Registry, artifact_digest

CONTRACT = MeasurementContract()


def cand(loader, action_id=None, version="v1", lineage=None):
    action_id = action_id or f"lander.{loader}"
    action = Action(action_id, np.full(8, 0.5), 0.2, np.zeros(8))
    return CandidateSkill(f"{action_id}.{version}", action, loader, artifact_digest(loader),
                          behavior_version=version, lineage_id=lineage)


def evidence_for(candidate, terminal, fuel=0.0, steps=10, contract=CONTRACT):
    """Valid matched-seed evidence: `terminal` reward on the last step, `fuel` cost per step."""
    episodes = []
    for seed in contract.seeds:
        raw = np.zeros((steps, 4))
        raw[:, 2:] = -fuel / 2
        raw[-1, 0] = terminal
        mapped = np.column_stack((raw[:, 0] + raw[:, 1], raw[:, 2] + raw[:, 3]))
        objectives = (mapped * contract.gamma ** np.arange(steps)[:, None]).sum(axis=0)
        episodes.append({"seed": seed, "initial_observation": [seed / 100, 1.4, 0, 0, 0, 0, 0, 0],
                         "payoff": float(objectives.sum()), "objectives": objectives.tolist(),
                         "terminated": True, "truncated": False, "stop_reason": "terminated",
                         "steps": steps, "landing_success": terminal >= 100, "crashed": terminal <= -100,
                         "raw_rewards": raw.tolist()})
    evidence = {"contract": asdict(contract), "behavior": candidate.metadata(), "episodes": episodes}
    evidence.update(summarize(episodes))
    return evidence


STIM = Stimulus(0.1, 0.5, 0.2, 0.3)
START = [0.42, 1.4, 0, 0, 0, 0, 0, 0]  # seed 42's evaluated start in evidence_for()


def code(exc_info):
    return exc_info.value.code


def outcome(crashed=False):
    return {"seed": 42, "steps": 100, "objectives": [10.0, -20.0], "landing_success": not crashed,
            "crashed": crashed, "initial_observation": START}


@pytest.fixture
def world(tmp_path):
    c = {"noop": cand("noop"), "good": cand("main_engine"), "bad": cand("left_engine"),
         "marginal": cand("right_engine")}
    registry = Registry(list(c.values()))
    root = tmp_path / "repo"
    repo = CertifiedRepository(root, registry, CONTRACT)
    repo.set_baseline(evidence_for(c["noop"], -100))
    ev = {"good": evidence_for(c["good"], 100, 0.2),      # lands, small fuel cost: passes CDS
          "bad": evidence_for(c["bad"], -100, 10),        # crashes and burns fuel: fails every gate
          "marginal": evidence_for(c["marginal"], -100, 0.2)}  # needs an epsilon budget
    return SimpleNamespace(repo=repo, registry=registry, c=c, ev=ev, root=root)


def admit_good(w):
    return w.repo.admit(w.c["good"], w.ev["good"])


def generation_dir(w):
    return w.root / json.loads((w.root / "CURRENT").read_text())["generation"]


# ------------------------------------------------------------- baseline

class TestBaseline:
    def test_needed_before_use(self, tmp_path, world):
        fresh = CertifiedRepository(tmp_path / "other", world.registry, CONTRACT)
        with pytest.raises(BoundaryError) as e:
            fresh.baseline_id
        assert code(e) == "INVALID_EVIDENCE"

    def test_must_be_the_idle_controller(self, tmp_path, world):
        fresh = CertifiedRepository(tmp_path / "other", world.registry, CONTRACT)
        with pytest.raises(BoundaryError) as e:
            fresh.set_baseline(world.ev["good"])
        assert code(e) == "INVALID_EVIDENCE"

    def test_cannot_be_replaced(self, world):
        world.repo.set_baseline(evidence_for(world.c["noop"], -100))  # same evidence: fine
        with pytest.raises(BoundaryError) as e:
            world.repo.set_baseline(evidence_for(world.c["noop"], -100, fuel=1))
        assert code(e) == "IDENTITY_CONFLICT"


# ---------------------------------------------------------------- admit

class TestAdmit:
    def test_certifies_an_improvement(self, world):
        result = admit_good(world)
        assert result["admitted"] and result["certificate"]["skill_id"] == "lander.main_engine.v1"
        assert world.repo.verify_ready() == ["lander.main_engine.v1"]
        assert (world.root / "CURRENT").exists()

    def test_rejects_and_remembers_a_worse_controller(self, world):
        result = world.repo.admit(world.c["bad"], world.ev["bad"])
        assert not result["admitted"] and result["reason"] == "GATE_REJECTED"
        assert world.repo.manifests == {} and world.repo.library.get_admitted_skills() == []
        records = json.loads((world.root / "rejections.json").read_text())
        assert [r["skill_id"] for r in records] == ["lander.left_engine.v1"]
        assert len(world.repo.scheduler.state["entries"]) == 1

    def test_epsilon_budget_admits_a_marginal_controller(self, world):
        assert not world.repo.admit(world.c["marginal"], world.ev["marginal"], epsilon=0)["admitted"]
        result = world.repo.admit(world.c["marginal"], world.ev["marginal"], epsilon=5.0)
        assert result["admitted"] and result["certificate"]["epsilon"] == 5.0

    def test_negative_epsilon_is_refused(self, world):
        with pytest.raises(BoundaryError):
            world.repo.admit(world.c["good"], world.ev["good"], epsilon=-1)

    def test_retry_is_idempotent_but_conflicting_retry_is_refused(self, world):
        admit_good(world)
        assert admit_good(world)["admitted"]
        assert len(world.repo.library.get_admitted_skills()) == 1
        with pytest.raises(BoundaryError) as e:
            world.repo.admit(world.c["good"], evidence_for(world.c["good"], 100, 0.4))
        assert code(e) == "IDENTITY_CONFLICT"

    def test_evidence_for_another_behavior_is_refused(self, world):
        with pytest.raises(BoundaryError) as e:
            world.repo.admit(world.c["good"], world.ev["bad"])
        assert code(e) == "IDENTITY_CONFLICT"

    def test_failed_save_rolls_back_only_that_insertion(self, world, monkeypatch):
        real_save = world.repo.save
        monkeypatch.setattr(world.repo, "save", lambda: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            admit_good(world)
        assert world.repo.manifests == {} and world.repo.evidence == {}
        assert world.repo.store.get_certificate("lander.main_engine.v1") is None
        assert world.repo.library.get_skill("lander.main_engine.v1") is None
        monkeypatch.setattr(world.repo, "save", real_save)
        assert admit_good(world)["admitted"]

    def test_tampered_evidence_is_detected(self, world):
        admit_good(world)
        world.repo.evidence["lander.main_engine.v1"]["mean_payoff"] += 1
        with pytest.raises(BoundaryError) as e:
            world.repo.verify_entry("lander.main_engine.v1")
        assert code(e) == "PERSISTENCE_CONFLICT"


# ---------------------------------------------------------- persistence

class TestPersistence:
    def fresh(self, w, contract=CONTRACT):
        return CertifiedRepository(w.root, w.registry, contract)

    def test_roundtrip(self, world):
        admit_good(world)
        restored = self.fresh(world)
        assert restored.load() == ["lander.main_engine.v1"]
        _, cert = restored.verify_entry("lander.main_engine.v1")
        assert cert.skill_id == "lander.main_engine.v1"
        assert callable(restored.library.get_skill("lander.main_engine.v1").policy)

    def test_orphan_from_interrupted_save_is_ignored(self, world):
        admit_good(world)
        (world.root / "generation-dead.pending").mkdir()
        assert self.fresh(world).load() == ["lander.main_engine.v1"]

    def test_modified_generation_fails_closed_and_leaves_instance_empty(self, world):
        admit_good(world)
        library = generation_dir(world) / "library.json"
        library.write_text(library.read_text() + " ")
        restored = self.fresh(world)
        with pytest.raises(BoundaryError) as e:
            restored.load()
        assert code(e) == "PERSISTENCE_CONFLICT" and restored.manifests == {}

    @pytest.mark.parametrize("name", ["../evil", "generation-/../evil"])
    def test_pointer_cannot_escape_the_repository(self, world, name):
        admit_good(world)
        (world.root / "CURRENT").write_text(json.dumps({"generation": name, "commit_digest": "x"}))
        with pytest.raises(BoundaryError) as e:
            self.fresh(world).load()
        assert code(e) == "PERSISTENCE_CONFLICT"

    def test_other_measurement_contract_is_refused(self, world):
        admit_good(world)
        with pytest.raises(BoundaryError) as e:
            self.fresh(world, replace(CONTRACT, horizon=100)).load()
        assert code(e) == "SCHEMA_MISMATCH"


# ------------------------------------------------- verify_entry catches tampering

class TestVerifyEntry:
    SKILL = "lander.main_engine.v1"

    @pytest.fixture
    def repo(self, world):
        admit_good(world)
        return world.repo

    def conflict(self, repo, expected="PERSISTENCE_CONFLICT"):
        with pytest.raises(BoundaryError) as e:
            repo.verify_entry(self.SKILL)
        assert code(e) == expected

    def test_partial_admission(self, repo):
        del repo.manifests[self.SKILL]
        self.conflict(repo)

    @pytest.mark.parametrize("field", ["version", "contract_id", "certificate_digest", "baseline_digest",
                                       "evidence_digest", "mapping_version", "budget_version", "context_schema"])
    def test_every_manifest_binding_is_checked(self, repo, field):
        repo.manifests[self.SKILL][field] = "tampered"
        self.conflict(repo)

    def test_behavior_must_match_the_registered_one(self, repo):
        repo.manifests[self.SKILL]["behavior"]["provider"] = "someone_else"
        self.conflict(repo)

    def test_manifest_contract_must_match_the_repository(self, repo):
        repo.manifests[self.SKILL]["contract"]["horizon"] = 100
        self.conflict(repo, "SCHEMA_MISMATCH")

    def test_global_certificate_cannot_carry_contextual_geometry(self, repo):
        repo.manifests[self.SKILL]["geometry"] = asdict(project(MotivationalState(np.full(8, 0.5), np.full(6, 0.5)), STIM))
        self.conflict(repo, "SCHEMA_MISMATCH")

    def test_identity_sets_must_agree_everywhere(self, repo):
        repo.library.remove_skill(self.SKILL)
        with pytest.raises(BoundaryError) as e:
            repo.verify_ready()
        assert code(e) == "PERSISTENCE_CONFLICT"

    def test_evidence_must_exist_for_every_certificate(self, repo):
        repo.evidence.pop(self.SKILL)
        with pytest.raises(BoundaryError) as e:
            repo.verify_ready()
        assert code(e) == "PERSISTENCE_CONFLICT"


# ---------------------------------------- coordinator on the real store

class TestCoordinatorOnRealRepository:
    @pytest.fixture(autouse=True)
    def patch_rollout(self, monkeypatch):
        self.result = outcome()
        monkeypatch.setattr(coord_mod, "rollout", lambda *a, **k: self.result)

    def coordinator(self, world, **kw):
        return Coordinator(world.repo, deadline_mode="cooperative", **kw)

    def test_only_certified_controllers_are_eligible(self, world):
        admit_good(world)
        world.repo.admit(world.c["bad"], world.ev["bad"])
        res = self.coordinator(world).select(START, STIM, [world.c["good"].action, world.c["bad"].action])
        assert res.selection.skill_id == "lander.main_engine.v1"
        assert dict(res.rejected) == {"lander.left_engine": "NOT_CERTIFIED"}

    def test_modified_descriptor_is_not_the_registered_behavior(self, world):
        admit_good(world)
        altered = replace(world.c["good"].action, risk_estimate=0.9)
        res = self.coordinator(world).select(START, STIM, [altered])
        assert dict(res.rejected) == {"lander.main_engine": "INVALID_ACTION"}

    def test_stop_then_validated_backup(self, world, tmp_path):
        admit_good(world)
        approval = tmp_path / "backup.json"
        approve_backup(world.repo, "lander.main_engine.v1", approval)
        c = self.coordinator(world, backup_approval=approval)
        res = c.select(START, STIM, [world.c["bad"].action])  # nothing certified for this set
        assert res.recovery.outcome == "backup" and res.recovery.backup_skill_id == "lander.main_engine.v1"
        executed, _ = c.execute_recovery(res.recovery, None, START, 42)
        assert executed["mode"] == "backup"

    def test_backup_not_offered_away_from_evaluated_starts(self, world, tmp_path):
        admit_good(world)
        approval = tmp_path / "backup.json"
        approve_backup(world.repo, "lander.main_engine.v1", approval)
        res = self.coordinator(world, backup_approval=approval).select(
            [0.9] + START[1:], STIM, [world.c["bad"].action])
        assert res.recovery.outcome == "stop" and res.recovery.reason == "BACKUP_NOT_APPLICABLE"

    def test_unexpected_crash_leads_to_a_certified_replacement(self, world, tmp_path, monkeypatch):
        admit_good(world)
        c = self.coordinator(world, suspend_unexpected=True)
        selection = c.select(START, STIM, [world.c["good"].action]).selection
        self.result = outcome(crashed=True)
        c.execute(selection, None, START, 42)
        event = c.reviews.read()["events"][0]
        assert event["reason"] == "UNEXPECTED_CRASH" and event["status"] == "open"

        # The replacement is a new immutable version in the same lineage,
        # certified in its own repository against its own evidence.
        fixed = cand("main_engine", action_id="lander.main_engine_fix", version="v2", lineage="lander.main_engine")
        review_repo = CertifiedRepository(tmp_path / "review", Registry([world.c["noop"], fixed]), CONTRACT)
        review_repo.set_baseline(evidence_for(world.c["noop"], -100))
        request = c.reviews.enqueue(event["event_id"], review_repo, fixed)
        assert request["reason"] == "outcome_review"
        assert c.reviews.enqueue(event["event_id"], review_repo, fixed) == request  # idempotent

        monkeypatch.setattr(evidence_mod, "collect", lambda env, registry, candidate, contract: evidence_for(fixed, 100, 0.2))
        dispatched = c.reviews.dispatch(review_repo, evaluation_env=None)
        assert dispatched["result"]["admitted"]
        state = c.reviews.read()
        assert state["events"][0]["status"] == "resolved_with_replacement"
        assert state["suspended"] == ["lander.main_engine.v1"]  # a passing repeat never unsuspends the original
        assert review_repo.verify_ready() == ["lander.main_engine_fix.v2"]

    def test_review_needs_a_new_version_in_the_same_lineage(self, world, tmp_path):
        admit_good(world)
        c = self.coordinator(world)
        selection = c.select(START, STIM, [world.c["good"].action]).selection
        self.result = outcome(crashed=True)
        c.execute(selection, None, START, 42)
        event_id = c.reviews.read()["events"][0]["event_id"]
        stranger = cand("main_engine", action_id="lander.other", version="v2")  # different lineage
        review_repo = CertifiedRepository(tmp_path / "review", Registry([world.c["noop"], stranger]), CONTRACT)
        review_repo.set_baseline(evidence_for(world.c["noop"], -100))
        with pytest.raises(BoundaryError) as e:
            c.reviews.enqueue(event_id, review_repo, stranger)
        assert code(e) == "IDENTITY_CONFLICT"

    def test_review_cannot_reuse_the_crashed_skill_id(self, world, tmp_path):
        admit_good(world)
        c = self.coordinator(world)
        selection = c.select(START, STIM, [world.c["good"].action]).selection
        self.result = outcome(crashed=True)
        c.execute(selection, None, START, 42)
        event_id = c.reviews.read()["events"][0]["event_id"]
        review_repo = CertifiedRepository(tmp_path / "review", world.registry, CONTRACT)
        review_repo.set_baseline(evidence_for(world.c["noop"], -100))
        with pytest.raises(BoundaryError) as e:
            c.reviews.enqueue(event_id, review_repo, world.c["good"])  # same skill_id, same lineage
        assert code(e) == "IDENTITY_CONFLICT"

    # --- review workflow edges -------------------------------------------------

    def crashed(self, world, **kw):
        admit_good(world)
        c = self.coordinator(world, **kw)
        selection = c.select(START, STIM, [world.c["good"].action]).selection
        self.result = outcome(crashed=True)
        c.execute(selection, None, START, 42)
        return c, c.reviews.read()["events"][0]["event_id"]

    def review_setup(self, world, tmp_path, contract=CONTRACT):
        fixed = cand("main_engine", action_id="lander.main_engine_fix", version="v2", lineage="lander.main_engine")
        review_repo = CertifiedRepository(tmp_path / "review", Registry([world.c["noop"], fixed]), contract)
        review_repo.set_baseline(evidence_for(world.c["noop"], -100, contract=contract))
        return fixed, review_repo

    def test_open_anomalies_are_listed_for_repair(self, world):
        c, event_id = self.crashed(world)
        assert [e["event_id"] for e in c.reviews.repair_requests()] == [event_id]

    def test_nothing_queued_means_nothing_dispatched(self, world, tmp_path):
        c, _ = self.crashed(world)
        _, review_repo = self.review_setup(world, tmp_path)
        assert c.reviews.dispatch(review_repo, None) is None

    def test_review_contract_must_include_the_anomalous_seed(self, world, tmp_path):
        c, event_id = self.crashed(world)
        other = MeasurementContract(seeds=(1, 2, 3, 4, 5))
        fixed, review_repo = self.review_setup(world, tmp_path, other)
        with pytest.raises(BoundaryError) as e:
            c.reviews.enqueue(event_id, review_repo, fixed)
        assert code(e) == "SCHEMA_MISMATCH"

    def test_feedback_must_refer_to_the_admitted_artifact(self, world, tmp_path):
        c, event_id = self.crashed(world)
        state = c.reviews.read()
        state["events"][0]["artifact_digest"] = "someone_elses"
        c.reviews.save(state)
        fixed, review_repo = self.review_setup(world, tmp_path)
        with pytest.raises(BoundaryError) as e:
            c.reviews.enqueue(event_id, review_repo, fixed)
        assert code(e) == "IDENTITY_CONFLICT"

    def test_replacement_that_fails_certification_stays_open_for_repair(self, world, tmp_path, monkeypatch):
        c, event_id = self.crashed(world)
        fixed, review_repo = self.review_setup(world, tmp_path)
        c.reviews.enqueue(event_id, review_repo, fixed)
        monkeypatch.setattr(evidence_mod, "collect", lambda env, registry, candidate, contract: evidence_for(fixed, -100, 10))
        dispatched = c.reviews.dispatch(review_repo, None)
        assert not dispatched["result"]["admitted"]
        assert [(e["event_id"], e["status"]) for e in c.reviews.repair_requests()] == [(event_id, "repair_required")]
