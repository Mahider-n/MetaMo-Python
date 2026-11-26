"""Core-logic tests for integration/lander (MetaMo–SubRep bridge).

Run from the repo root:  python -m pytest integration/tests -q

No SubRep stack needed: the repository is a small fake and `rollout` is
patched. The real repository is covered in test_lander_repository.py.
"""
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.state import Action, MotivationalState, Stimulus
from integration.lander import coordinator as coord_mod
from integration.lander.contracts import (BoundaryError, MeasurementContract, budget, canonical, digest,
                                          vector, weights)
from integration.lander.coordinator import Coordinator, outcome_stimulus
from integration.lander.motives import MotiveGeometry, project, robust_score
from integration.lander.persistence import atomic_write_json
from integration.lander.recovery import RecoveryPolicy, approve_backup
from integration.lander.registry import CandidateSkill
from integration.lander.scheduler import ReevaluationScheduler, SchedulerConfig

OBS = [0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def code(exc_info):
    return exc_info.value.code


# ---------------------------------------------------------------- fakes

def make_candidate(name, risk=0.2, lineage=None, artifact="a"):
    action = Action(f"lander.{name}", np.full(8, 0.5), risk, np.zeros(8))
    return CandidateSkill(f"lander.{name}.v1", action, name, artifact + name, lineage_id=lineage)


class FakeCert:
    def __init__(self, delta_r=1.0, delta_n=(1.0, 1.0), region="FULL_SIMPLEX", gate="CDS"):
        self.delta_r, self.delta_n = delta_r, np.array(delta_n, dtype=float)
        self.weight_region_type, self.gate_type, self.baseline_id = region, gate, "baseline-1"

    def to_dict(self):
        return {"delta_r": self.delta_r, "delta_n": self.delta_n.tolist(),
                "region": self.weight_region_type, "gate": self.gate_type}


class FakeRegistry:
    def __init__(self, candidates, delay=0.0):
        self.by_action = {c.action.id: c for c in candidates}
        self.by_skill_id = {c.skill_id: c for c in candidates}
        self.delay = delay

    def resolve(self, action):
        time.sleep(self.delay)
        if action.id not in self.by_action:
            raise BoundaryError("UNKNOWN_BEHAVIOR", action.id)
        return self.by_action[action.id]

    def by_skill(self, skill_id):
        return self.by_skill_id[skill_id]

    def policy(self, candidate, seed):
        return lambda obs: 0


class FakeRepository:
    """certs: {skill_id: FakeCert}. Only certified skills are admitted."""

    def __init__(self, root, candidates, certs, delay=0.0):
        self.root = Path(root)
        self.registry = FakeRegistry(candidates, delay)
        self.certs = certs
        self.contract = MeasurementContract()
        self.baseline_id = "baseline-1"
        self.library = SimpleNamespace(get_admitted_skills=lambda: [
            SimpleNamespace(skill_id=s, weight_region_type=c.weight_region_type) for s, c in certs.items()])
        self.manifests = {}
        self.evidence = {s: {"episodes": [good_episode(seed) for seed in self.contract.seeds]} for s in certs}

    def verify_entry(self, skill_id):
        return self.registry.by_skill(skill_id), self.certs[skill_id]


def good_episode(seed, crashed=False):
    return {"seed": seed, "landing_success": not crashed, "crashed": crashed,
            "initial_observation": [0.0, 1.4, 0, 0, 0, 0, 0, 0]}


def make_outcome(crashed=False, success=True):
    return {"seed": 42, "steps": 100, "objectives": [10.0, -20.0], "landing_success": success,
            "crashed": crashed, "initial_observation": OBS}


@pytest.fixture
def pool(tmp_path):
    names = ("good", "weak", "uncertified")
    cands = [make_candidate(n) for n in names]
    certs = {"lander.good.v1": FakeCert(2.0, (2.0, 1.0)), "lander.weak.v1": FakeCert(-5.0, (-5.0, -5.0))}
    repo = FakeRepository(tmp_path, cands, certs)
    return repo, {c.action.id.split(".")[1]: c.action for c in cands}


def make_coordinator(repo, **kw):
    kw.setdefault("deadline_mode", "cooperative")
    return Coordinator(repo, **kw)


STIM = Stimulus(0.1, 0.5, 0.2, 0.3)


# ------------------------------------------------------------ contracts

class TestContracts:
    def test_vector_rejects_bad_shape_nan_and_range(self):
        assert vector([0, 1], 2).shape == (2,)
        for bad in ([1], [0, np.nan], "abc"):
            with pytest.raises(BoundaryError):
                vector(bad, 2)
        with pytest.raises(BoundaryError):
            vector([0, 1.5], 2, unit=True)

    def test_weights_must_be_simplex(self):
        assert weights([0.25, 0.75]) == (0.25, 0.75)
        for bad in ([0.5, 0.6], [-0.1, 1.1]):
            with pytest.raises(BoundaryError) as e:
                weights(bad)
            assert code(e) == "INVALID_GEOMETRY"

    def test_weights_sum_tolerance_is_tight(self):
        with pytest.raises(BoundaryError):
            weights([0.5, 0.5000001])

    def test_budget(self):
        assert budget(0) == 0.0
        for bad in (-1, float("inf"), "1"):
            with pytest.raises(BoundaryError):
                budget(bad)

    def test_canonical_is_order_independent_and_rejects_nan(self):
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
        with pytest.raises(ValueError):
            canonical({"x": float("nan")})

    def test_measurement_contract_defaults_and_identity(self):
        c = MeasurementContract()
        assert c.identity == MeasurementContract().identity
        assert replace(c, gamma=0.9).identity != c.identity

    @pytest.mark.parametrize("override", [
        {"seeds": ()}, {"seeds": (1, 1)}, {"seeds": (-1,)}, {"horizon": 0}, {"horizon": 1.5},
        {"gamma": 1.5}, {"objective_names": ("fuel", "safety")}, {"scalarization": "other"}])
    def test_measurement_contract_rejects_bad_fields(self, override):
        with pytest.raises(BoundaryError) as e:
            MeasurementContract(**override)
        assert code(e) == "SCHEMA_MISMATCH"

    def test_require_detects_difference(self):
        c = MeasurementContract()
        c.require(MeasurementContract())
        with pytest.raises(BoundaryError):
            c.require(MeasurementContract(horizon=100))


# -------------------------------------------------------------- motives

class TestMotives:
    def test_project_weights_sum_to_one(self):
        state = MotivationalState(np.full(8, 0.5), np.full(6, 0.5))
        g = project(state, STIM, epsilon=0.1)
        assert sum(g.weights) == pytest.approx(1.0)
        assert g.region == "FULL_SIMPLEX" and g.epsilon == 0.1 and not g.zero_default_used

    def test_project_is_deterministic(self):
        state = MotivationalState(np.full(8, 0.3), np.full(6, 0.7))
        assert project(state, STIM).context_id == project(state, STIM).context_id

    def test_project_rejects_out_of_range_state(self):
        state = MotivationalState(np.full(8, 1.5), np.full(6, 0.5))
        with pytest.raises(BoundaryError):
            project(state, STIM)

    def test_zero_total_falls_back_to_even_weights(self):
        from core.config import G_TRANS
        G = np.zeros(8)
        G[G_TRANS] = 1.0
        g = project(MotivationalState(G, np.zeros(6)), Stimulus(0, 0, 0, 0))
        assert g.weights == (0.5, 0.5) and g.zero_default_used

    def _geometry(self, **kw):
        context = tuple([0.5] * 18)
        base = dict(context_id=digest(list(context)), context=context, weights=(0.5, 0.5), epsilon=0.0)
        base.update(kw)
        return MotiveGeometry(**base)

    def test_geometry_rejects_wrong_context_id(self):
        with pytest.raises(BoundaryError) as e:
            self._geometry(context_id="nope")
        assert code(e) == "INVALID_GEOMETRY"

    def test_global_geometry_cannot_claim_contextual_coverage(self):
        with pytest.raises(BoundaryError):
            self._geometry(alpha=(1.0, 1.0))

    def test_contextual_geometry_requirements(self):
        ok = dict(region="MDN_WX", alpha=(1.0, 1.0), support_values=(0.6, 0.6),
                  model_digest="d", model_path="m.pt", weight_owner="mdn")
        assert self._geometry(**ok).region == "MDN_WX"
        with pytest.raises(BoundaryError) as e:
            self._geometry(**{**ok, "weights": (0.9, 0.1)})
        assert code(e) == "WEIGHT_OUTSIDE_SUPPORT"
        with pytest.raises(BoundaryError):
            self._geometry(**{**ok, "support_values": (0.2, 0.2)})  # support cannot cover the simplex
        with pytest.raises(BoundaryError):
            self._geometry(**{**ok, "model_digest": None})

    @pytest.mark.parametrize("override, expected", [
        ({"objective_names": ("fuel", "safety")}, "SCHEMA_MISMATCH"), ({"contract_version": "old"}, "SCHEMA_MISMATCH"),
        ({"region": "SOMETHING"}, "INVALID_GEOMETRY"), ({"weight_owner": "nobody"}, "INVALID_GEOMETRY")])
    def test_geometry_rejects_unsupported_schema(self, override, expected):
        with pytest.raises(BoundaryError) as e:
            self._geometry(**override)
        assert code(e) == expected

    def test_robust_score_full_simplex_is_worst_case_objective(self):
        assert robust_score(1.0, np.array([3.0, -2.0])) == pytest.approx(-1.0)


# ---------------------------------------------------------- persistence

def test_atomic_write_json_roundtrip(tmp_path):
    p = tmp_path / "sub" / "x.json"
    atomic_write_json(p, {"b": 1, "a": [1, 2]})
    assert json.loads(p.read_text()) == {"a": [1, 2], "b": 1}
    assert not p.with_suffix(".tmp").exists()


# -------------------------------------------------------------- feedback

class TestOutcomeStimulus:
    def test_success(self):
        s = outcome_stimulus(make_outcome())
        assert (s.conduciveness, s.risk) == (1.0, 0.0) and s.effort == pytest.approx(0.2)

    def test_crash(self):
        s = outcome_stimulus(make_outcome(crashed=True, success=False))
        assert (s.conduciveness, s.risk) == (0.0, 1.0)

    def test_neither_landed_nor_crashed(self):
        assert outcome_stimulus(make_outcome(success=False)).risk == 0.5

    def test_effort_is_clipped(self):
        o = make_outcome()
        o["objectives"], o["steps"] = [0, -5000], 10
        assert outcome_stimulus(o).effort == 1.0


# ----------------------------------------------------------- coordinator

class TestSelect:
    def test_screens_then_selects_certified_action(self, pool):
        repo, a = pool
        res = make_coordinator(repo).select(OBS, STIM, [a["good"], a["weak"], a["uncertified"]])
        assert res.selection.action_id == "lander.good"
        assert res.selection.skill_id == "lander.good.v1"
        assert dict(res.rejected) == {"lander.weak": "CURRENT_BUDGET_EXCEEDED",
                                      "lander.uncertified": "NOT_CERTIFIED"}
        assert [x.id for x in res.eligible_actions] == ["lander.good"]

    def test_epsilon_widens_budget(self, pool):
        repo, a = pool
        res = make_coordinator(repo, epsilon=100.0).select(OBS, STIM, [a["weak"]])
        assert res.selection.action_id == "lander.weak"

    def test_empty_eligible_set_stops(self, pool):
        repo, a = pool
        res = make_coordinator(repo).select(OBS, STIM, [a["weak"], a["uncertified"]])
        assert res.selection is None and res.recovery.outcome == "stop"
        assert res.recovery.reason == "EMPTY_ELIGIBLE_SET"
        assert res.no_selection.recovery_decision_id == res.recovery.decision_id

    def test_unknown_action_is_rejected_not_fatal(self, pool):
        repo, a = pool
        stranger = Action("lander.stranger", np.full(8, 0.5), 0.2, np.zeros(8))
        res = make_coordinator(repo).select(OBS, STIM, [stranger, a["good"]])
        assert dict(res.rejected)["lander.stranger"] == "UNKNOWN_BEHAVIOR"
        assert res.selection.action_id == "lander.good"

    def test_suspended_skill_is_rejected(self, pool):
        repo, a = pool
        c = make_coordinator(repo)
        c.feedback_state["suspended"].append("lander.good.v1")
        res = c.select(OBS, STIM, [a["good"]])
        assert dict(res.rejected) == {"lander.good": "SUSPENDED_FOR_REEVALUATION"}

    def test_duplicate_ids_and_bad_inputs_raise(self, pool):
        repo, a = pool
        c = make_coordinator(repo)
        with pytest.raises(BoundaryError) as e:
            c.select(OBS, STIM, [a["good"], a["good"]])
        assert code(e) == "IDENTITY_CONFLICT"
        with pytest.raises(BoundaryError):
            c.select(OBS[:5], STIM, [a["good"]])
        with pytest.raises(BoundaryError):
            c.select(OBS, Stimulus(0, 2.0, 0, 0), [a["good"]])

    def test_process_mode_rejects_oversized_action_list(self, pool):
        repo, a = pool
        with pytest.raises(BoundaryError) as e:
            Coordinator(repo).select(OBS, STIM, [a["good"]] * 129)
        assert code(e) == "INVALID_ACTION"

    def test_constructor_validation(self, pool):
        repo, _ = pool
        for kw in ({"selector": "x"}, {"decision_budget_seconds": 0}, {"deadline_mode": "x"}):
            with pytest.raises(ValueError):
                Coordinator(repo, **kw)

    def test_subrep_selects_uses_certificate_score(self, tmp_path):
        cands = [make_candidate("low"), make_candidate("high")]
        certs = {"lander.low.v1": FakeCert(1.0, (1.0, 1.0)), "lander.high.v1": FakeCert(5.0, (5.0, 5.0))}
        repo = FakeRepository(tmp_path, cands, certs)
        res = make_coordinator(repo, selector="subrep_selects").select(OBS, STIM, [c.action for c in cands])
        assert res.selection.action_id == "lander.high" and res.selection.owner == "subrep_selects"

    def test_weight_override_replaces_motive_weights(self, pool):
        repo, a = pool
        res = make_coordinator(repo, weight_override=(1.0, 0.0)).select(OBS, STIM, [a["good"]])
        g = res.selection.geometry
        assert g.weights == (1.0, 0.0) and g.geometry_source == "explicit_configured_weights"

    def test_consensus_penalizes_subsystem_disagreement(self, pool, tmp_path):
        repo, a = pool
        trace = tmp_path / "t.jsonl"
        sa = MotivationalState(np.full(8, 0.9), np.full(6, 0.9))
        sb = MotivationalState(np.full(8, 0.1), np.full(6, 0.1))
        c = make_coordinator(repo, state_a=sa, state_b=sb, trace_path=trace)
        c.select(OBS, STIM, [a["good"]])
        decision = c.bimonad.decision
        x = decision.score_candidate(c.bimonad.decision_context(sa, STIM), a["good"])
        y = decision.score_candidate(c.bimonad.decision_context(sb, STIM), a["good"])
        assert x != pytest.approx(y)  # otherwise this test proves nothing
        logged = json.loads(trace.read_text().splitlines()[0])["consensus_scores"]
        assert logged == [pytest.approx((x + y) / 2 - 0.25 * abs(x - y))]

    def test_global_screening_ignores_contextual_certificates(self, pool):
        repo, a = pool
        repo.certs["lander.good.v1"] = FakeCert(region="MDN_WX")
        res = make_coordinator(repo).select(OBS, STIM, [a["good"]])
        assert dict(res.rejected) == {"lander.good": "NOT_CERTIFIED"}

    def test_contextual_entry_with_changed_checkpoint_is_rejected(self, pool):
        repo, a = pool
        repo.certs["lander.good.v1"] = FakeCert(region="MDN_WX")
        repo.manifests["lander.good.v1"] = {"geometry": {"model_digest": "old"}}
        repo.library.query_admissible = lambda w, **kw: [SimpleNamespace(skill_id="lander.good.v1")]
        context = tuple([0.5] * 18)
        geometry = MotiveGeometry(digest(list(context)), context, (0.5, 0.5), 0.0, region="MDN_WX",
                                  alpha=(1.0, 1.0), support_values=(0.6, 0.6), model_digest="new",
                                  model_path="m.pt", weight_owner="mdn")
        _, rejected, _ = make_coordinator(repo)._screen([a["good"]], geometry)
        assert dict(rejected) == {"lander.good": "MODEL_CHANGED"}

    def test_budget_spent_while_screening_stops_and_marks_unexamined_actions(self, pool):
        repo, a = pool
        repo.registry.delay = 0.2
        res = make_coordinator(repo, decision_budget_seconds=0.15).select(OBS, STIM, [a["good"], a["weak"]])
        assert res.selection is None and res.recovery.reason == "DECISION_BUDGET_EXHAUSTED"
        assert dict(res.rejected)["lander.weak"] == "DECISION_BUDGET_EXHAUSTED"

    @pytest.mark.parametrize("failure, expected", [("boundary", "IDENTITY_CONFLICT"), ("crash", "DECISION_FAILURE"),
                                                   ("exit", "DECISION_FAILURE")])
    def test_decision_worker_failures_reach_the_caller(self, pool, monkeypatch, failure, expected):
        repo, a = pool

        def broken(self, *args, **kwargs):
            if failure == "boundary":
                raise BoundaryError("IDENTITY_CONFLICT", "raised in the worker")
            if failure == "crash":
                raise RuntimeError("boom")
            os._exit(3)  # worker dies without sending anything

        monkeypatch.setattr(Coordinator, "_select_impl", broken)
        with pytest.raises(BoundaryError) as e:
            Coordinator(repo).select(OBS, STIM, [a["good"]])
        assert code(e) == expected

    def test_process_mode_matches_cooperative(self, pool):
        repo, a = pool
        res = Coordinator(repo).select(OBS, STIM, [a["good"], a["weak"]])
        assert res.selection.action_id == "lander.good"
        assert res.eligible_actions[0] is a["good"]  # caller's exact object is preserved

    def test_deadline_exhausted_returns_stop(self, pool):
        repo, a = pool
        repo.registry.delay = 1.0
        c = Coordinator(repo, decision_budget_seconds=0.2)
        t = time.monotonic()
        res = c.select(OBS, STIM, [a["good"]])
        assert time.monotonic() - t < 0.8
        assert res.selection is None and res.recovery.reason == "DECISION_BUDGET_EXHAUSTED"

    def test_trace_is_written_as_jsonl(self, pool, tmp_path):
        repo, a = pool
        trace = tmp_path / "t" / "trace.jsonl"
        make_coordinator(repo, trace_path=trace).select(OBS, STIM, [a["good"]])
        events = [json.loads(line)["event"] for line in trace.read_text().splitlines()]
        assert events == ["selection", "eligibility"]


class TestExecute:
    @pytest.fixture(autouse=True)
    def fake_rollout(self, monkeypatch):
        self.outcome = make_outcome()
        monkeypatch.setattr(coord_mod, "rollout", lambda *a, **k: self.outcome)

    def test_execute_applies_outcome_once(self, pool):
        repo, a = pool
        c = make_coordinator(repo)
        before = c.state_a.copy()
        sel = c.select(OBS, STIM, [a["good"]]).selection
        result, feedback = c.execute(sel, env=None, observation=OBS, seed=42)
        assert result["skill_id"] == "lander.good.v1" and result["mode"] == "normal"
        assert feedback == outcome_stimulus(self.outcome)
        assert not np.array_equal(before.G, c.state_a.G) or not np.array_equal(before.M, c.state_a.M)
        with pytest.raises(BoundaryError) as e:  # a decision executes once only
            c.execute(sel, None, OBS, 42)
        assert code(e) == "STALE_SNAPSHOT"

    def test_both_subsystem_states_advance(self, pool):
        repo, a = pool
        c = make_coordinator(repo)
        old = (c.state_a.copy(), c.state_b.copy())
        sel = c.select(OBS, STIM, [a["good"]]).selection
        c.execute(sel, None, OBS, 42)
        for new, before in zip((c.state_a, c.state_b), old):
            assert not (np.array_equal(new.G, before.G) and np.array_equal(new.M, before.M))

    def test_execute_rejects_changed_observation(self, pool):
        repo, a = pool
        c = make_coordinator(repo)
        sel = c.select(OBS, STIM, [a["good"]]).selection
        with pytest.raises(BoundaryError) as e:
            c.execute(sel, None, [0.1] + OBS[1:], 42)
        assert code(e) == "STALE_SNAPSHOT"

    def test_execute_rejects_changed_certificate(self, pool):
        repo, a = pool
        c = make_coordinator(repo)
        sel = c.select(OBS, STIM, [a["good"]]).selection
        repo.certs["lander.good.v1"] = FakeCert(2.0, (2.0, 1.5))
        with pytest.raises(BoundaryError) as e:
            c.execute(sel, None, OBS, 42)
        assert code(e) == "STALE_SNAPSHOT"

    def test_cancel_clears_pending(self, pool):
        repo, a = pool
        c = make_coordinator(repo)
        sel = c.select(OBS, STIM, [a["good"]]).selection
        c.cancel()
        with pytest.raises(BoundaryError):
            c.execute(sel, None, OBS, 42)

    def test_failed_rollout_is_traced_and_not_retryable(self, pool, tmp_path, monkeypatch):
        repo, a = pool
        trace = tmp_path / "trace.jsonl"
        c = make_coordinator(repo, trace_path=trace)
        sel = c.select(OBS, STIM, [a["good"]]).selection
        monkeypatch.setattr(coord_mod, "rollout", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            c.execute(sel, None, OBS, 42)
        assert any(json.loads(l)["event"] == "execution_failure" for l in trace.read_text().splitlines())
        with pytest.raises(BoundaryError):
            c.execute(sel, None, OBS, 42)

    def test_unexpected_crash_opens_review_and_can_suspend(self, pool):
        repo, a = pool
        c = make_coordinator(repo, suspend_unexpected=True)
        sel = c.select(OBS, STIM, [a["good"]]).selection
        self.outcome = make_outcome(crashed=True, success=False)
        c.execute(sel, None, OBS, 42)
        state = c.reviews.read()
        assert [e["reason"] for e in state["events"]] == ["UNEXPECTED_CRASH"]
        assert state["suspended"] == ["lander.good.v1"]
        # next cycle: the suspended skill is no longer eligible
        again = make_coordinator(repo).select(OBS, STIM, [a["good"]])
        assert dict(again.rejected) == {"lander.good": "SUSPENDED_FOR_REEVALUATION"}

    def test_crash_seen_in_admission_is_not_an_anomaly(self, pool):
        repo, a = pool
        repo.evidence["lander.good.v1"]["episodes"][0] = good_episode(42, crashed=True)
        c = make_coordinator(repo)
        sel = c.select(OBS, STIM, [a["good"]]).selection
        self.outcome = make_outcome(crashed=True, success=False)
        c.execute(sel, None, OBS, 42)
        assert c.reviews.read()["events"] == []


# -------------------------------------------------------------- recovery

class TestRecovery:
    def test_without_approval_file_nothing_applies(self, pool):
        repo, _ = pool
        rec, reason = RecoveryPolicy(repo).applicable(OBS, time.monotonic() + 5)
        assert rec is None and reason == "NO_APPLICABLE_BACKUP"

    def test_approve_requires_thresholds(self, pool, tmp_path):
        repo, _ = pool
        repo.evidence["lander.good.v1"]["episodes"] = [good_episode(s, crashed=True) for s in range(5)]
        with pytest.raises(BoundaryError) as e:
            approve_backup(repo, "lander.good.v1", tmp_path / "ok.json")
        assert code(e) == "BACKUP_NOT_APPROVED"

    def test_thresholds_must_be_rates(self, pool, tmp_path):
        repo, _ = pool
        for kw in ({"minimum_success_rate": 1.5}, {"maximum_crash_rate": -0.1}):
            with pytest.raises(ValueError):
                approve_backup(repo, "lander.good.v1", tmp_path / "x.json", **kw)

    def test_suspended_backup_is_not_offered(self, pool, tmp_path):
        repo, _ = pool
        path = tmp_path / "backup.json"
        approve_backup(repo, "lander.good.v1", path)
        (repo.root / "feedback.json").write_text(json.dumps({"events": [], "suspended": ["lander.good.v1"]}))
        rec, reason = RecoveryPolicy(repo, path).applicable(OBS, time.monotonic() + 5)
        assert rec is None and reason == "SUSPENDED_FOR_REEVALUATION"

    def test_unreadable_approval_is_not_approved(self, pool, tmp_path):
        repo, _ = pool
        path = tmp_path / "backup.json"
        path.write_text("{not json")
        assert RecoveryPolicy(repo, path).applicable(OBS, time.monotonic() + 5) == (None, "BACKUP_NOT_APPROVED")

    def test_backup_decision_goes_stale_if_the_observation_changes(self, pool, tmp_path, monkeypatch):
        repo, a = pool
        monkeypatch.setattr(coord_mod, "rollout", lambda *a, **k: pytest.fail("must not execute"))
        path = tmp_path / "backup.json"
        approve_backup(repo, "lander.good.v1", path)
        c = make_coordinator(repo, backup_approval=path)
        res = c.select(OBS, STIM, [a["weak"]])
        with pytest.raises(BoundaryError) as e:
            c.execute_recovery(res.recovery, None, [0.5] + OBS[1:], 42)
        assert code(e) == "STALE_SNAPSHOT"

    def test_backup_decision_is_refused_if_the_approval_disappears(self, pool, tmp_path, monkeypatch):
        repo, a = pool
        monkeypatch.setattr(coord_mod, "rollout", lambda *a, **k: pytest.fail("must not execute"))
        path = tmp_path / "backup.json"
        approve_backup(repo, "lander.good.v1", path)
        c = make_coordinator(repo, backup_approval=path)
        res = c.select(OBS, STIM, [a["weak"]])
        path.unlink()
        with pytest.raises(BoundaryError) as e:
            c.execute_recovery(res.recovery, None, OBS, 42)
        assert code(e) == "BACKUP_NOT_APPROVED"

    def test_approve_requires_five_episodes(self, pool, tmp_path):
        repo, _ = pool
        repo.evidence["lander.good.v1"]["episodes"] = repo.evidence["lander.good.v1"]["episodes"][:4]
        with pytest.raises(BoundaryError) as e:
            approve_backup(repo, "lander.good.v1", tmp_path / "ok.json")
        assert code(e) == "BACKUP_NOT_APPROVED"

    def test_approve_requires_global_cds_certificate(self, pool, tmp_path):
        repo, _ = pool
        repo.certs["lander.good.v1"] = FakeCert(region="MDN_WX")
        with pytest.raises(BoundaryError):
            approve_backup(repo, "lander.good.v1", tmp_path / "ok.json")

    def test_applicable_only_at_evaluated_start(self, pool, tmp_path):
        repo, _ = pool
        path = tmp_path / "backup.json"
        approve_backup(repo, "lander.good.v1", path)
        policy = RecoveryPolicy(repo, path)
        rec, reason = policy.applicable(OBS, time.monotonic() + 5)
        assert reason == "APPLICABLE_VALIDATED_BACKUP" and rec["approval"]["skill_id"] == "lander.good.v1"
        rec, reason = policy.applicable([0.5] + OBS[1:], time.monotonic() + 5)
        assert rec is None and reason == "BACKUP_NOT_APPLICABLE"

    def test_tampered_approval_is_rejected(self, pool, tmp_path):
        repo, _ = pool
        path = tmp_path / "backup.json"
        approve_backup(repo, "lander.good.v1", path)
        record = json.loads(path.read_text())
        record["approval"]["success_rate"] = 0.5
        path.write_text(json.dumps(record))
        rec, reason = RecoveryPolicy(repo, path).applicable(OBS, time.monotonic() + 5)
        assert rec is None and reason == "BACKUP_NOT_APPROVED"

    def test_expired_deadline(self, pool, tmp_path):
        repo, _ = pool
        path = tmp_path / "backup.json"
        approve_backup(repo, "lander.good.v1", path)
        rec, reason = RecoveryPolicy(repo, path).applicable(OBS, time.monotonic() - 1)
        assert rec is None and reason == "NO_APPLICABLE_BACKUP"

    def test_empty_set_falls_back_to_backup_and_executes(self, pool, tmp_path, monkeypatch):
        repo, a = pool
        path = tmp_path / "backup.json"
        approve_backup(repo, "lander.good.v1", path)
        monkeypatch.setattr(coord_mod, "rollout", lambda *a, **k: make_outcome())
        c = make_coordinator(repo, backup_approval=path)
        res = c.select(OBS, STIM, [a["weak"]])
        assert res.recovery.outcome == "backup" and res.recovery.backup_skill_id == "lander.good.v1"
        result, _ = c.execute_recovery(res.recovery, None, OBS, 42)
        assert result["mode"] == "backup" and result["certificate_digest"] is None
        with pytest.raises(BoundaryError):
            c.execute_recovery(res.recovery, None, OBS, 42)


# ------------------------------------------------------------- scheduler

class TestScheduler:
    def make(self, tmp_path, t, **cfg):
        return ReevaluationScheduler(tmp_path / "s.json", SchedulerConfig(**cfg), clock=lambda: t[0])

    def rejection(self, event_id="e1", deficit=2.0):
        return {"reason": "GATE_REJECTED", "skill_id": "lander.x.v1", "contract_id": "c", "scope": "FULL_SIMPLEX",
                "event_id": event_id, "artifact_digest": "old",
                "details": {"robust_score": -deficit, "epsilon": 0.0}}

    def test_config_validation(self):
        for kw in ({"half_life_seconds": 0}, {"fixed_increase": 0}, {"max_retries": 0}, {"cooldown_seconds": -1}):
            with pytest.raises(ValueError):
                SchedulerConfig(**kw)

    def test_observe_penalizes_gate_rejections_only_once_per_event(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t)
        key = s.observe(self.rejection())
        penalty = s.state["entries"][key]["penalty"]
        assert penalty == pytest.approx(1.0 + 0.1 * 2.0)
        assert s.observe(self.rejection()) == key and s.state["entries"][key]["penalty"] == penalty
        assert s.observe({**self.rejection("e2"), "reason": "REPAIR_FAILED"}) is None

    def test_penalty_halves_each_half_life(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t, half_life_seconds=100.0)
        entry = s.state["entries"][s.observe(self.rejection())]
        assert s.penalty(entry, at=1100.0) == pytest.approx(entry["penalty"] / 2)

    def test_state_survives_restart_and_config_change_is_refused(self, tmp_path):
        t = [1000.0]
        key = self.make(tmp_path, t).observe(self.rejection())
        assert key in self.make(tmp_path, t).state["entries"]
        with pytest.raises(BoundaryError) as e:
            self.make(tmp_path, t, max_retries=7)
        assert code(e) == "SCHEMA_MISMATCH"

    def test_request_needs_real_trigger_cooldown_and_lineage(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t, cooldown_seconds=60.0)
        key = s.observe(self.rejection())
        revised = make_candidate("x", artifact="new")
        revised = replace(revised, artifact_digest="new-digest")
        with pytest.raises(BoundaryError) as e:
            s.request(key, revised, "behavior_revision", "")
        assert code(e) == "NO_RECONSIDERATION_TRIGGER"
        with pytest.raises(BoundaryError) as e:  # same artifact, only the label changed
            s.request(key, replace(revised, artifact_digest="old"), "behavior_revision", "old")
        assert code(e) == "NO_RECONSIDERATION_TRIGGER"
        with pytest.raises(BoundaryError) as e:
            s.request(key, revised, "behavior_revision", "new-digest")
        assert code(e) == "COOLDOWN"
        t[0] += 61
        with pytest.raises(BoundaryError) as e:
            s.request(key, make_candidate("other"), "behavior_revision", "x")
        assert code(e) == "IDENTITY_CONFLICT"
        request = s.request(key, revised, "behavior_revision", "new-digest")
        assert request["remaining_budget"] == 3

    def test_attempt_lifecycle_and_budget(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t, cooldown_seconds=1.0, max_retries=1)
        key = s.observe(self.rejection())
        revised = replace(make_candidate("x"), artifact_digest="v2")
        t[0] += 2
        req = s.request(key, revised, "behavior_revision", "v2")
        assert s.next_due()["request_id"] == req["request_id"]
        s.begin(req)
        with pytest.raises(BoundaryError):  # one attempt in flight at a time
            s.begin(req)
        s.finish(req, admitted=False, reason="GATE_REJECTED")
        assert s.state["events"][-1]["event"] == "budget_exhausted"
        assert s.next_due() is None

    def test_interrupted_attempt_is_not_replayed_after_restart(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t, cooldown_seconds=1.0)
        key = s.observe(self.rejection())
        t[0] += 2
        req = s.request(key, replace(make_candidate("x"), artifact_digest="v2"), "behavior_revision", "v2")
        s.begin(req)
        s2 = self.make(tmp_path, t, cooldown_seconds=1.0)
        entry = s2.state["entries"][key]
        assert entry["in_flight"] is None and entry["attempts"] == 1
        assert any(e["event"] == "interrupted_attempt" for e in s2.state["events"])

    def test_context_change_only_reopens_contextual_rejections(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t, cooldown_seconds=1.0)
        candidate = make_candidate("x")
        global_key = s.observe(self.rejection())
        contextual = {**self.rejection("e9"), "scope": "MDN_WX",
                      "details": {"robust_score": -1.0, "epsilon": 0.0, "geometry": {"context_id": "ctx1"}}}
        contextual_key = s.observe(contextual)
        t[0] += 2
        with pytest.raises(BoundaryError) as e:  # pure motive changes never void global evidence
            s.request(global_key, candidate, "context_change", "ctx2", context_id="ctx2")
        assert code(e) == "NO_RECONSIDERATION_TRIGGER"
        with pytest.raises(BoundaryError):  # same context as the rejection
            s.request(contextual_key, candidate, "context_change", "ctx1", context_id="ctx1")
        assert s.request(contextual_key, candidate, "context_change", "ctx2", context_id="ctx2")["context_id"] == "ctx2"

    def test_second_queued_request_cannot_reuse_a_spent_attempt(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t, cooldown_seconds=1.0, max_retries=1)
        key = s.observe(self.rejection())
        t[0] += 2
        first = s.request(key, replace(make_candidate("x"), artifact_digest="v2"), "behavior_revision", "v2")
        second = s.request(key, replace(make_candidate("x"), artifact_digest="v3"), "behavior_revision", "v3")
        s.begin(first)
        s.finish(first, admitted=False)
        t[0] += 2
        with pytest.raises(BoundaryError) as e:
            s.begin(second)
        assert code(e) == "REEVALUATION_UNAVAILABLE"

    def test_dispatch_order_uses_the_penalty_after_decay(self, tmp_path):
        t = [0.0]
        s = self.make(tmp_path, t, half_life_seconds=100.0, cooldown_seconds=1.0)
        old_heavy = s.observe({**self.rejection("ea", deficit=10.0), "skill_id": "lander.a.v1"})  # penalty 2.0 at t=0
        t[0] = 300.0
        recent_light = s.observe({**self.rejection("eb", deficit=2.0), "skill_id": "lander.b.v1"})  # penalty 1.2 at t=300
        t[0] = 301.0
        for key, name in ((old_heavy, "a"), (recent_light, "b")):
            digest_ = "v-" + name
            s.request(key, replace(make_candidate(name), artifact_digest=digest_), "behavior_revision", digest_)
        assert s.next_due()["skill_id"] == "lander.a.v1"  # 2.0 decayed to ~0.25, 1.2 has barely decayed

    def test_nothing_due_means_nothing_to_dispatch(self, tmp_path):
        assert self.make(tmp_path, [1000.0]).next_due() is None

    def test_a_new_change_reopens_an_exhausted_budget_but_the_same_one_cannot(self, tmp_path):
        t = [1000.0]
        s = self.make(tmp_path, t, cooldown_seconds=1.0, max_retries=1)
        key = s.observe(self.rejection())
        t[0] += 2
        first = s.request(key, replace(make_candidate("x"), artifact_digest="v2"), "behavior_revision", "v2")
        s.begin(first)
        s.finish(first, admitted=False)
        t[0] += 2
        assert (s.state["entries"][key]["attempts"], s.state["entries"][key]["epoch"]) == (1, 0)
        with pytest.raises(BoundaryError) as e:
            s.request(key, replace(make_candidate("x"), artifact_digest="v2"), "behavior_revision", "v2")
        assert code(e) == "NO_RECONSIDERATION_TRIGGER"
        s.request(key, replace(make_candidate("x"), artifact_digest="v3"), "behavior_revision", "v3")
        assert (s.state["entries"][key]["attempts"], s.state["entries"][key]["epoch"]) == (0, 1)
        assert any(event["event"] == "budget_reopened" for event in s.state["events"])
