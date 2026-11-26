"""Evidence boundary and SubRep binding.

Everything the certificate maths trusts goes through validate_evidence(), so each way
evidence can lie gets its own test. No simulator, torch or hyperon needed.
"""
import sys
import types

import numpy as np
import pytest

from integration.lander import evidence as evidence_mod
from integration.lander import vendor
from integration.lander.contracts import BoundaryError
from integration.lander.evidence import ExecutionCancelled, ObservedEnvironment, matched_improvements, validate_evidence
from dataclasses import asdict

from core.state import Action
from integration.lander.contracts import MeasurementContract
from integration.lander.evidence import summarize
from integration.lander.registry import CandidateSkill, artifact_digest

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


if not (vendor.SUBREP / "certification/certificate_schema.py").is_file():
    pytest.skip("SubRep submodule not checked out", allow_module_level=True)


def code(exc_info):
    return exc_info.value.code


@pytest.fixture
def good():
    return evidence_for(cand("main_engine"), 100, 0.2)


def first(evidence):
    return evidence["episodes"][0]


# --------------------------------------------------------- validate_evidence

TAMPERING = {
    "other contract": (lambda e: e["contract"].update(horizon=100), "SCHEMA_MISMATCH"),
    "different reset seeds": (lambda e: first(e).update(seed=999), "SCHEMA_MISMATCH"),
    "short observation": (lambda e: first(e).update(initial_observation=[0.0] * 3), "INVALID_EVIDENCE"),
    "step count disagrees with rewards": (lambda e: first(e).update(steps=3), "INVALID_EVIDENCE"),
    "non finite reward": (lambda e: first(e)["raw_rewards"][0].__setitem__(0, float("nan")), "INVALID_EVIDENCE"),
    "positive engine cost": (lambda e: first(e)["raw_rewards"][0].__setitem__(2, 0.5), "SCHEMA_MISMATCH"),
    "flag is not a bool": (lambda e: first(e).update(crashed=0), "INVALID_EVIDENCE"),
    "wrong stop reason": (lambda e: first(e).update(stop_reason="max_steps"), "INVALID_EVIDENCE"),
    "unfinished episode shorter than horizon": (lambda e: first(e).update(
        terminated=False, stop_reason="max_steps", landing_success=False, crashed=False), "INVALID_EVIDENCE"),
    "success flag disagrees with terminal reward": (lambda e: first(e).update(landing_success=False), "INVALID_EVIDENCE"),
    "crash flag disagrees with terminal reward": (lambda e: first(e).update(crashed=True), "INVALID_EVIDENCE"),
    "episode return disagrees with rewards": (lambda e: first(e).update(payoff=first(e)["payoff"] + 5), "INVALID_EVIDENCE"),
    "objectives disagree with rewards": (lambda e: first(e).update(objectives=[0.0, 0.0]), "INVALID_EVIDENCE"),
    "aggregate disagrees with episodes": (lambda e: e.update(mean_payoff=e["mean_payoff"] + 1), "INVALID_EVIDENCE"),
}


def test_valid_evidence_passes(good):
    validate_evidence(good, CONTRACT)


@pytest.mark.parametrize("name", TAMPERING)
def test_each_kind_of_bad_evidence_is_refused(good, name):
    tamper, expected = TAMPERING[name]
    tamper(good)
    with pytest.raises(BoundaryError) as e:
        validate_evidence(good, CONTRACT)
    assert code(e) == expected


def test_episode_longer_than_the_horizon_is_refused():
    with pytest.raises(BoundaryError) as e:
        validate_evidence(evidence_for(cand("main_engine"), 100, 0.2, steps=CONTRACT.horizon + 1), CONTRACT)
    assert code(e) == "INVALID_EVIDENCE"


def test_running_to_the_horizon_without_terminating_is_valid():
    evidence = evidence_for(cand("main_engine"), 100, 0.2, steps=CONTRACT.horizon)
    for episode in evidence["episodes"]:
        episode.update(terminated=False, stop_reason="max_steps", landing_success=False, crashed=False)
    validate_evidence(evidence, CONTRACT)


# ---------------------------------------------------- matched_improvements

def test_improvements_are_differences_of_the_matched_means(good):
    base = evidence_for(cand("noop"), -100)
    dr, dn = matched_improvements(base, good, CONTRACT)
    assert dr == pytest.approx(good["mean_payoff"] - base["mean_payoff"])
    assert np.allclose(dn, np.array(good["mean_objectives"]) - np.array(base["mean_objectives"]), atol=1e-4)


def test_baseline_and_candidate_must_start_from_the_same_observation(good):
    base = evidence_for(cand("noop"), -100)
    first(good)["initial_observation"][0] += 0.5
    with pytest.raises(BoundaryError) as e:
        matched_improvements(base, good, CONTRACT)
    assert code(e) == "SCHEMA_MISMATCH"


def test_invalid_evidence_never_reaches_the_comparison(good):
    base = evidence_for(cand("noop"), -100)
    good["mean_payoff"] += 1
    with pytest.raises(BoundaryError) as e:
        matched_improvements(base, good, CONTRACT)
    assert code(e) == "INVALID_EVIDENCE"


# ------------------------------------------------------ ObservedEnvironment

class FakeEnv:
    def __init__(self, reward=(1.0, -0.2), raw=(1.0, 0.0, -0.1, -0.1)):
        self.reward, self.raw, self.steps = np.array(reward), raw, 0

    def step(self, action):
        self.steps += 1
        return np.zeros(8), self.reward, False, False, {"raw_rewards": self.raw, "landing_success": False}


class TestObservedEnvironment:
    def test_records_raw_rewards_and_passes_results_through(self):
        env = ObservedEnvironment(FakeEnv())
        result = env.step(0)
        assert env.raw_rewards == [[1.0, 0.0, -0.1, -0.1]] and result[1].tolist() == [1.0, -0.2]
        assert env.last_info["landing_success"] is False

    def test_mapped_reward_must_match_raw_components(self):
        with pytest.raises(BoundaryError) as e:
            ObservedEnvironment(FakeEnv(reward=(9.0, -0.2))).step(0)
        assert code(e) == "SCHEMA_MISMATCH"

    def test_engine_cost_must_stay_non_positive(self):
        with pytest.raises(BoundaryError) as e:
            ObservedEnvironment(FakeEnv(reward=(1.0, 0.0), raw=(1.0, 0.0, 0.1, -0.1))).step(0)
        assert code(e) == "SCHEMA_MISMATCH"

    def test_malformed_raw_rewards_are_refused(self):
        with pytest.raises(BoundaryError) as e:
            ObservedEnvironment(FakeEnv(raw=(1.0, 0.0, -0.1))).step(0)
        assert code(e) == "INVALID_EVIDENCE"

    def test_cancellation_stops_before_the_next_simulator_step(self):
        inner = FakeEnv()
        env = ObservedEnvironment(inner, cancelled=lambda: inner.steps >= 2)
        env.step(0)
        env.step(0)
        with pytest.raises(ExecutionCancelled):
            env.step(0)
        assert inner.steps == 2


def test_wrong_simulator_version_is_refused(monkeypatch):
    monkeypatch.setattr(evidence_mod, "version", lambda package: "0.0.0")
    with pytest.raises(BoundaryError) as e:
        evidence_mod.make_environment(CONTRACT)
    assert code(e) == "SCHEMA_MISMATCH"
    assert "mo-gymnasium" in str(e.value)


# ------------------------------------------------------------ SubRep binding

class TestVendor:
    def test_source_revision_is_a_git_commit(self):
        for path in (vendor.SUBREP, vendor.SUBREP.parents[1]):
            assert len(vendor.source_revision(path)) == 40

    def test_missing_submodule_gives_an_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vendor, "SUBREP", tmp_path)
        with pytest.raises(RuntimeError, match="submodule update"):
            vendor.enable_subrep()

    def test_a_different_installed_package_with_the_same_name_is_refused(self, monkeypatch):
        impostor = types.ModuleType("utils")
        impostor.__file__ = "/somewhere/else/utils.py"
        monkeypatch.setitem(sys.modules, "utils", impostor)
        with pytest.raises(RuntimeError, match="collision"):
            vendor.enable_subrep()

    def test_enabling_twice_adds_the_path_once(self):
        vendor.enable_subrep()
        vendor.enable_subrep()
        assert sys.path.count(str(vendor.SUBREP)) == 1

