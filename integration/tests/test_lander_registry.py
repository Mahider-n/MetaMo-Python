"""Registry: artifact pinning (swap detection), the controlled policy variants, explicit policy bindings.

No simulator needed. Needs the SubRep submodule; the PPO policy tests also need torch
(skipped without it). Swaps are done on a temporary copy of SubRep's files, never on the submodule.
"""
from dataclasses import replace
import shutil

import numpy as np
import pytest

from core.state import Action, Stimulus
from integration.lander import registry as registry_mod
from integration.lander.contracts import BoundaryError
from integration.lander.registry import CandidateSkill, PolicyBinding, Registry, artifact_digest

if not (registry_mod.SUBREP / "models/pilot_ppo.pt").is_file():
    pytest.skip("SubRep submodule not checked out", allow_module_level=True)

FIXED = {"noop": 0, "left_engine": 1, "main_engine": 2, "right_engine": 3}
PPO = ("ppo_deterministic", "ppo_stochastic", "ppo_noisy", "ppo_then_side_tradeoff")
OBS = np.random.default_rng(0).uniform(-1, 1, (200, 8)).astype(np.float32)


def code(exc_info):
    return exc_info.value.code


def cand(loader, **kw):
    action = Action(f"lander.{loader}", np.full(8, 0.5), 0.2, np.zeros(8))
    return CandidateSkill(f"lander.{loader}.v1", action, loader, artifact_digest(loader), **kw)


def action_of(output):
    """SkillExecutor accepts an int or an (action, behavior_probability) tuple."""
    return output[0] if isinstance(output, tuple) else output


def run(policy, n=200):
    return [action_of(policy(o)) for o in OBS[:n]]


def append_byte(path):
    path.write_bytes(path.read_bytes() + b"\0")


@pytest.fixture
def subrep_copy(tmp_path, monkeypatch):
    """The files artifact_digest() reads, copied to tmp and swapped in for SUBREP."""
    for rel in ("env/lunar_lander_wrapper.py", "env/skill_executor.py", "pilot/rl_pilot.py", "models/pilot_ppo.pt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(registry_mod.SUBREP / rel, tmp_path / rel)
    monkeypatch.setattr(registry_mod, "SUBREP", tmp_path)
    return tmp_path


# ------------------------------------------------------ artifact identity

class TestArtifactIdentity:
    def test_digests_group_by_what_the_controller_depends_on(self):
        simple = {artifact_digest(loader) for loader in (*FIXED, "random")}
        pilot = {artifact_digest(loader) for loader in PPO}
        assert len(simple) == 1 and len(pilot) == 1 and simple != pilot

    def test_unknown_loader_is_refused(self):
        with pytest.raises(BoundaryError) as e:
            artifact_digest("mystery")
        assert code(e) == "UNRESOLVED_EXECUTABLE"

    def test_swapping_the_checkpoint_changes_only_pilot_backed_digests(self, subrep_copy):
        before = {loader: artifact_digest(loader) for loader in (*FIXED, "random", *PPO)}
        append_byte(subrep_copy / "models/pilot_ppo.pt")
        after = {loader: artifact_digest(loader) for loader in before}
        assert {l for l in before if before[l] != after[l]} == set(PPO)

    def test_changing_the_executor_changes_every_digest(self, subrep_copy):
        before = {loader: artifact_digest(loader) for loader in ("noop", "ppo_deterministic")}
        append_byte(subrep_copy / "env/skill_executor.py")
        assert all(artifact_digest(loader) != before[loader] for loader in before)

    def test_missing_artifact_is_refused(self, subrep_copy):
        (subrep_copy / "models/pilot_ppo.pt").unlink()
        assert artifact_digest("noop")
        with pytest.raises(BoundaryError) as e:
            artifact_digest("ppo_deterministic")
        assert code(e) == "UNRESOLVED_EXECUTABLE"


# ------------------------------------------------------ registry guards

class TestRegistryGuards:
    def test_construction_rejects_inconsistent_candidates(self):
        good = cand("noop")
        for bad, expected in ((replace(good, behavior_version="v2"), "IDENTITY_CONFLICT"),
                              (replace(good, environment="other-env"), "SCHEMA_MISMATCH"),
                              (replace(good, objective_schema="other"), "SCHEMA_MISMATCH"),
                              (replace(good, artifact_digest="0" * 64), "ARTIFACT_CHANGED")):
            with pytest.raises(BoundaryError) as e:
                Registry([bad])
            assert code(e) == expected
        with pytest.raises(BoundaryError) as e:
            Registry([good, good])
        assert code(e) == "IDENTITY_CONFLICT"

    def test_lookups(self):
        c = cand("noop")
        registry = Registry([c])
        assert registry.resolve(c.action) is c and registry.by_skill(c.skill_id) is c
        for call in (lambda: registry.resolve(cand("main_engine").action), lambda: registry.by_skill("lander.x.v1")):
            with pytest.raises(BoundaryError) as e:
                call()
            assert code(e) == "UNKNOWN_BEHAVIOR"
        with pytest.raises(BoundaryError) as e:  # same id, different descriptor
            registry.resolve(replace(c.action, risk_estimate=0.9))
        assert code(e) == "INVALID_ACTION"

    def test_swapped_checkpoint_is_refused_on_every_access_path(self, subrep_copy):
        idle, pilot = cand("noop"), cand("ppo_deterministic")
        registry = Registry([idle, pilot])
        append_byte(subrep_copy / "models/pilot_ppo.pt")
        for call in (lambda: registry.verify(pilot), lambda: registry.resolve(pilot.action),
                     lambda: registry.by_skill(pilot.skill_id), lambda: registry.policy(pilot, 1)):
            with pytest.raises(BoundaryError) as e:
                call()
            assert code(e) == "ARTIFACT_CHANGED"
        assert registry.resolve(idle.action) is idle  # unrelated controllers are unaffected

    def test_swap_invalidates_certified_entries_and_the_host_refuses_to_run_them(self, subrep_copy, tmp_path):
        pytest.importorskip("hyperon")
        pytest.importorskip("torch")
        from integration.lander.coordinator import Coordinator
        from integration.lander.storage import CertifiedRepository
        from .test_lander_repository import CONTRACT, evidence_for
        idle, pilot = cand("noop"), cand("ppo_deterministic")
        registry = Registry([idle, pilot])
        repo = CertifiedRepository(tmp_path / "repo", registry, CONTRACT)
        repo.set_baseline(evidence_for(idle, -100))
        assert repo.admit(pilot, evidence_for(pilot, 100, 0.2))["admitted"]
        append_byte(subrep_copy / "models/pilot_ppo.pt")
        with pytest.raises(BoundaryError) as e:
            repo.verify_entry(pilot.skill_id)
        assert code(e) == "ARTIFACT_CHANGED"
        with pytest.raises(BoundaryError) as e:
            CertifiedRepository(repo.root, registry, CONTRACT).load()
        assert code(e) == "ARTIFACT_CHANGED"
        result = Coordinator(repo, deadline_mode="cooperative").select([0, 1.4, 0, 0, 0, 0, 0, 0], Stimulus(.2, .5, .5, .2), [pilot.action])
        assert result.selection is None and dict(result.rejected) == {pilot.action.id: "ARTIFACT_CHANGED"}


# --------------------------------------------------------- policy variants

class TestPolicies:
    @pytest.fixture(scope="class")
    def registry(self):
        pytest.importorskip("torch")
        return Registry([cand(loader) for loader in (*FIXED, "random", *PPO)])

    def policy(self, registry, loader, seed=1):
        return registry.policy(registry.by_skill(f"lander.{loader}.v1"), seed)

    def test_fixed_controllers_always_return_their_action(self, registry):
        for loader, action in FIXED.items():
            assert set(run(self.policy(registry, loader), 20)) == {action}

    def test_random_is_seeded(self, registry):
        a, b = run(self.policy(registry, "random", 1)), run(self.policy(registry, "random", 1))
        assert a == b and set(a) <= {0, 1, 2, 3} and a != run(self.policy(registry, "random", 2))

    def test_pilot_reports_a_valid_behavior_probability(self, registry):
        action, probability = self.policy(registry, "ppo_deterministic")(OBS[0])
        assert action in (0, 1, 2, 3) and 0 < probability <= 1

    def test_pilot_is_deterministic_and_ignores_the_seed(self, registry):
        pilot = run(self.policy(registry, "ppo_deterministic", 1))
        assert set(pilot) <= {0, 1, 2, 3} and len(set(pilot)) > 1
        assert pilot == run(self.policy(registry, "ppo_deterministic", 1)) == run(self.policy(registry, "ppo_deterministic", 99))

    def test_side_tradeoff_follows_the_pilot_then_fires_the_side_engine(self, registry):
        pilot = run(self.policy(registry, "ppo_deterministic"))
        tradeoff = run(self.policy(registry, "ppo_then_side_tradeoff"))
        assert any(a != 1 for a in pilot[29:])  # otherwise the override would be invisible
        assert tradeoff[:29] == pilot[:29] and set(tradeoff[29:]) == {1}

    def test_noisy_pilot_is_seeded_and_really_deviates(self, registry):
        pilot = run(self.policy(registry, "ppo_deterministic"))
        noisy = run(self.policy(registry, "ppo_noisy", 1))
        assert noisy == run(self.policy(registry, "ppo_noisy", 1)) != run(self.policy(registry, "ppo_noisy", 2))
        assert 0 < sum(a != b for a, b in zip(noisy, pilot)) < 0.5 * len(pilot)  # ~15% random actions

    def test_stochastic_pilot_is_seeded(self, registry):
        assert run(self.policy(registry, "ppo_stochastic", 1)) == run(self.policy(registry, "ppo_stochastic", 1))

    def test_every_request_gets_a_fresh_controller(self, registry):
        used = self.policy(registry, "ppo_then_side_tradeoff")
        run(used)  # step counter now far past the override
        assert run(self.policy(registry, "ppo_then_side_tradeoff"), 29) == run(self.policy(registry, "ppo_deterministic"), 29)


# ------------------------------------------------------- explicit bindings

def make_factory(action):
    def factory(seed):
        return lambda obs: action
    return factory


def make_unserializable_factory():
    thing = object()
    def factory(seed):
        return thing
    return factory


def make_non_callable_factory():
    def factory(seed):
        return 5
    return factory


class TestPolicyBinding:
    @pytest.fixture
    def weights(self, tmp_path):
        path = tmp_path / "weights.bin"
        path.write_bytes(b"w")
        return path

    def candidate(self, binding):
        action = Action("lander.custom", np.full(8, 0.5), 0.2, np.zeros(8))
        return CandidateSkill("lander.custom.v1", action, "custom", binding.digest(), provider=binding.provider_version)

    def test_digest_pins_config_artifact_and_provider_version(self, weights):
        base = PolicyBinding(make_factory(2), (str(weights),), "p1")
        assert base.digest() == PolicyBinding(make_factory(2), (str(weights),), "p1").digest()
        assert PolicyBinding(make_factory(3), (str(weights),), "p1").digest() != base.digest()
        assert PolicyBinding(make_factory(2), (str(weights),), "p2").digest() != base.digest()
        before = base.digest()
        append_byte(weights)
        assert base.digest() != before

    def test_artifacts_and_version_are_required(self, weights):
        for binding in (PolicyBinding(make_factory(2), (), "p1"), PolicyBinding(make_factory(2), (str(weights),), "")):
            with pytest.raises(BoundaryError) as e:
                binding.digest()
            assert code(e) == "UNRESOLVED_EXECUTABLE"

    def test_unserializable_factory_state_is_refused(self, weights):
        with pytest.raises(BoundaryError) as e:
            PolicyBinding(make_unserializable_factory(), (str(weights),), "p1").digest()
        assert code(e) == "UNRESOLVED_EXECUTABLE"

    def test_registry_runs_the_bound_factory_and_detects_artifact_change(self, weights):
        binding = PolicyBinding(make_factory(2), (str(weights),), "p1")
        c = self.candidate(binding)
        registry = Registry([c], {"custom": binding})
        assert registry.policy(c, 7)(None) == 2
        append_byte(weights)
        with pytest.raises(BoundaryError) as e:
            registry.policy(c, 7)
        assert code(e) == "ARTIFACT_CHANGED"

    def test_provider_version_must_match_the_binding(self, weights):
        binding = PolicyBinding(make_factory(2), (str(weights),), "p1")
        with pytest.raises(BoundaryError) as e:
            Registry([replace(self.candidate(binding), provider="someone_else")], {"custom": binding})
        assert code(e) == "SCHEMA_MISMATCH"

    def test_factory_must_return_a_controller(self, weights):
        binding = PolicyBinding(make_non_callable_factory(), (str(weights),), "p1")
        c = self.candidate(binding)
        with pytest.raises(BoundaryError) as e:
            Registry([c], {"custom": binding}).policy(c, 1)
        assert code(e) == "UNRESOLVED_EXECUTABLE"
