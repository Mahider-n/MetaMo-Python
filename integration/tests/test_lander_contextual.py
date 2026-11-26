"""MDN contextual path: checkpoint approval, pinning, contextual certification, coordinator use.

The MDN is trained on small synthetic rows (same schema as experiments.training_rows),
so this runs in the fast default suite without the simulator. Needs hyperon and torch;
the module skips itself otherwise.
"""
import json
from dataclasses import replace
import shutil

import numpy as np
import pytest

for package in ("hyperon", "torch"):
    pytest.importorskip(package)
import torch

from core.state import MotivationalState, Stimulus
from integration.lander import contextual
from integration.lander import coordinator as coord_mod
from integration.lander.contextual import ContextualModel, train_checkpoint
from integration.lander.contracts import BoundaryError
from integration.lander.coordinator import Coordinator
from integration.lander.experiments import MagusDecision, MetaMoPseudoBimonad, OpenPsiAppraisal
from integration.lander.motives import project
from integration.lander.registry import Registry
from integration.lander.storage import CertifiedRepository
from .test_lander_repository import CONTRACT, cand, evidence_for

STIM = Stimulus(0.2, 0.5, 0.5, 0.2)
CENTER = MotivationalState(np.full(8, 0.5), np.full(6, 0.5))
OBS = [0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TRAIN, VALIDATION = tuple(range(6)), (6, 7)


def code(exc_info):
    return exc_info.value.code


def synthetic_rows(semantics=None):
    """Contexts built exactly like training_rows(); candidates are far from every gate boundary."""
    rng = np.random.default_rng(17)
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
    candidates = [{"skill_id": "lander.good.v1", "delta_r": 5.0, "delta_n": [5.0, 5.0]},
                  {"skill_id": "lander.bad.v1", "delta_r": -50.0, "delta_n": [-50.0, -50.0]}]
    rows = []
    for seed in TRAIN + VALIDATION:
        for _ in range(16):
            stimulus = Stimulus(float(rng.uniform()), 0.5, 0.3, 0.2)
            a = MotivationalState(rng.uniform(0.1, 0.9, 8), rng.uniform(0.1, 0.9, 6))
            b = MotivationalState(rng.uniform(0.1, 0.9, 8), rng.uniform(0.1, 0.9, 6))
            ca, cb = bimonad.decision_context(a, stimulus), bimonad.decision_context(b, stimulus)
            geo = project(bimonad.parallel_merge(ca, cb), stimulus)
            rows.append({"seed": seed, "context": list(geo.context), "weights": list(geo.weights),
                         "support_values": np.minimum(1, np.array(geo.weights) + 0.15).tolist(),
                         "semantics": semantics or contextual.semantics(CONTRACT), "evidence_digest": "synthetic",
                         "candidates": candidates, "context_a": [*ca.G.tolist(), *ca.M.tolist()],
                         "context_b": [*cb.G.tolist(), *cb.M.tolist()]})
    return rows


ROWS = synthetic_rows()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    path = tmp_path_factory.mktemp("mdn") / "master.pt"
    return path, train_checkpoint(ROWS, CONTRACT, path, TRAIN, VALIDATION, epochs=200)


@pytest.fixture
def checkpoint(trained, tmp_path):
    path = tmp_path / "mdn.pt"
    shutil.copy(trained[0], path)
    return path


def swap_checkpoint(path):
    """A different but valid, approved checkpoint at the same path."""
    train_checkpoint(ROWS, CONTRACT, path, TRAIN, VALIDATION, epochs=200, seed=18)


@pytest.fixture
def world(tmp_path, checkpoint):
    c = {"noop": cand("noop"), "good": cand("main_engine"), "bad": cand("left_engine")}
    repo = CertifiedRepository(tmp_path / "repo", Registry(list(c.values())), CONTRACT)
    repo.set_baseline(evidence_for(c["noop"], -100))
    model = ContextualModel(checkpoint, CONTRACT)
    geometry = model.geometry(project(CENTER, STIM, 5.0))
    return type("World", (), dict(c=c, repo=repo, model=model, geometry=geometry, checkpoint=checkpoint,
                                  good_evidence=evidence_for(c["good"], 100, 0.2),
                                  bad_evidence=evidence_for(c["bad"], -100, 10)))


# ------------------------------------------------------------- training

class TestTraining:
    def test_checkpoint_passes_held_out_acceptance(self, trained):
        metadata = trained[1]
        assert metadata["approved"] and contextual.accepted(metadata["validation"])
        assert not set(metadata["train_seeds"]) & set(metadata["validation_seeds"])

    @pytest.mark.parametrize("train, validation, epochs", [
        (TRAIN, (5, 6), 10), (TRAIN, (), 10), ((), VALIDATION, 10), (TRAIN, VALIDATION, 0)])
    def test_bad_partitions_or_epochs_are_refused(self, tmp_path, train, validation, epochs):
        with pytest.raises(BoundaryError) as e:
            train_checkpoint(ROWS, CONTRACT, tmp_path / "m.pt", train, validation, epochs)
        assert code(e) == "INVALID_DATASET"

    def test_rows_measured_under_other_semantics_are_refused(self, tmp_path):
        other = contextual.semantics(replace(CONTRACT, gamma=0.9))
        with pytest.raises(BoundaryError) as e:
            train_checkpoint(synthetic_rows(other), CONTRACT, tmp_path / "m.pt", TRAIN, VALIDATION, epochs=10)
        assert code(e) == "SCHEMA_MISMATCH"


# ---------------------------------------------------------------- model

class TestModel:
    def test_constructor_needs_explicit_owner_and_fallback(self, checkpoint):
        for kw in ({"weight_owner": "x"}, {"fallback": "x"}):
            with pytest.raises(ValueError):
                ContextualModel(checkpoint, CONTRACT, **kw)

    def test_unapproved_checkpoint_is_refused(self, checkpoint):
        data = torch.load(checkpoint, weights_only=True)
        data["metadata"]["approved"] = False
        torch.save(data, checkpoint)
        with pytest.raises(BoundaryError) as e:
            ContextualModel(checkpoint, CONTRACT)
        assert code(e) == "MODEL_NOT_APPROVED"

    def test_missing_checkpoint_errors_or_falls_back_explicitly(self, tmp_path):
        with pytest.raises(BoundaryError) as e:
            ContextualModel(tmp_path / "none.pt", CONTRACT)
        assert code(e) == "MODEL_NOT_APPROVED"
        fallback = ContextualModel(tmp_path / "none.pt", CONTRACT, fallback="global").geometry(project(CENTER, STIM))
        assert fallback.region == "FULL_SIMPLEX" and fallback.geometry_source == "explicit_global_fallback"

    def test_geometry_is_feasible_contextual_and_pinned(self, world):
        g = world.geometry
        assert g.region == "MDN_WX" and g.model_digest == world.model.sha
        assert all(0 <= s <= 1 for s in g.support_values) and sum(g.support_values) >= 1
        assert all(w <= s + 1e-9 for w, s in zip(g.weights, g.support_values))
        assert g.weights == project(CENTER, STIM, 5.0).weights  # owner "metamo" keeps MetaMo's weights

    def test_mdn_owner_takes_weights_from_alpha(self, world):
        g = ContextualModel(world.checkpoint, CONTRACT, weight_owner="mdn").geometry(project(CENTER, STIM))
        alpha = np.array(g.alpha)
        assert g.weights == pytest.approx(tuple(alpha / alpha.sum())) and sum(g.weights) == pytest.approx(1.0)

    def test_infeasible_support_is_refused_and_counted(self, checkpoint, monkeypatch):
        model = ContextualModel(checkpoint, CONTRACT)
        monkeypatch.setattr(model.model, "forward_inference", lambda x: (torch.ones(2), torch.tensor([0.1, 0.1])))
        with pytest.raises(BoundaryError) as e:
            model.geometry(project(CENTER, STIM))
        assert code(e) == "INVALID_GEOMETRY" and model.invalid_support_events == 1

    def test_replaced_checkpoint_is_detected(self, world):
        swap_checkpoint(world.checkpoint)
        with pytest.raises(BoundaryError) as e:
            world.model.geometry(project(CENTER, STIM))
        assert code(e) == "MODEL_CHANGED"

    def test_replaced_checkpoint_with_global_fallback(self, checkpoint):
        model = ContextualModel(checkpoint, CONTRACT, fallback="global")
        swap_checkpoint(checkpoint)
        assert model.geometry(project(CENTER, STIM)).region == "FULL_SIMPLEX" and model.failures == 1


# --------------------------------------- contextual certification + host

class TestContextualCertification:
    def test_admission_records_the_audit_geometry(self, world):
        result = world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        cert = result["certificate"]
        assert result["admitted"] and cert["weight_region_type"] == "MDN_WX"
        assert tuple(cert["certification_context"]) == world.geometry.context
        assert tuple(cert["mdn_alpha"]) == world.geometry.alpha
        assert tuple(cert["wx_support_values"]) == world.geometry.support_values
        assert world.repo.manifests["lander.main_engine.v1"]["geometry"]["model_digest"] == world.model.sha

    def test_contextual_certificate_needs_its_geometry_manifest(self, world):
        world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        del world.repo.manifests["lander.main_engine.v1"]["geometry"]
        with pytest.raises(BoundaryError) as e:
            world.repo.verify_entry("lander.main_engine.v1")
        assert code(e) == "SCHEMA_MISMATCH"

    def test_audit_geometry_must_match_what_the_checkpoint_infers(self, world):
        world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        record = world.repo.manifests["lander.main_engine.v1"]["geometry"]
        record["alpha"] = [float(a) + 1.0 for a in record["alpha"]]
        with pytest.raises(BoundaryError) as e:
            world.repo.verify_entry("lander.main_engine.v1")
        assert code(e) == "INVALID_GEOMETRY"

    def test_contextual_certificate_survives_save_and_load(self, world):
        world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        restored = CertifiedRepository(world.repo.root, world.repo.registry, CONTRACT)
        assert restored.load() == ["lander.main_engine.v1"]

    def test_rejection_is_scoped_to_the_context(self, world):
        assert not world.repo.admit(world.c["bad"], world.bad_evidence, 5.0, world.geometry)["admitted"]
        record = json.loads((world.repo.root / "rejections.json").read_text())[0]
        assert record["scope"] == "MDN_WX" and record["details"]["geometry"]["context_id"] == world.geometry.context_id

    def test_only_a_contextual_host_can_use_the_certificate(self, world):
        world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        action = world.c["good"].action
        blind = Coordinator(world.repo, deadline_mode="cooperative").select(OBS, STIM, [action])
        assert dict(blind.rejected) == {"lander.main_engine": "NOT_CERTIFIED"}
        aware = Coordinator(world.repo, contextual_model=world.model, deadline_mode="cooperative").select(OBS, STIM, [action])
        assert aware.selection.skill_id == "lander.main_engine.v1" and aware.selection.geometry.region == "MDN_WX"

    def test_swapped_checkpoint_invalidates_the_stored_certificate(self, world):
        world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        swap_checkpoint(world.checkpoint)
        with pytest.raises(BoundaryError) as e:
            world.repo.verify_entry("lander.main_engine.v1")
        assert code(e) == "MODEL_CHANGED"

    def test_host_stops_when_checkpoint_changes_between_cycles(self, world):
        world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        host = Coordinator(world.repo, contextual_model=world.model, deadline_mode="cooperative")
        swap_checkpoint(world.checkpoint)
        result = host.select(OBS, STIM, [world.c["good"].action])
        assert result.selection is None and result.recovery.reason == "MODEL_CHANGED"

    def test_decision_is_stale_if_checkpoint_changes_before_execution(self, world, monkeypatch):
        monkeypatch.setattr(coord_mod, "rollout", lambda *a, **k: pytest.fail("must not execute"))
        world.repo.admit(world.c["good"], world.good_evidence, 5.0, world.geometry)
        host = Coordinator(world.repo, contextual_model=world.model, deadline_mode="cooperative")
        selection = host.select(OBS, STIM, [world.c["good"].action]).selection
        swap_checkpoint(world.checkpoint)
        with pytest.raises(BoundaryError) as e:
            host.execute(selection, None, OBS, 42)
        assert code(e) == "STALE_SNAPSHOT" and "checkpoint changed" in str(e.value)  # the precise reason, not a later check
