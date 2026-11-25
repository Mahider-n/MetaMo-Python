"""Train and activate SubRep's SASP MDN with explicit held-out acceptance.

Targets distill the declared motive mapping and an evidence-dispersion support
rule. This is learned contextual geometry, not a learned policy/outcome model.
"""
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from .contracts import BoundaryError, digest, file_digest, vector
from .motives import CONTEXT_SCHEMA, MAPPING_VERSION
from .vendor import SUBREP, enable_subrep

enable_subrep()
from generator.mdn import MotiveDecompositionNetwork
from utils.support_geometry import worst_case_over_support_region

MODEL_VERSION = "metamo_sasp_geometry_v2"
THRESHOLDS = {"weight_mse_max": .01, "support_mse_max": .02, "coverage_min": .95,
              "gate_agreement_min": .9, "selection_agreement_min": .9}


def semantics(contract):
    record = asdict(contract)
    record.pop("seeds")  # Held-out seeds must differ; measurement meanings must not.
    return digest(record)


def evaluate_model(model, rows):
    x = torch.tensor([r["context"] for r in rows], dtype=torch.float32)
    target_w = np.array([r["weights"] for r in rows])
    target_s = np.array([r["support_values"] for r in rows])
    with torch.no_grad():
        alpha, support = model.forward_inference(x)
        w = (alpha / alpha.sum(-1, keepdim=True)).numpy()
        s = support.numpy()
    covered = np.all(target_w <= s + 1e-9, axis=1)
    feasible = np.isfinite(s).all(1) & (s >= 0).all(1) & (s <= 1).all(1) & (s.sum(1) >= 1 - 1e-9)
    agreements, selections = [], []
    per_gate = {"CDS": [], "PDS_5": []}
    per_selector = {"subrep_selects": [], "subrep_filter_then_magus": []}
    from core.state import Action, MotivationalState
    from magus.decision import MagusDecision
    magus = MagusDecision()
    for row, predicted_s, expected_s, ok in zip(rows, s, target_s, feasible):
        for gate_name, epsilon in (("CDS", 0.), ("PDS_5", 5.)):
            predicted, expected = [], []
            for candidate in row["candidates"]:
                dr, dn = candidate["delta_r"], candidate["delta_n"]
                target_pass = dr + worst_case_over_support_region(dn, expected_s) >= -epsilon
                pred_pass = bool(ok and dr + worst_case_over_support_region(dn, predicted_s) >= -epsilon)
                agreements.append(pred_pass == target_pass)
                per_gate[gate_name].append(pred_pass == target_pass)
                if target_pass:
                    expected.append(candidate)
                if pred_pass:
                    predicted.append(candidate)
            for selector in per_selector:
                def best(candidates):
                    if not candidates:
                        return None
                    def score(c):
                        if selector == "subrep_selects":
                            return c["delta_r"] + np.dot(row["weights"], c["delta_n"])
                        description = c.get("action", {"id": c["skill_id"], "goal_correlations": [0.] * 8,
                                                        "risk_estimate": 0., "delta_g": [0.] * 8})
                        action = Action(description["id"], np.array(description["goal_correlations"]),
                                        description["risk_estimate"], np.array(description["delta_g"]))
                        scores = []
                        for context in (row.get("context_a", row["context"][:14]), row.get("context_b", row["context"][:14])):
                            scores.append(magus.score_candidate(MotivationalState(np.array(context[:8]), np.array(context[8:14])), action))
                        return sum(scores) / 2 - .25 * abs(scores[0] - scores[1])
                    # MAGUS keeps original order; SubRep uses lexical ties.
                    return (sorted(candidates, key=lambda c: (-score(c), c["skill_id"]))[0] if selector == "subrep_selects"
                            else max(candidates, key=score))["skill_id"]
                same = best(predicted) == best(expected)
                selections.append(same)
                per_selector[selector].append(same)
    return {"contexts": len(rows), "gate_comparisons": len(agreements),
            "weight_mse": float(np.mean((w - target_w) ** 2)),
            "support_mse": float(np.mean((s - target_s) ** 2)),
            "weight_coverage": float(covered.mean()), "invalid_support_events": int((~feasible).sum()),
            "gate_agreement": float(np.mean(agreements)), "selection_agreement": float(np.mean(selections)),
            "per_gate_agreement": {k: float(np.mean(v)) for k, v in per_gate.items()},
            "per_selector_agreement": {k: float(np.mean(v)) for k, v in per_selector.items()},
            "validation_weight_owner": "metamo"}


def accepted(metrics):
    return (metrics["contexts"] > 0 and metrics["gate_comparisons"] > 0
            and np.isfinite([metrics[k] for k in ("weight_mse", "support_mse", "weight_coverage", "gate_agreement", "selection_agreement")]).all()
            and all(v >= .9 for v in metrics["per_gate_agreement"].values())
            and all(v >= .9 for v in metrics["per_selector_agreement"].values())
            and metrics["weight_mse"] <= THRESHOLDS["weight_mse_max"]
            and metrics["support_mse"] <= THRESHOLDS["support_mse_max"]
            and metrics["weight_coverage"] >= THRESHOLDS["coverage_min"]
            and metrics["gate_agreement"] >= THRESHOLDS["gate_agreement_min"]
            and metrics["selection_agreement"] >= THRESHOLDS["selection_agreement_min"]
            and metrics["invalid_support_events"] == 0)


def train_checkpoint(rows, contract, path, train_seeds, validation_seeds, epochs=600, seed=17):
    if set(train_seeds) & set(validation_seeds) or not train_seeds or not validation_seeds:
        raise BoundaryError("INVALID_DATASET", "nonempty disjoint seed partitions required")
    train = [r for r in rows if r["seed"] in train_seeds]
    heldout = [r for r in rows if r["seed"] in validation_seeds]
    if not train or not heldout or epochs <= 0:
        raise BoundaryError("INVALID_DATASET", "training and held-out contexts required")
    for row in rows:
        vector(row["context"], 18, unit=True)
        if row["semantics"] != semantics(contract) or not row.get("evidence_digest"):
            raise BoundaryError("SCHEMA_MISMATCH", "training evidence semantics differ")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    config = {"input_dim": 18, "num_objectives": 2, "hidden_dim": 32, "num_hidden_layers": 2}
    model = MotiveDecompositionNetwork(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    x = torch.tensor([r["context"] for r in train], dtype=torch.float32)
    w = torch.tensor([r["weights"] for r in train], dtype=torch.float32)
    support = torch.tensor([r["support_values"] for r in train], dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        alpha, s = model.forward_inference(x)
        mean = alpha / alpha.sum(-1, keepdim=True)
        loss = ((mean - w) ** 2).mean() + ((s - support) ** 2).mean() + 10 * torch.relu(w + .01 - s).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    metrics = evaluate_model(model, heldout)
    metadata = {"version": MODEL_VERSION, "semantics": semantics(contract), "context_schema": CONTEXT_SCHEMA,
                "architecture_digest": file_digest(SUBREP / "generator/mdn.py"),
                "mapping_version": MAPPING_VERSION, "objective_names": ["safety", "fuel"],
                "config": config, "train_seeds": list(train_seeds), "validation_seeds": list(validation_seeds),
                "dataset_digest": digest(rows), "seed": seed, "epochs": epochs,
                "target_rule": "mapping_distillation_plus_measured_dispersion_slack_v1",
                "thresholds": THRESHOLDS, "validation": metrics, "approved": accepted(metrics)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, path)
    return metadata


class ContextualModel:
    def __init__(self, path, contract, weight_owner="metamo", fallback="error"):
        if weight_owner not in ("metamo", "mdn") or fallback not in ("error", "global"):
            raise ValueError("explicit weight owner and error/global fallback required")
        self.path = Path(path).resolve()
        self.contract = contract
        self.weight_owner, self.fallback = weight_owner, fallback
        self.failures = 0
        self.queries = 0
        self.invalid_support_events = 0
        self.model, self.metadata, self.sha = None, None, None
        try:
            self.sha = file_digest(self.path)
            checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
            m = checkpoint["metadata"]
            if (m["version"] != MODEL_VERSION or m["semantics"] != semantics(contract)
                    or m.get("architecture_digest") != file_digest(SUBREP / "generator/mdn.py")
                    or m["context_schema"] != CONTEXT_SCHEMA or m["mapping_version"] != MAPPING_VERSION
                    or m["objective_names"] != ["safety", "fuel"] or m["config"]["input_dim"] != 18
                    or m["config"]["num_objectives"] != 2 or not m["approved"]
                    or not accepted(m["validation"]) or m["thresholds"] != THRESHOLDS
                    or not m["train_seeds"] or not m["validation_seeds"]
                    or set(m["train_seeds"]) & set(m["validation_seeds"])):
                raise BoundaryError("MODEL_NOT_APPROVED", "checkpoint schema or held-out acceptance failed")
            self.model = MotiveDecompositionNetwork(**m["config"])
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            self.model.eval()
            self.metadata = m
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.model = None
            if fallback == "error":
                raise BoundaryError("MODEL_NOT_APPROVED", str(exc)) from exc

    def geometry(self, base):
        self.queries += 1
        try:
            if self.model is None:
                raise BoundaryError("MODEL_UNAVAILABLE", "explicit global fallback")
            if file_digest(self.path) != self.sha:
                raise BoundaryError("MODEL_CHANGED", "checkpoint digest changed")
            with torch.no_grad():
                a, s = self.model.forward_inference(torch.tensor(base.context, dtype=torch.float32))
            alpha = tuple(float(v) for v in a)
            support = tuple(float(v) for v in s)
            if not np.isfinite(support).all() or min(support) < 0 or max(support) > 1 or sum(support) < 1 - 1e-9:
                self.invalid_support_events += 1
                raise BoundaryError("INVALID_GEOMETRY", "MDN emitted infeasible support")
            w = tuple(float(v) for v in (a.double() / a.double().sum())) if self.weight_owner == "mdn" else base.weights
            return replace(base, weights=w, region="MDN_WX", alpha=alpha, support_values=support,
                           model_digest=self.sha, model_path=str(self.path), weight_owner=self.weight_owner,
                           geometry_source="trained_sasp_mdn")
        except (OSError, ValueError, RuntimeError) as exc:
            self.failures += 1
            if self.fallback == "error":
                raise BoundaryError(getattr(exc, "code", "INVALID_GEOMETRY"), str(exc)) from exc
            return replace(base, geometry_source="explicit_global_fallback", fallback_reason=str(exc))
