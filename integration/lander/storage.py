"""Verified admission and crash-consistent snapshots around native SubRep stores.

An immutable generation contains MeTTa, library JSON, evidence and manifests.
Only the atomic CURRENT pointer makes a generation operational. Orphans from
an interrupted save are ignored; incompatible or corrupt commits fail closed.
Single writer, synchronous host: no cross-process locking is claimed.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid
import numpy as np
from .contracts import BoundaryError, MeasurementContract, budget, canonical, digest, file_digest
from .evidence import matched_improvements, validate_evidence
from .motives import BUDGET_VERSION, CONTEXT_SCHEMA, MAPPING_VERSION, MotiveGeometry, robust_score
from .persistence import atomic_write_json, fsync_dir, write_json
from .vendor import SUBREP, enable_subrep, source_revision

enable_subrep()
from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from certification.metta_storage import CertificateStore
from library.skill_library import SkillLibrary


def now():
    return datetime.now(timezone.utc).isoformat()


class RejectionMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []

    def record(self, candidate, contract, reason, evidence=None, **details):
        scope = {"skill_id": candidate.skill_id, "artifact_digest": candidate.artifact_digest,
                 "contract_id": contract.identity, "scope": details.pop("scope", "FULL_SIMPLEX"), "reason": reason,
                 "lineage_id": candidate.lineage_id or candidate.action.id}
        record = {**scope, "rejection_key": digest(scope), "event_id": uuid.uuid4().hex,
                  "timestamp": now(), "evidence_digest": digest(evidence) if evidence else None,
                  "evidence": evidence, "details": details}
        updated = self.records + [record]
        atomic_write_json(self.path, updated)
        self.records = updated
        return record


class CertifiedRepository:
    def __init__(self, root, registry, contract, scheduler_config=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.contract = contract
        self.store = CertificateStore()
        self.library = SkillLibrary(cert_store=self.store)
        self.manifests = {}
        self.evidence = {}
        self.baseline = None
        self.rejections = RejectionMemory(self.root / "rejections.json")
        from .scheduler import ReevaluationScheduler
        self.scheduler = ReevaluationScheduler(self.root / "scheduler.json", scheduler_config)
        for record in self.rejections.records:
            self.scheduler.observe(record)

    @property
    def baseline_id(self):
        if self.baseline is None:
            raise BoundaryError("INVALID_EVIDENCE", "baseline not configured")
        return "idle_v1_" + digest(self.baseline)

    def set_baseline(self, evidence):
        validate_evidence(evidence, self.contract)
        candidate = self.registry.by_skill(evidence["behavior"]["skill_id"])
        if candidate.loader != "noop" or evidence["behavior"] != candidate.metadata():
            raise BoundaryError("INVALID_EVIDENCE", "baseline must resolve to versioned idle controller")
        if self.baseline is not None and digest(self.baseline) != digest(evidence):
            raise BoundaryError("IDENTITY_CONFLICT", "cannot replace baseline inside repository")
        self.baseline = json.loads(canonical(evidence))

    def admit(self, candidate, evidence, epsilon=0.0, geometry=None):
        epsilon = budget(epsilon)
        self.registry.verify(candidate)
        if evidence["behavior"] != candidate.metadata():
            raise BoundaryError("IDENTITY_CONFLICT", "evidence behavior differs")
        delta_r, delta_n = matched_improvements(self.baseline, evidence, self.contract)
        if geometry is not None and geometry.region == "MDN_WX":
            self.verify_geometry(geometry)
        support = None if geometry is None else geometry.support_values
        if CDSGate().admit(delta_r, delta_n, support_values=support):
            gate, eps = CDSGate(), 0.0
        elif PDSGate(epsilon).admit(delta_r, delta_n, support_values=support):
            gate, eps = PDSGate(epsilon), epsilon
        else:
            record = self.rejections.record(candidate, self.contract, "GATE_REJECTED", evidence,
                                   delta_r=delta_r, delta_n=delta_n.tolist(), epsilon=epsilon,
                                   robust_score=robust_score(delta_r, delta_n, geometry),
                                   scope=geometry.region if geometry else "FULL_SIMPLEX",
                                   geometry=asdict(geometry) if geometry else None)
            self.scheduler.observe(record)
            return {"skill_id": candidate.skill_id, "admitted": False, "reason": "GATE_REJECTED",
                    "delta_r": delta_r, "delta_n": delta_n.tolist()}
        existing = self.store.get_certificate(candidate.skill_id)
        if existing is not None:
            if (digest(self.evidence[candidate.skill_id]) != digest(evidence) or existing.epsilon != eps
                    or digest(self.manifests[candidate.skill_id].get("geometry")) != digest(asdict(geometry) if geometry and geometry.region == "MDN_WX" else None)):
                raise BoundaryError("IDENTITY_CONFLICT", "conflicting admission retry")
            self.verify_entry(candidate.skill_id)
            return {"skill_id": candidate.skill_id, "admitted": True, "certificate": existing.to_dict()}
        contextual = geometry is not None and geometry.region == "MDN_WX"
        audit = {"weight_region_type": "MDN_WX", "certification_context": geometry.context,
                 "mdn_alpha": geometry.alpha, "wx_support_directions": ((1., 0.), (0., 1.)),
                 "wx_support_values": geometry.support_values} if contextual else {"weight_region_type": "FULL_SIMPLEX"}
        cert = Certificate(candidate.skill_id, gate.get_gate_type(), delta_r, tuple(delta_n),
                           gate.get_admission_margin(delta_r, delta_n, support_values=support), eps, now(),
                           self.contract.seeds[0], self.contract.gamma, self.baseline_id,
                           self.contract.environment, self.contract.horizon, "metamo_subrep_bridge_v1", **audit)
        manifest = {"version": "lander_manifest_v1", "behavior": candidate.metadata(),
                    "contract": asdict(self.contract), "contract_id": self.contract.identity,
                    "certificate_digest": digest(cert.to_dict()), "baseline_digest": digest(self.baseline),
                    "evidence_digest": digest(evidence), "mapping_version": MAPPING_VERSION,
                    "budget_version": BUDGET_VERSION, "context_schema": CONTEXT_SCHEMA,
                    "geometry": asdict(geometry) if contextual else None,
                    "source_revisions": {
                        "subrep": source_revision(SUBREP),
                        "host_base": source_revision(Path(__file__).resolve().parents[2])}}
        inserted = False
        try:
            inserted = self.store.add(cert)
            if not inserted:
                raise BoundaryError("PERSISTENCE_CONFLICT", "certificate insertion refused")
            if not self.library.add_skill(candidate.skill_id, cert, self.registry.policy(candidate, self.contract.seeds[0]), **audit):
                raise BoundaryError("PERSISTENCE_CONFLICT", "library insertion refused")
            self.manifests[candidate.skill_id] = manifest
            self.evidence[candidate.skill_id] = json.loads(canonical(evidence))
            self.verify_entry(candidate.skill_id)
            self.save()
        except Exception:
            # Only this insertion is undone. Prior committed generations survive.
            self.library.remove_skill(candidate.skill_id)
            if inserted:
                self.store.remove_skill(candidate.skill_id)
            self.manifests.pop(candidate.skill_id, None)
            self.evidence.pop(candidate.skill_id, None)
            raise
        return {"skill_id": candidate.skill_id, "admitted": True, "certificate": cert.to_dict()}

    def verify_entry(self, skill_id):
        cert = self.store.get_certificate(skill_id)
        entry = self.library.get_skill(skill_id)
        manifest = self.manifests.get(skill_id)
        if cert is None or entry is None or manifest is None:
            raise BoundaryError("PERSISTENCE_CONFLICT", "partial admission")
        candidate = self.registry.by_skill(skill_id)
        self.contract.require(MeasurementContract(**manifest["contract"]))
        if (manifest["version"] != "lander_manifest_v1" or manifest["contract_id"] != self.contract.identity
                or manifest["behavior"] != candidate.metadata()
                or manifest["certificate_digest"] != digest(cert.to_dict())
                or entry.certificate.to_dict() != cert.to_dict()
                or manifest["baseline_digest"] != digest(self.baseline)
                or manifest["evidence_digest"] != digest(self.evidence[skill_id])
                or manifest["mapping_version"] != MAPPING_VERSION
                or manifest["budget_version"] != BUDGET_VERSION
                or manifest["context_schema"] != CONTEXT_SCHEMA):
            raise BoundaryError("PERSISTENCE_CONFLICT", "certificate/manifest/behavior mismatch")
        if (cert.weight_region_type != entry.weight_region_type
                or len(cert.delta_n) != 2 or cert.baseline_id != self.baseline_id
                or cert.environment != self.contract.environment or cert.gamma != self.contract.gamma
                or cert.episode_length != self.contract.horizon or cert.seed != self.contract.seeds[0]):
            raise BoundaryError("SCHEMA_MISMATCH", "certificate semantics differ")
        dr, dn = matched_improvements(self.baseline, self.evidence[skill_id], self.contract)
        geo_record = manifest.get("geometry")
        geometry = MotiveGeometry(**geo_record) if geo_record else None
        if cert.weight_region_type == "MDN_WX":
            if geometry is None or geometry.region != "MDN_WX":
                raise BoundaryError("SCHEMA_MISMATCH", "contextual certificate lacks geometry manifest")
            self.verify_geometry(geometry)
            if (tuple(cert.certification_context) != geometry.context or tuple(cert.mdn_alpha) != geometry.alpha
                    or tuple(cert.wx_support_values) != geometry.support_values
                    or cert.wx_support_directions != ((1., 0.), (0., 1.))
                    or entry.wx_support_values != cert.wx_support_values or entry.mdn_alpha != cert.mdn_alpha
                    or entry.certification_context != cert.certification_context
                    or entry.wx_support_directions != cert.wx_support_directions):
                raise BoundaryError("PERSISTENCE_CONFLICT", "contextual audit fields differ")
        elif geometry is not None:
            raise BoundaryError("SCHEMA_MISMATCH", "global certificate claims contextual geometry")
        margin = robust_score(dr, dn, geometry) + cert.epsilon
        if (dr != cert.delta_r or tuple(dn) != cert.delta_n or margin < 0
                or not math.isclose(margin, cert.admission_margin, abs_tol=1e-7)):
            raise BoundaryError("INVALID_EVIDENCE", "certificate mathematics disagree with evidence")
        if not callable(entry.policy):
            raise BoundaryError("UNRESOLVED_EXECUTABLE", skill_id)
        return candidate, cert

    def verify_geometry(self, geometry):
        if geometry.region == "MDN_WX":
            from .contextual import ContextualModel
            model = ContextualModel(geometry.model_path, self.contract, weight_owner=geometry.weight_owner)
            if model.sha != geometry.model_digest:
                raise BoundaryError("MODEL_CHANGED", "certificate checkpoint digest differs")
            from dataclasses import replace
            base = replace(geometry, region="FULL_SIMPLEX", alpha=None, support_values=None,
                           model_digest=None, model_path=None, geometry_source="explicit_metamo_mapping")
            actual = model.geometry(base)
            if not np.allclose(actual.alpha, geometry.alpha, atol=1e-7) or not np.allclose(actual.support_values, geometry.support_values, atol=1e-7):
                raise BoundaryError("INVALID_GEOMETRY", "audit geometry differs from checkpoint inference")

    def verify_ready(self):
        ids = set(self.manifests)
        if ids != {c.skill_id for c in self.store.load_all()} or ids != {e.skill_id for e in self.library.get_admitted_skills()} or ids != set(self.evidence):
            raise BoundaryError("PERSISTENCE_CONFLICT", "certificate/library/manifest/evidence IDs differ")
        if self.baseline is not None:
            self.set_baseline(self.baseline)
        for skill_id in ids:
            self.verify_entry(skill_id)
        return sorted(ids)

    def save(self):
        self.verify_ready()
        generation = "generation-" + uuid.uuid4().hex
        staging = self.root / (generation + ".pending")
        staging.mkdir()
        self.store.save_to_file(staging / "certificates.metta")
        self.library.save(str(staging / "library.json"))
        write_json(staging / "manifest.json", {"contract": asdict(self.contract), "baseline": self.baseline,
                                              "manifests": self.manifests, "evidence": self.evidence})
        files = ("certificates.metta", "library.json", "manifest.json")
        for name in files:
            with (staging / name).open("rb") as f:
                os.fsync(f.fileno())
        index = {name: file_digest(staging / name) for name in files}
        write_json(staging / "commit.json", index)
        fsync_dir(staging)
        staging.replace(self.root / generation)
        fsync_dir(self.root)
        pointer = self.root / "CURRENT.tmp"
        write_json(pointer, {"generation": generation, "commit_digest": digest(index)})
        pointer.replace(self.root / "CURRENT")
        # The pointer replacement is the commit point. A later directory fsync
        # failure must not cause in-memory rollback of the now visible commit.
        try:
            fsync_dir(self.root)
        except OSError:
            pass

    def load(self):
        pointer = json.loads((self.root / "CURRENT").read_text())
        name = pointer["generation"]
        if not name.startswith("generation-") or Path(name).name != name:
            raise BoundaryError("PERSISTENCE_CONFLICT", "invalid generation pointer")
        directory = self.root / name
        index = json.loads((directory / "commit.json").read_text())
        if set(index) != {"certificates.metta", "library.json", "manifest.json"} or digest(index) != pointer["commit_digest"]:
            raise BoundaryError("PERSISTENCE_CONFLICT", "invalid commit index")
        if any(file_digest(directory / p) != sha for p, sha in index.items()):
            raise BoundaryError("PERSISTENCE_CONFLICT", "partial or modified generation")
        data = json.loads((directory / "manifest.json").read_text())
        self.contract.require(MeasurementContract(**data["contract"]))
        # Build a separate verified repository: a failed load never exposes
        # partly loaded policies in the active instance.
        restored = CertifiedRepository(self.root, self.registry, self.contract, self.scheduler.config)
        restored.set_baseline(data["baseline"])
        restored.store.load_from_file(directory / "certificates.metta")
        restored.library.load(str(directory / "library.json"))
        restored.manifests, restored.evidence = data["manifests"], data["evidence"]
        for entry in restored.library.get_admitted_skills():
            candidate = self.registry.by_skill(entry.skill_id)
            restored.library.register_policy(entry.skill_id, self.registry.policy(candidate, self.contract.seeds[0]))
        restored.verify_ready()
        self.store, self.library = restored.store, restored.library
        self.baseline, self.manifests, self.evidence = restored.baseline, restored.manifests, restored.evidence
        self.rejections = restored.rejections
        self.scheduler = restored.scheduler
        return self.verify_ready()
