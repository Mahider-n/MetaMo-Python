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
import subprocess
import uuid
import numpy as np
from .contracts import BoundaryError, MeasurementContract, budget, canonical, digest, file_digest
from .evidence import matched_improvements, validate_evidence
from .motives import BUDGET_VERSION, CONTEXT_SCHEMA, MAPPING_VERSION
from .vendor import SUBREP, enable_subrep

enable_subrep()
from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from certification.metta_storage import CertificateStore
from library.skill_library import SkillLibrary


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write(canonical(data) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RejectionMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []

    def record(self, candidate, contract, reason, evidence=None, **details):
        scope = {"skill_id": candidate.skill_id, "artifact_digest": candidate.artifact_digest,
                 "contract_id": contract.identity, "scope": "FULL_SIMPLEX", "reason": reason}
        record = {**scope, "rejection_key": digest(scope), "event_id": uuid.uuid4().hex,
                  "timestamp": now(), "evidence_digest": digest(evidence) if evidence else None,
                  "evidence": evidence, "details": details}
        updated = self.records + [record]
        temp = self.path.with_suffix(".tmp")
        write_json(temp, updated)
        temp.replace(self.path)
        fsync_dir(self.path.parent)
        self.records = updated
        return record


class CertifiedRepository:
    def __init__(self, root, registry, contract):
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

    def admit(self, candidate, evidence, epsilon=0.0):
        epsilon = budget(epsilon)
        self.registry.verify(candidate)
        if evidence["behavior"] != candidate.metadata():
            raise BoundaryError("IDENTITY_CONFLICT", "evidence behavior differs")
        delta_r, delta_n = matched_improvements(self.baseline, evidence, self.contract)
        if CDSGate().admit(delta_r, delta_n):
            gate, eps = CDSGate(), 0.0
        elif PDSGate(epsilon).admit(delta_r, delta_n):
            gate, eps = PDSGate(epsilon), epsilon
        else:
            self.rejections.record(candidate, self.contract, "GATE_REJECTED", evidence,
                                   delta_r=delta_r, delta_n=delta_n.tolist(), epsilon=epsilon,
                                   robust_score=float(delta_r + min(delta_n)))
            return {"skill_id": candidate.skill_id, "admitted": False, "reason": "GATE_REJECTED",
                    "delta_r": delta_r, "delta_n": delta_n.tolist()}
        existing = self.store.get_certificate(candidate.skill_id)
        if existing is not None:
            if digest(self.evidence[candidate.skill_id]) != digest(evidence) or existing.epsilon != eps:
                raise BoundaryError("IDENTITY_CONFLICT", "conflicting admission retry")
            self.verify_entry(candidate.skill_id)
            return {"skill_id": candidate.skill_id, "admitted": True, "certificate": existing.to_dict()}
        cert = Certificate(candidate.skill_id, gate.get_gate_type(), delta_r, tuple(delta_n),
                           gate.get_admission_margin(delta_r, delta_n), eps, now(),
                           self.contract.seeds[0], self.contract.gamma, self.baseline_id,
                           self.contract.environment, self.contract.horizon, "metamo_subrep_bridge_v1")
        manifest = {"version": "lander_manifest_v1", "behavior": candidate.metadata(),
                    "contract": asdict(self.contract), "contract_id": self.contract.identity,
                    "certificate_digest": digest(cert.to_dict()), "baseline_digest": digest(self.baseline),
                    "evidence_digest": digest(evidence), "mapping_version": MAPPING_VERSION,
                    "budget_version": BUDGET_VERSION, "context_schema": CONTEXT_SCHEMA,
                    "source_revisions": {
                        "subrep": subprocess.check_output(["git", "-C", str(SUBREP), "rev-parse", "HEAD"], text=True).strip(),
                        "host_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}}
        inserted = False
        try:
            inserted = self.store.add(cert)
            if not inserted:
                raise BoundaryError("PERSISTENCE_CONFLICT", "certificate insertion refused")
            if not self.library.add_skill(candidate.skill_id, cert, self.registry.policy(candidate, self.contract.seeds[0])):
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
        if (cert.weight_region_type != "FULL_SIMPLEX" or entry.weight_region_type != "FULL_SIMPLEX"
                or len(cert.delta_n) != 2 or cert.baseline_id != self.baseline_id
                or cert.environment != self.contract.environment or cert.gamma != self.contract.gamma
                or cert.episode_length != self.contract.horizon or cert.seed != self.contract.seeds[0]):
            raise BoundaryError("SCHEMA_MISMATCH", "certificate semantics differ")
        dr, dn = matched_improvements(self.baseline, self.evidence[skill_id], self.contract)
        margin = float(dr + min(dn) + cert.epsilon)
        if (dr != cert.delta_r or tuple(dn) != cert.delta_n or margin < 0
                or not math.isclose(margin, cert.admission_margin, abs_tol=1e-7)):
            raise BoundaryError("INVALID_EVIDENCE", "certificate mathematics disagree with evidence")
        if not callable(entry.policy):
            raise BoundaryError("UNRESOLVED_EXECUTABLE", skill_id)
        return candidate, cert

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
        restored = CertifiedRepository(self.root, self.registry, self.contract)
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
        return self.verify_ready()
