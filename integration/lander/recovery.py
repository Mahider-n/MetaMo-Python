"""Explicit, evidence-backed backup approval and observation applicability."""
import json
from pathlib import Path
import numpy as np
from .contracts import BoundaryError, digest, vector
from .persistence import write_json


def approve_backup(repository, skill_id, path, minimum_success_rate=.2, maximum_crash_rate=.2):
    """Engineering validation policy, not a claim of universal safe control."""
    if not 0 <= minimum_success_rate <= 1 or not 0 <= maximum_crash_rate <= 1:
        raise ValueError("backup rate thresholds must be in [0,1]")
    candidate, cert = repository.verify_entry(skill_id)
    if cert.weight_region_type != "FULL_SIMPLEX" or cert.gate_type != "CDS":
        raise BoundaryError("BACKUP_NOT_APPROVED", "backup requires a global CDS certificate")
    evidence = repository.evidence[skill_id]
    episodes = evidence["episodes"]
    if len(episodes) < 5:
        raise BoundaryError("BACKUP_NOT_APPROVED", "at least five evaluation episodes required")
    success = sum(e["landing_success"] for e in episodes) / len(episodes)
    crash = sum(e["crashed"] for e in episodes) / len(episodes)
    if success < minimum_success_rate or crash > maximum_crash_rate:
        raise BoundaryError("BACKUP_NOT_APPROVED", "declared empirical backup thresholds failed")
    approval = {"version": "measured_backup_v2", "authority": "configured_empirical_backup_policy_v1",
                "skill_id": skill_id, "behavior": candidate.metadata(), "contract_id": repository.contract.identity,
                "certificate_digest": digest(cert.to_dict()), "evidence_digest": digest(evidence),
                "episode_count": len(episodes), "success_rate": success, "crash_rate": crash,
                "minimum_success_rate": minimum_success_rate, "maximum_crash_rate": maximum_crash_rate,
                "evaluated_starts": [{"seed": e["seed"], "observation": e["initial_observation"]} for e in episodes],
                "applicability": "exact_evaluated_seed_and_start_v1"}
    record = {"approval": approval, "approval_reference": digest(approval)}
    write_json(path, record)
    return record


class RecoveryPolicy:
    def __init__(self, repository, approval_path=None):
        self.repository = repository
        self.path = Path(approval_path) if approval_path else None

    def applicable(self, observation, deadline, seed=None):
        import time
        if self.path is None or time.monotonic() >= deadline:
            return None, "NO_APPLICABLE_BACKUP"
        try:
            record = json.loads(self.path.read_text())
            approval = record["approval"]
            if digest(approval) != record["approval_reference"] or approval["version"] != "measured_backup_v2":
                raise BoundaryError("BACKUP_NOT_APPROVED", "approval contents differ")
            candidate, cert = self.repository.verify_entry(approval["skill_id"])
            feedback = self.repository.root / "feedback.json"
            if feedback.exists() and candidate.skill_id in json.loads(feedback.read_text())["suspended"]:
                return None, "SUSPENDED_FOR_REEVALUATION"
            evidence = self.repository.evidence[candidate.skill_id]
            episodes = evidence["episodes"]
            success = sum(e["landing_success"] for e in episodes) / len(episodes)
            crash = sum(e["crashed"] for e in episodes) / len(episodes)
            starts = [{"seed": e["seed"], "observation": e["initial_observation"]} for e in episodes]
            if (approval["authority"] != "configured_empirical_backup_policy_v1"
                    or approval["contract_id"] != self.repository.contract.identity
                    or approval["behavior"] != candidate.metadata()
                    or approval["certificate_digest"] != digest(cert.to_dict())
                    or approval["evidence_digest"] != digest(evidence)
                    or cert.weight_region_type != "FULL_SIMPLEX" or cert.gate_type != "CDS"
                    or len(episodes) < 5 or approval["episode_count"] != len(episodes)
                    or success != approval["success_rate"] or crash != approval["crash_rate"]
                    or success < approval["minimum_success_rate"] or crash > approval["maximum_crash_rate"]
                    or approval["evaluated_starts"] != starts
                    or approval["applicability"] != "exact_evaluated_seed_and_start_v1"):
                raise BoundaryError("BACKUP_NOT_APPROVED", "backup identity or evidence mismatch")
            obs = vector(observation, 8)
            if not any((seed is None or seed == start["seed"]) and np.allclose(obs, start["observation"], atol=1e-7, rtol=0)
                       for start in starts):
                return None, "BACKUP_NOT_APPLICABLE"
            if time.monotonic() >= deadline:
                return None, "DECISION_BUDGET_EXHAUSTED"
            return record, "APPLICABLE_VALIDATED_BACKUP"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return None, getattr(exc, "code", "BACKUP_NOT_APPROVED")
