"""Persistent bounded reconsideration. Decay never alters certification evidence."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import uuid
from .contracts import BoundaryError, MeasurementContract, digest, file_digest
from .persistence import atomic_write_json


@dataclass(frozen=True)
class SchedulerConfig:
    half_life_seconds: float = 3600.
    cooldown_seconds: float = 60.
    fixed_increase: float = 1.
    deficit_sensitivity: float = .1
    maximum_penalty: float = 100.
    max_retries: int = 3
    version: str = "reconsideration_seconds_v1"

    def __post_init__(self):
        values = (self.half_life_seconds, self.cooldown_seconds, self.fixed_increase,
                  self.deficit_sensitivity, self.maximum_penalty)
        if not all(math.isfinite(v) for v in values) or min(values) < 0 or self.half_life_seconds <= 0 or self.fixed_increase <= 0:
            raise ValueError("invalid scheduler parameters")
        if type(self.max_retries) is not int or self.max_retries < 1:
            raise ValueError("positive integer retry budget required")


class ReevaluationScheduler:
    def __init__(self, path, config=None, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.config = config or SchedulerConfig()
        self.state = {"version": 1, "config": asdict(self.config), "entries": {}, "events": []}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["config"] != asdict(self.config):
                raise BoundaryError("SCHEMA_MISMATCH", "scheduler configuration requires explicit migration")
            # An interrupted claim consumed its retry budget. It may not be
            # replayed after restart with a fresh attempt identity.
            for entry in self.state["entries"].values():
                if entry.get("in_flight"):
                    self.state["events"].append({"event": "interrupted_attempt", "request": entry["in_flight"]})
                    entry["in_flight"] = None
            self.save()
        self.state.setdefault("pending_requests", {})
        for entry in self.state["entries"].values():
            entry.setdefault("epoch", 0)
        # Pre-epoch queued requests cannot be safely migrated into a budget.
        self.state["pending_requests"] = {k: r for k, r in self.state["pending_requests"].items() if "epoch" in r}

    def save(self):
        atomic_write_json(self.path, self.state)

    def penalty(self, entry, at=None):
        elapsed = max(0, (self.clock() if at is None else at) - entry["last_failure"])
        return entry["penalty"] * 2 ** (-elapsed / self.config.half_life_seconds)

    def observe(self, record):
        if record["reason"] not in ("GATE_REJECTED", "REVIEW_REQUIRED"):
            return None  # Repair failures are not repeated failed mathematics.
        lineage = record.get("lineage_id", record["skill_id"].rsplit(".", 1)[0])
        key = digest([lineage, record["contract_id"], record["scope"]])
        entry = self.state["entries"].get(key)
        if entry and record["event_id"] in entry["rejection_ids"]:
            return key
        t = self.clock()
        if entry is None:
            entry = {"key": key, "lineage_id": lineage, "contract_id": record["contract_id"],
                     "scope": record["scope"], "penalty": 0., "last_failure": t,
                     "attempts": 0, "epoch": 0, "triggers": [], "rejection_ids": [], "in_flight": None}
            self.state["entries"][key] = entry
        is_review = record["reason"] == "REVIEW_REQUIRED"
        deficit = 0. if is_review else max(0., -(record["details"]["robust_score"] + record["details"]["epsilon"]))
        if not is_review:
            entry["penalty"] = min(self.config.maximum_penalty,
                                   self.penalty(entry, t) + self.config.fixed_increase + self.config.deficit_sensitivity * deficit)
        entry.update(last_failure=t, skill_id=record["skill_id"], artifact_digest=record["artifact_digest"],
                     rejection_id=record["event_id"], admitted=False,
                     evidence_digest=record.get("evidence_digest"),
                     context_id=(record["details"].get("geometry") or {}).get("context_id"))
        if is_review:
            entry["last_failure"] = t - self.config.cooldown_seconds
            entry["review_event_id"] = record["event_id"]
        entry["rejection_ids"].append(record["event_id"])
        self.state["events"].append({"event": "rejection", "key": key, "rejection_id": record["event_id"],
                                     "deficit": deficit, "penalty": entry["penalty"], "time": t})
        self.save()
        return key

    def request(self, key, candidate, reason, change_token, context_id=None, base_priority=1., evidence_reference=None):
        if reason not in ("behavior_revision", "context_change", "additional_evidence", "outcome_review") or not change_token:
            raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "a relevant, identified change is required")
        entry = self.state["entries"][key]
        if reason == "outcome_review" and change_token != entry.get("review_event_id"):
            raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "review event is not registered")
        lineage = candidate.lineage_id or candidate.action.id
        if lineage != entry["lineage_id"]:
            raise BoundaryError("IDENTITY_CONFLICT", "request must retain rejection lineage")
        if reason == "behavior_revision" and (candidate.artifact_digest == entry["artifact_digest"] or change_token != candidate.artifact_digest):
            raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "new version labels alone are not behavior revisions")
        if reason == "context_change" and change_token != context_id:
            raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "context change token must identify the actual snapshot")
        if reason == "context_change" and (entry["scope"] == "FULL_SIMPLEX" or context_id == entry.get("context_id")):
            raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "pure motive changes do not invalidate global evidence; contextual region must actually change")
        if reason == "additional_evidence":
            from .evidence import validate_evidence
            if not evidence_reference or file_digest(Path(evidence_reference)) != change_token:
                raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "additional evidence requires a matching file digest")
            evidence = json.loads(Path(evidence_reference).read_text())
            protocol = MeasurementContract(**evidence["contract"])
            validate_evidence(evidence, protocol)
            if (protocol.identity != entry["contract_id"] or evidence["behavior"] != candidate.metadata()
                    or digest(evidence) == entry.get("evidence_digest")):
                raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "additional evidence is duplicate or incompatible")
        token = digest([reason, change_token])
        if token in entry["triggers"] or entry["in_flight"] or entry["admitted"]:
            raise BoundaryError("NO_RECONSIDERATION_TRIGGER", "trigger consumed, pending, or already admitted")
        if self.clock() < max(entry["last_failure"], entry.get("last_attempt", 0)) + self.config.cooldown_seconds:
            raise BoundaryError("COOLDOWN", "minimum cooldown has not elapsed")
        # A genuinely new change reopens an exhausted epoch. The trigger is
        # retained durably so new request IDs cannot repeatedly reopen it.
        if entry["attempts"] >= self.config.max_retries:
            entry["epoch"] += 1
            entry["attempts"] = 0
            self.state["pending_requests"] = {k: r for k, r in self.state["pending_requests"].items() if r["key"] != key}
            self.state["events"].append({"event": "budget_reopened", "key": key, "trigger": token})
        request = {"request_id": uuid.uuid4().hex, "key": key, "rejection_id": entry["rejection_id"],
                   "skill_id": candidate.skill_id, "artifact_digest": candidate.artifact_digest,
                   "reason": reason, "change_token": change_token, "trigger": token,
                   "context_id": context_id, "scheduled_time": self.clock(),
                   "epoch": entry["epoch"], "base_priority": float(base_priority),
                   "evidence_reference": str(evidence_reference) if evidence_reference else None,
                   "cooldown_seconds": self.config.cooldown_seconds, "attempt_count": entry["attempts"],
                   "remaining_budget": self.config.max_retries - entry["attempts"],
                   "priority": float(base_priority - self.penalty(entry))}
        self.state["pending_requests"][request["request_id"]] = json.loads(json.dumps(request))
        self.save()
        return request

    def begin(self, request):
        entry = self.state["entries"][request["key"]]
        if (request["trigger"] in entry["triggers"] or entry["in_flight"] or entry["admitted"]
                or entry["attempts"] >= self.config.max_retries
                or request["attempt_count"] != entry["attempts"]
                or request.get("epoch") != entry["epoch"]
                or self.state["pending_requests"].get(request["request_id"]) != request
                or self.clock() < max(entry["last_failure"], entry.get("last_attempt", 0)) + self.config.cooldown_seconds):
            raise BoundaryError("REEVALUATION_UNAVAILABLE", "cooldown, trigger or budget no longer permits this attempt")
        entry["triggers"].append(request["trigger"])
        entry["attempts"] += 1
        entry["last_attempt"] = self.clock()
        entry["in_flight"] = request
        self.state["events"].append({"event": "attempt_started", "request": request, "time": self.clock()})
        self.save()

    def finish(self, request, admitted, reason=None):
        entry = self.state["entries"][request["key"]]
        if entry["in_flight"] != request:
            raise BoundaryError("IDENTITY_CONFLICT", "attempt is not active")
        entry["in_flight"] = None
        self.state["pending_requests"].pop(request["request_id"], None)
        entry["admitted"] = bool(admitted)
        self.state["events"].append({"event": "attempt_completed", "request_id": request["request_id"],
                                     "key": request["key"], "admitted": bool(admitted), "reason": reason,
                                     "time": self.clock(), "remaining_budget": self.config.max_retries - entry["attempts"]})
        if not admitted and entry["attempts"] >= self.config.max_retries:
            self.state["events"].append({"event": "budget_exhausted", "key": request["key"], "time": self.clock()})
        self.save()

    def reevaluate(self, request, repository, evaluation_env, geometry=None, epsilon=0):
        from .evidence import collect
        candidate = repository.registry.by_skill(request["skill_id"])
        if candidate.artifact_digest != request["artifact_digest"]:
            raise BoundaryError("ARTIFACT_CHANGED", "requested artifact changed")
        if request["reason"] == "context_change" and (geometry is None or geometry.context_id != request["context_id"]):
            raise BoundaryError("STALE_SNAPSHOT", "reevaluation context differs from request")
        self.begin(request)
        try:
            if request["reason"] == "additional_evidence":
                path = Path(request["evidence_reference"])
                if file_digest(path) != request["change_token"]:
                    raise BoundaryError("INVALID_EVIDENCE", "queued additional evidence changed")
                evidence = json.loads(path.read_text())
            else:
                evidence = collect(evaluation_env, repository.registry, candidate, repository.contract)
            result = repository.admit(candidate, evidence, epsilon, geometry=geometry)
        except Exception as exc:
            self.finish(request, False, getattr(exc, "code", type(exc).__name__))
            raise
        self.finish(request, result["admitted"], result.get("reason"))
        return result

    def next_due(self):
        """Choose at dispatch time using current decay, budgets and stable ties."""
        due = []
        for request in self.state["pending_requests"].values():
            entry = self.state["entries"][request["key"]]
            if (entry["in_flight"] or entry["admitted"] or request.get("epoch") != entry["epoch"]
                    or entry["attempts"] >= self.config.max_retries or request["trigger"] in entry["triggers"]
                    or self.clock() < max(entry["last_failure"], entry.get("last_attempt", 0)) + self.config.cooldown_seconds
                    or self.clock() < request["scheduled_time"]):
                continue
            due.append((request["base_priority"] - self.penalty(entry), request))
        if not due:
            return None
        _, request = sorted(due, key=lambda p: (-p[0], p[1]["scheduled_time"], p[1]["request_id"]))[0]
        # Refresh bookkeeping for a request queued before another completed
        # attempt in this same epoch, without granting a new retry budget.
        request["attempt_count"] = self.state["entries"][request["key"]]["attempts"]
        request["remaining_budget"] = self.config.max_retries - request["attempt_count"]
        request["priority"] = request["base_priority"] - self.penalty(self.state["entries"][request["key"]])
        self.save()
        return json.loads(json.dumps(request))

    def dispatch(self, repository, evaluation_env, geometries=None, epsilon=0):
        request = self.next_due()
        if request is None:
            return None
        geometry = (geometries or {}).get(request["context_id"])
        return {"request": request, "result": self.reevaluate(request, repository, evaluation_env, geometry, epsilon)}

    def queue_review(self, event, candidate, contract):
        review_trigger = digest([event["event_id"], candidate.artifact_digest, contract.identity])
        record = {"reason": "REVIEW_REQUIRED", "lineage_id": candidate.lineage_id or candidate.action.id,
                  "skill_id": candidate.skill_id, "artifact_digest": candidate.artifact_digest,
                  "contract_id": contract.identity, "scope": "REVIEW", "event_id": review_trigger,
                  "evidence_digest": event["outcome_digest"], "details": {"anomaly": event["reason"], "feedback_event_id": event["event_id"]}}
        key = self.observe(record)
        return self.request(key, candidate, "outcome_review", review_trigger)
