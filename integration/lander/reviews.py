"""Durable anomaly-to-provider repair requests and replacement certification.

An old certificate is never overwritten or unsuspended by a passing repeat.
The replacement receives its own immutable identity and baseline/evidence set.
"""
import json
from .contracts import BoundaryError
from .persistence import atomic_write_json


class FeedbackReviews:
    def __init__(self, repository):
        self.repository = repository
        self.path = repository.root / "feedback.json"

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {"events": [], "suspended": []}

    def save(self, state):
        atomic_write_json(self.path, state)

    def repair_requests(self):
        return [e for e in self.read()["events"] if e.get("status", "open") in ("open", "repair_required")]

    def enqueue(self, event_id, review_repository, replacement):
        state = self.read()
        event = next(e for e in state["events"] if e["event_id"] == event_id)
        original = self.repository.manifests[event["skill_id"]]["behavior"]
        if original["artifact_digest"] != event["artifact_digest"]:
            raise BoundaryError("IDENTITY_CONFLICT", "feedback artifact does not match immutable admission")
        if (replacement.skill_id == original["skill_id"]
                or (replacement.lineage_id or replacement.action.id) != original["lineage_id"]):
            raise BoundaryError("IDENTITY_CONFLICT", "review requires a new immutable version in the same lineage")
        # The anomaly start must be included in the declared comparison protocol.
        if event["outcome"]["seed"] not in review_repository.contract.seeds:
            raise BoundaryError("SCHEMA_MISMATCH", "review evidence must include the anomalous reset seed")
        review_repository.verify_ready()
        review_repository.registry.by_skill(replacement.skill_id)
        if event.get("status") == "queued":
            return event["request"]
        request = review_repository.scheduler.queue_review(event, replacement, review_repository.contract)
        event.update(status="queued", request=request, review_contract=review_repository.contract.identity)
        self.save(state)
        return request

    def dispatch(self, review_repository, evaluation_env, epsilon=0):
        dispatched = review_repository.scheduler.dispatch(review_repository, evaluation_env, epsilon=epsilon)
        if dispatched is None:
            return None
        state = self.read()
        for event in state["events"]:
            if event.get("request", {}).get("request_id") == dispatched["request"]["request_id"]:
                result = dispatched["result"]
                event["status"] = "resolved_with_replacement" if result["admitted"] else "repair_required"
                event["resolution"] = result
                event["review_contract"] = review_repository.contract.identity
                self.save(state)
                break
        return dispatched
