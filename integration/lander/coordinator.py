"""Synchronous host: SubRep screens, MAGUS selects, executor runs, host commits."""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import time
import uuid
import numpy as np
from category.bimonad import MetaMoPseudoBimonad
from core.state import MotivationalState, Stimulus
from dynamics.coherence import blend_states
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from .contracts import BoundaryError, action_record, budget, canonical, digest, vector, weights
from .evidence import rollout
from .motives import MotiveGeometry, project, robust_score
from .recovery import RecoveryPolicy
from .reviews import FeedbackReviews


def state_record(state):
    return {"G": vector(state.G, 8, unit=True).tolist(), "M": vector(state.M, 6, unit=True).tolist()}


@dataclass(frozen=True)
class NoSelection:
    cycle_id: str
    context_id: str
    rejected: tuple
    decision_deadline: float
    recovery_decision_id: str


@dataclass(frozen=True)
class RecoveryDecision:
    decision_id: str
    cycle_id: str
    outcome: str
    reason: str
    latency_seconds: float
    policy: str = "stop_only_v1"
    context_id: str | None = None
    backup_skill_id: str | None = None
    approval_reference: str | None = None
    applicability: bool = False
    backup_artifact_digest: str | None = None
    no_selection_reference: str | None = None


@dataclass(frozen=True)
class SelectionDecision:
    cycle_id: str
    action_id: str
    skill_id: str
    certificate_digest: str
    geometry: MotiveGeometry
    eligible_ids: tuple
    snapshot_digest: str
    owner: str = "subrep_filter_then_magus"
    behavior_version: str = "v1"


@dataclass
class CycleResult:
    selection: SelectionDecision | None
    no_selection: NoSelection | None
    recovery: RecoveryDecision | None
    eligible_actions: tuple
    rejected: tuple
    certificate_references: dict = field(default_factory=dict)


class Coordinator:
    def __init__(self, repository, state_a=None, state_b=None, epsilon=0.0,
                 decision_budget_seconds=5.0, trace_path=None, selector="subrep_filter_then_magus",
                 contextual_model=None, backup_approval=None, suspend_unexpected=False, weight_override=None,
                 deadline_mode="process"):
        self.repository = repository
        self.registry = repository.registry
        self.bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
        self.state_a = state_a or MotivationalState(np.full(8, .5), np.full(6, .5))
        self.state_b = state_b or self.state_a.copy()
        self.epsilon = budget(epsilon)
        if not np.isfinite(decision_budget_seconds) or decision_budget_seconds <= 0:
            raise ValueError("decision budget must be positive and finite")
        self.decision_budget_seconds = float(decision_budget_seconds)
        self.trace_path = Path(trace_path) if trace_path else None
        self._pending = None
        if selector not in ("subrep_filter_then_magus", "subrep_selects"):
            raise ValueError("unknown final selector")
        self.selector = selector
        self.contextual_model = contextual_model
        self.recovery_policy = RecoveryPolicy(repository, backup_approval)
        self._backup_pending = None
        self.cancelled = False
        self.weight_override = weights(weight_override) if weight_override is not None else None
        self.suspend_unexpected = suspend_unexpected
        self.reviews = FeedbackReviews(repository)
        self.feedback_state = self.reviews.read()
        if deadline_mode not in ("process", "cooperative"):
            raise ValueError("unknown deadline mode")
        self.deadline_mode = deadline_mode
        self._trace_queue = []

    def flush_traces(self):
        """Host calls outside its decision deadline (also done before execution)."""
        while self._trace_queue:
            self.trace(self._trace_queue.pop(0))

    def cancel(self):
        self.cancelled = True
        self._pending = self._backup_pending = None

    def trace(self, event):
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(canonical(event) + "\n")

    def _snapshot(self, obs, actions, geometry):
        return {"observation": vector(obs, 8).tolist(), "state_a": state_record(self.state_a),
                "state_b": state_record(self.state_b), "actions": [action_record(a) for a in actions],
                "geometry": asdict(geometry), "epsilon": self.epsilon, "selector": self.selector,
                "weight_override": self.weight_override}

    def _screen(self, actions, geometry, deadline=None):
        # Upstream returns FULL_SIMPLEX entries using stored epsilon. Enforce
        # schema and the current allowed budget again at this boundary.
        kwargs = {} if geometry.region == "FULL_SIMPLEX" else {"support_directions": np.eye(2), "support_values": np.array(geometry.support_values)}
        # In global fallback mode contextual entries cannot be considered.
        if geometry.region == "FULL_SIMPLEX":
            admissible = {e.skill_id for e in self.repository.library.get_admitted_skills() if e.weight_region_type == "FULL_SIMPLEX"}
        else:
            admissible = {e.skill_id for e in self.repository.library.query_admissible(np.array(geometry.weights), **kwargs)}
        eligible, rejected, references = [], [], {}
        for action in actions:
            if deadline is not None and time.monotonic() >= deadline:
                return (), tuple(rejected + [(action.id, "DECISION_BUDGET_EXHAUSTED")]), {}
            try:
                candidate = self.registry.resolve(action)
                if candidate.skill_id not in admissible:
                    raise BoundaryError("NOT_CERTIFIED", candidate.skill_id)
                _, cert = self.repository.verify_entry(candidate.skill_id)
                if candidate.skill_id in self.feedback_state["suspended"]:
                    raise BoundaryError("SUSPENDED_FOR_REEVALUATION", candidate.skill_id)
                if cert.weight_region_type == "MDN_WX" and self.repository.manifests[candidate.skill_id]["geometry"]["model_digest"] != geometry.model_digest:
                    raise BoundaryError("MODEL_CHANGED", "runtime checkpoint differs from certificate")
                checked_geometry = geometry if cert.weight_region_type == "MDN_WX" else None
                if robust_score(cert.delta_r, cert.delta_n, checked_geometry) < -geometry.epsilon:
                    raise BoundaryError("CURRENT_BUDGET_EXCEEDED", candidate.skill_id)
                eligible.append(action)  # Preserve the caller's exact Action objects and order.
                references[action.id] = (candidate, cert)
            except BoundaryError as exc:
                rejected.append((action.id, exc.code))
        return tuple(eligible), tuple(rejected), references

    def _stop(self, cycle, geometry, rejected, started, deadline, reason, observation=None):
        approval, applicable_reason = (None, reason)
        if reason == "EMPTY_ELIGIBLE_SET" and observation is not None:
            approval, applicable_reason = self.recovery_policy.applicable(observation, deadline)
        recovery = RecoveryDecision(uuid.uuid4().hex, cycle, "backup" if approval else "stop",
                                    applicable_reason if approval or self.recovery_policy.path else reason,
                                    time.monotonic() - started,
                                    "validated_backup_or_stop_v1" if self.recovery_policy.path else "stop_only_v1",
                                    geometry.context_id, approval["approval"]["skill_id"] if approval else None,
                                    approval["approval_reference"] if approval else None, bool(approval))
        recovery = replace(recovery, backup_artifact_digest=approval["approval"]["behavior"]["artifact_digest"] if approval else None,
                           no_selection_reference=cycle)
        no_selection = NoSelection(cycle, geometry.context_id, rejected, deadline, recovery.decision_id)
        self.trace({"event": "no_selection", **asdict(no_selection), "geometry": asdict(geometry)})
        self.trace({"event": "recovery", **asdict(recovery)})
        if approval:
            self._backup_pending = (recovery, self._snapshot(observation, (), geometry), geometry)
        return CycleResult(None, no_selection, recovery, (), rejected)

    def select(self, observation, stimulus, actions):
        if self.deadline_mode == "cooperative":
            return self._select_impl(observation, stimulus, actions)
        from .deadlines import run_decision
        started = time.monotonic()
        deadline = started + self.decision_budget_seconds
        self._pending = self._backup_pending = None
        self.cancelled = False
        if not isinstance(actions, (list, tuple)) or len(actions) > 128:
            raise BoundaryError("INVALID_ACTION", "bounded list/tuple of at most 128 actions required")
        vector(observation, 8)
        state_record(self.state_a)
        state_record(self.state_b)
        vector([stimulus.novelty, stimulus.conduciveness, stimulus.risk, stimulus.effort], 4, unit=True)
        result = None if time.monotonic() >= deadline else run_decision(self, observation, stimulus, actions, started, deadline)
        if result is not None:
            return result
        # Minimal pure-data stop path. Never call appraisers, registry, backup
        # verification or synchronous filesystem writes after a timeout.
        cycle = uuid.uuid4().hex
        context_id = digest([state_record(self.state_a), state_record(self.state_b)])
        recovery = RecoveryDecision(uuid.uuid4().hex, cycle, "stop", "DECISION_BUDGET_EXHAUSTED",
                                    time.monotonic() - started, context_id=context_id, no_selection_reference=cycle)
        no_selection = NoSelection(cycle, context_id, (), deadline, recovery.decision_id)
        self._trace_queue.extend([{"event": "no_selection", **asdict(no_selection)}, {"event": "recovery", **asdict(recovery)}])
        return CycleResult(None, no_selection, recovery, (), ())

    def _select_impl(self, observation, stimulus, actions, decision_started=None, decision_deadline=None):
        started = time.monotonic() if decision_started is None else decision_started
        deadline = started + self.decision_budget_seconds if decision_deadline is None else decision_deadline
        self._pending = None
        self._backup_pending = None
        self.cancelled = False
        cycle = uuid.uuid4().hex
        vector(observation, 8)
        state_record(self.state_a)
        state_record(self.state_b)
        # Validate stimulus before the existing appraisal can clip bad inputs.
        vector([stimulus.novelty, stimulus.conduciveness, stimulus.risk, stimulus.effort], 4, unit=True)
        actions = tuple(actions)
        for action in actions:
            action_record(action)
        if len({a.id for a in actions}) != len(actions):
            raise BoundaryError("IDENTITY_CONFLICT", "duplicate candidate action IDs")
        # Exactly one operational appraisal per subsystem. Consensus geometry
        # is the existing conservative merge of these two decision contexts.
        context_a = self.bimonad.decision_context(self.state_a, stimulus)
        context_b = self.bimonad.decision_context(self.state_b, stimulus)
        geometry = project(self.bimonad.parallel_merge(context_a, context_b), stimulus, self.epsilon)
        if self.weight_override is not None:
            geometry = replace(geometry, weights=weights(self.weight_override), geometry_source="explicit_configured_weights")
        if self.contextual_model is not None:
            try:
                geometry = self.contextual_model.geometry(geometry)
            except BoundaryError as exc:
                return self._stop(cycle, geometry, (("geometry", exc.code),), started, deadline, exc.code)
        eligible, rejected, refs = self._screen(actions, geometry, deadline)
        if time.monotonic() >= deadline:
            return self._stop(cycle, geometry, rejected, started, deadline, "DECISION_BUDGET_EXHAUSTED")
        if not eligible:
            return self._stop(cycle, geometry, rejected, started, deadline, "EMPTY_ELIGIBLE_SET", observation)
        # Subsystem recommendations and final consensus see precisely the same
        # screened subset. They never apply competing predicted state changes.
        decision = self.bimonad.decision
        scores = []
        recommendations = []
        if self.selector == "subrep_filter_then_magus":
            recommendation_a, _ = decision.decide(context_a, eligible)
            recommendation_b, _ = decision.decide(context_b, eligible)
            recommendations = [recommendation_a.id, recommendation_b.id]
            for action in eligible:
                a = decision.score_candidate(context_a, action)
                b = decision.score_candidate(context_b, action)
                scores.append((a + b) / 2 - .25 * abs(a - b))
            chosen = eligible[int(np.argmax(scores))]
        else:
            for action in eligible:
                cert = refs[action.id][1]
                scores.append(cert.delta_r + float(np.dot(geometry.weights, cert.delta_n)))
            chosen = sorted(zip(eligible, scores), key=lambda pair: (-pair[1], refs[pair[0].id][0].skill_id))[0][0]
        if time.monotonic() >= deadline:
            return self._stop(cycle, geometry, rejected, started, deadline, "DECISION_BUDGET_EXHAUSTED")
        candidate, cert = refs[chosen.id]
        snapshot = self._snapshot(observation, actions, geometry)
        selection = SelectionDecision(cycle, chosen.id, candidate.skill_id, digest(cert.to_dict()), geometry,
                                      tuple(a.id for a in eligible), digest(snapshot), self.selector, candidate.behavior_version)
        self._pending = (selection, snapshot, actions, chosen, context_a, context_b)
        self.trace({"event": "selection", **asdict(selection), "snapshot": snapshot,
                    "rejected": rejected, "subsystem_actions": recommendations,
                    "consensus_scores": scores, "latency_seconds": time.monotonic() - started})
        certificate_refs = {a.id: {"skill_id": refs[a.id][0].skill_id, "certificate_digest": digest(refs[a.id][1].to_dict())}
                            for a in eligible}
        self.trace({"event": "eligibility", "cycle_id": cycle, "geometry": asdict(geometry),
                    "certificate_references": certificate_refs, "rejected": rejected, "selection_owner": self.selector})
        return CycleResult(selection, None, None, eligible, rejected, certificate_refs)

    def execute(self, selection, env, observation, seed):
        self.flush_traces()
        if self._pending is None or self._pending[0] != selection:
            raise BoundaryError("STALE_SNAPSHOT", "selection is no longer pending")
        _, _, actions, chosen, context_a, context_b = self._pending
        if digest(self._snapshot(observation, actions, selection.geometry)) != selection.snapshot_digest:
            self._pending = None
            raise BoundaryError("STALE_SNAPSHOT", "state, observation, actions or epsilon changed; screen again")
        if self.contextual_model is not None and selection.geometry.region == "MDN_WX":
            from .contracts import file_digest
            if file_digest(self.contextual_model.path) != selection.geometry.model_digest:
                self._pending = None
                raise BoundaryError("STALE_SNAPSHOT", "contextual checkpoint changed; screen again")
        _, _, refs = self._screen(actions, selection.geometry)
        if chosen.id not in refs or digest(refs[chosen.id][1].to_dict()) != selection.certificate_digest:
            self._pending = None
            raise BoundaryError("STALE_SNAPSHOT", "selected evidence changed; screen again")
        self._pending = None  # A decision can execute once only, even after failure.
        candidate, cert = refs[chosen.id]
        outcome, duration = self._execute_policy(candidate, env, observation, seed, {
            "mode": "normal", "cycle_id": selection.cycle_id,
            "skill_id": selection.skill_id, "certificate_digest": selection.certificate_digest,
        })
        # Only the executed action's delta is applied once, then blended using
        # the existing MetaMo stabilization. Observations affect next appraisal.
        self.state_a = blend_states(self.state_a, self.bimonad.transition_selected(context_a, chosen))
        self.state_b = blend_states(self.state_b, self.bimonad.transition_selected(context_b, chosen))
        feedback = outcome_stimulus(outcome)
        result = {"event": "execution_outcome", "execution_id": uuid.uuid4().hex,
                  "cycle_id": selection.cycle_id, "skill_id": candidate.skill_id,
                  "selection_reference": selection.snapshot_digest,
                  "action_id": chosen.id, "behavior_version": candidate.behavior_version,
                  "certificate_digest": selection.certificate_digest, "baseline_id": cert.baseline_id,
                  "contract_id": self.repository.contract.identity, "mode": "normal",
                  "geometry": asdict(selection.geometry), "outcome": outcome,
                  "duration_seconds": duration,
                  "feedback_version": "lander_next_appraisal_v1", "next_stimulus": asdict(feedback),
                  "state_a": state_record(self.state_a), "state_b": state_record(self.state_b)}
        self.trace(result)
        self._record_feedback(candidate, outcome, selection.cycle_id)
        return result, feedback

    def _execute_policy(self, candidate, env, observation, seed, failure_context):
        """Execute a fresh policy without resetting the operational simulator."""
        started = time.monotonic()
        try:
            outcome = rollout(
                env, self.registry.policy(candidate, seed), self.repository.contract, seed,
                initial_obs=observation, cancelled=lambda: self.cancelled,
            )
        except Exception as exc:
            self.trace({"event": "execution_failure", **failure_context,
                        "reason": type(exc).__name__, "detail": str(exc),
                        "duration_seconds": time.monotonic() - started,
                        "partial_outcome": getattr(exc, "partial_outcome", None)})
            raise
        return outcome, time.monotonic() - started

    def _record_feedback(self, candidate, outcome, cycle_id):
        evidence = self.repository.evidence[candidate.skill_id]
        # A crash is an anomaly if it never occurred in the admission sample.
        # This flags review, not a baseline-relative certification claim.
        unexpected = outcome["crashed"] and not any(e["crashed"] for e in evidence["episodes"])
        if unexpected:
            self.feedback_state = self.reviews.read()
            event = {"event_id": uuid.uuid4().hex, "status": "open", "cycle_id": cycle_id, "skill_id": candidate.skill_id,
                     "artifact_digest": candidate.artifact_digest, "contract_id": self.repository.contract.identity,
                     "reason": "UNEXPECTED_CRASH", "outcome_digest": digest(outcome), "outcome": outcome,
                     "requires_evaluation": True, "lineage_id": candidate.lineage_id or candidate.action.id}
            self.feedback_state["events"].append(event)
            if self.suspend_unexpected and candidate.skill_id not in self.feedback_state["suspended"]:
                self.feedback_state["suspended"].append(candidate.skill_id)
            self.reviews.save(self.feedback_state)
            self.trace({"event": "evaluation_requested", **event})

    def execute_recovery(self, recovery, env, observation, seed):
        self.flush_traces()
        if self._backup_pending is None or self._backup_pending[0] != recovery:
            raise BoundaryError("STALE_SNAPSHOT", "no applicable pending recovery")
        _, snapshot, geometry = self._backup_pending
        self._backup_pending = None
        if self.cancelled or self._snapshot(observation, (), geometry) != snapshot:
            raise BoundaryError("STALE_SNAPSHOT", "recovery context changed")
        approval, reason = self.recovery_policy.applicable(observation, time.monotonic() + self.decision_budget_seconds, seed=seed)
        if approval is None or approval["approval_reference"] != recovery.approval_reference:
            raise BoundaryError("BACKUP_NOT_APPROVED", reason)
        candidate = self.registry.by_skill(recovery.backup_skill_id)
        outcome, duration = self._execute_policy(candidate, env, observation, seed, {
            "mode": "backup", "cycle_id": recovery.cycle_id,
            "recovery_decision_id": recovery.decision_id,
        })
        result = {"event": "execution_outcome", "mode": "backup", "execution_id": uuid.uuid4().hex,
                  "cycle_id": recovery.cycle_id, "recovery_decision_id": recovery.decision_id,
                  "approval_reference": recovery.approval_reference, "certificate_digest": None,
                  "skill_id": candidate.skill_id, "action_id": candidate.action.id,
                  "behavior_version": candidate.behavior_version, "contract_id": self.repository.contract.identity,
                  "geometry": asdict(geometry), "baseline_id": self.repository.baseline_id,
                  "duration_seconds": duration, "outcome": outcome,
                  "next_stimulus": asdict(outcome_stimulus(outcome)), "feedback_version": "lander_next_appraisal_v1"}
        # Backup effects feed next appraisal only; rejected candidate predictions
        # are never applied during recovery.
        self.trace(result)
        self._record_feedback(candidate, outcome, recovery.cycle_id)
        return result, outcome_stimulus(outcome)


def outcome_stimulus(outcome):
    """Observed two-objective evidence affects next appraisal only."""
    fuel_cost = max(0.0, -float(outcome["objectives"][1]))
    return Stimulus(novelty=0.0, conduciveness=1.0 if outcome["landing_success"] else 0.0,
                    risk=1.0 if outcome["crashed"] else (0.5 if not outcome["landing_success"] else 0.0),
                    effort=float(np.clip(fuel_cost / max(1, outcome["steps"]), 0, 1)))
