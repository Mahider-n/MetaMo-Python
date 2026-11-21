"""Synchronous host: SubRep screens, MAGUS selects, executor runs, host commits."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import uuid
import numpy as np
from category.bimonad import MetaMoPseudoBimonad
from core.state import MotivationalState, Stimulus
from dynamics.coherence import blend_states
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from .contracts import BoundaryError, action_record, budget, canonical, digest, vector
from .evidence import rollout
from .motives import MotiveGeometry, project


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


@dataclass
class CycleResult:
    selection: SelectionDecision | None
    no_selection: NoSelection | None
    recovery: RecoveryDecision | None
    eligible_actions: tuple
    rejected: tuple


class Coordinator:
    def __init__(self, repository, state_a=None, state_b=None, epsilon=0.0,
                 decision_budget_seconds=5.0, trace_path=None):
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

    def trace(self, event):
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(canonical(event) + "\n")

    def _snapshot(self, obs, actions, geometry):
        return {"observation": vector(obs, 8).tolist(), "state_a": state_record(self.state_a),
                "state_b": state_record(self.state_b), "actions": [action_record(a) for a in actions],
                "geometry": asdict(geometry), "epsilon": self.epsilon}

    def _screen(self, actions, geometry, deadline=None):
        # Upstream returns FULL_SIMPLEX entries using stored epsilon. Enforce
        # schema and the current allowed budget again at this boundary.
        admissible = {e.skill_id for e in self.repository.library.query_admissible(np.array(geometry.weights))}
        eligible, rejected, references = [], [], {}
        for action in actions:
            if deadline is not None and time.monotonic() >= deadline:
                return (), tuple(rejected + [(action.id, "DECISION_BUDGET_EXHAUSTED")]), {}
            try:
                candidate = self.registry.resolve(action)
                if candidate.skill_id not in admissible:
                    raise BoundaryError("NOT_CERTIFIED", candidate.skill_id)
                _, cert = self.repository.verify_entry(candidate.skill_id)
                if cert.delta_r + min(cert.delta_n) < -geometry.epsilon:
                    raise BoundaryError("CURRENT_BUDGET_EXCEEDED", candidate.skill_id)
                eligible.append(action)  # Preserve the caller's exact Action objects and order.
                references[action.id] = (candidate, cert)
            except BoundaryError as exc:
                rejected.append((action.id, exc.code))
        return tuple(eligible), tuple(rejected), references

    def _stop(self, cycle, geometry, rejected, started, deadline, reason):
        recovery = RecoveryDecision(uuid.uuid4().hex, cycle, "stop", reason, time.monotonic() - started)
        no_selection = NoSelection(cycle, geometry.context_id, rejected, deadline, recovery.decision_id)
        self.trace({"event": "no_selection", **asdict(no_selection), "geometry": asdict(geometry)})
        self.trace({"event": "recovery", **asdict(recovery)})
        return CycleResult(None, no_selection, recovery, (), rejected)

    def select(self, observation, stimulus, actions):
        started = time.monotonic()
        deadline = started + self.decision_budget_seconds
        self._pending = None
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
        eligible, rejected, refs = self._screen(actions, geometry, deadline)
        if time.monotonic() >= deadline:
            return self._stop(cycle, geometry, rejected, started, deadline, "DECISION_BUDGET_EXHAUSTED")
        if not eligible:
            return self._stop(cycle, geometry, rejected, started, deadline, "EMPTY_ELIGIBLE_SET")
        if time.monotonic() >= deadline:
            return self._stop(cycle, geometry, rejected, started, deadline, "DECISION_BUDGET_EXHAUSTED")
        # Subsystem recommendations and final consensus see precisely the same
        # screened subset. They never apply competing predicted state changes.
        decision = self.bimonad.decision
        recommendation_a, _ = decision.decide(context_a, eligible)
        recommendation_b, _ = decision.decide(context_b, eligible)
        scores = []
        for action in eligible:
            a = decision.score_candidate(context_a, action)
            b = decision.score_candidate(context_b, action)
            scores.append((a + b) / 2 - .25 * abs(a - b))
        chosen = eligible[int(np.argmax(scores))]  # First maximum, original ordering.
        if time.monotonic() >= deadline:
            return self._stop(cycle, geometry, rejected, started, deadline, "DECISION_BUDGET_EXHAUSTED")
        candidate, cert = refs[chosen.id]
        snapshot = self._snapshot(observation, actions, geometry)
        selection = SelectionDecision(cycle, chosen.id, candidate.skill_id, digest(cert.to_dict()), geometry,
                                      tuple(a.id for a in eligible), digest(snapshot))
        self._pending = (selection, snapshot, actions, chosen, context_a, context_b)
        self.trace({"event": "selection", **asdict(selection), "snapshot": snapshot,
                    "rejected": rejected, "subsystem_actions": [recommendation_a.id, recommendation_b.id],
                    "consensus_scores": scores, "latency_seconds": time.monotonic() - started})
        return CycleResult(selection, None, None, eligible, rejected)

    def execute(self, selection, env, observation, seed):
        if self._pending is None or self._pending[0] != selection:
            raise BoundaryError("STALE_SNAPSHOT", "selection is no longer pending")
        _, snapshot, actions, chosen, context_a, context_b = self._pending
        if digest(self._snapshot(observation, actions, selection.geometry)) != selection.snapshot_digest:
            self._pending = None
            raise BoundaryError("STALE_SNAPSHOT", "state, observation, actions or epsilon changed; screen again")
        _, _, refs = self._screen(actions, selection.geometry)
        if chosen.id not in refs or digest(refs[chosen.id][1].to_dict()) != selection.certificate_digest:
            self._pending = None
            raise BoundaryError("STALE_SNAPSHOT", "selected evidence changed; screen again")
        self._pending = None  # A decision can execute once only, even after failure.
        started = time.monotonic()
        try:
            candidate, cert = refs[chosen.id]
            # A fresh policy lifecycle for this atomic rollout. Never reset the
            # operational environment here to collect certification evidence.
            outcome = rollout(env, self.registry.policy(candidate, seed), self.repository.contract,
                              seed, initial_obs=observation)
        except Exception as exc:
            self.trace({"event": "execution_failure", "cycle_id": selection.cycle_id,
                        "skill_id": selection.skill_id, "certificate_digest": selection.certificate_digest,
                        "reason": type(exc).__name__, "detail": str(exc)})
            raise
        # Only the executed action's delta is applied once, then blended using
        # the existing MetaMo stabilization. Observations affect next appraisal.
        self.state_a = blend_states(self.state_a, self.bimonad.transition_selected(context_a, chosen))
        self.state_b = blend_states(self.state_b, self.bimonad.transition_selected(context_b, chosen))
        feedback = outcome_stimulus(outcome)
        result = {"event": "execution_outcome", "execution_id": uuid.uuid4().hex,
                  "cycle_id": selection.cycle_id, "skill_id": candidate.skill_id,
                  "action_id": chosen.id, "behavior_version": candidate.behavior_version,
                  "certificate_digest": selection.certificate_digest, "baseline_id": cert.baseline_id,
                  "contract_id": self.repository.contract.identity, "mode": "normal",
                  "geometry": asdict(selection.geometry), "outcome": outcome,
                  "duration_seconds": time.monotonic() - started,
                  "feedback_version": "lander_next_appraisal_v1", "next_stimulus": asdict(feedback),
                  "state_a": state_record(self.state_a), "state_b": state_record(self.state_b)}
        self.trace(result)
        return result, feedback


def outcome_stimulus(outcome):
    """Observed two-objective evidence affects next appraisal only."""
    fuel_cost = max(0.0, -float(outcome["objectives"][1]))
    return Stimulus(novelty=0.0, conduciveness=1.0 if outcome["landing_success"] else 0.0,
                    risk=1.0 if outcome["crashed"] else (0.5 if not outcome["landing_success"] else 0.0),
                    effort=float(np.clip(fuel_cost / max(1, outcome["steps"]), 0, 1)))
