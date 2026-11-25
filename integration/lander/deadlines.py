"""Linux process watchdog for decision work, including blocking native calls.

No untrusted decision callback or synchronous trace I/O runs on the waiting host.
This bounds waiting by the wall-clock budget; Linux scheduling/IPC overhead is
measured separately and is not a hard real-time guarantee.
"""
import multiprocessing as mp
import time
from .contracts import BoundaryError


def run_decision(host, observation, stimulus, actions, started, deadline):
    context = mp.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    def worker():
        receiver.close()
        events = []
        host.trace = events.append
        try:
            result = host._select_impl(observation, stimulus, actions, decision_started=started, decision_deadline=deadline)
            counters = None if host.contextual_model is None else (
                host.contextual_model.queries, host.contextual_model.failures, host.contextual_model.invalid_support_events)
            sender.send(("ok", result, host._pending, host._backup_pending, events, counters))
        except Exception as exc:
            sender.send(("error", getattr(exc, "code", "DECISION_FAILURE"), str(exc)))
        finally:
            sender.close()
    process = context.Process(target=worker, daemon=True)
    try:
        process.start()
        sender.close()
        # Reserve response handling time rather than waiting until the last ns.
        reserve = min(.002, host.decision_budget_seconds * .2)
        remaining = max(0., deadline - time.monotonic() - reserve)
        if not receiver.poll(remaining):
            return None
        try:
            message = receiver.recv()
        except EOFError as exc:
            raise BoundaryError("DECISION_FAILURE", "decision worker exited without a result") from exc
        if time.monotonic() >= deadline:
            return None
        if message[0] == "error":
            raise BoundaryError(message[1], message[2])
        _, result, pending, backup, events, counters = message
        by_id = {a.id: a for a in actions}
        result.eligible_actions = tuple(by_id[a.id] for a in result.eligible_actions)
        if pending:
            selection, snapshot, _, chosen, ca, cb = pending
            pending = (selection, snapshot, tuple(actions), by_id[chosen.id], ca, cb)
        host._pending, host._backup_pending = pending, backup
        host._trace_queue.extend(events)
        if counters:
            host.contextual_model.queries, host.contextual_model.failures, host.contextual_model.invalid_support_events = counters
        return result
    finally:
        receiver.close()
        sender.close()
        # Never synchronously wait for a stuck decision worker to clean up.
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=0)
