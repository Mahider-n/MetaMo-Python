"""Matched resets and fresh policies; all rollouts use SubRep's executor."""
from dataclasses import asdict
from importlib.metadata import version
import numpy as np
from .contracts import BoundaryError, MeasurementContract, digest, vector
from .vendor import enable_subrep


def make_environment(contract):
    for package, expected in (("mo-gymnasium", contract.mo_gymnasium_version),
                              ("gymnasium", contract.gymnasium_version),
                              ("box2d-py", contract.box2d_version)):
        if version(package) != expected:
            raise BoundaryError("SCHEMA_MISMATCH", f"{package}: expected {expected}, got {version(package)}")
    enable_subrep()
    from env.lunar_lander_wrapper import SubRepEnv
    return SubRepEnv(seed=contract.seeds[0])


class ExecutionCancelled(RuntimeError):
    pass


class ObservedEnvironment:
    """Capture raw reward/terminal diagnostics discarded by SkillExecutor."""
    def __init__(self, env, cancelled=None):
        self.env = env
        self.raw_rewards = []
        self.last_info = {}
        self.cancelled = cancelled

    def step(self, action):
        if self.cancelled is not None and self.cancelled():
            raise ExecutionCancelled("host cancellation before next simulator step")
        result = self.env.step(action)
        self.last_info = result[4]
        raw = vector(self.last_info["raw_rewards"], 4, "INVALID_EVIDENCE")
        mapped = vector(result[1], 2, "INVALID_EVIDENCE")
        if not np.allclose(mapped, [raw[0] + raw[1], raw[2] + raw[3]], atol=1e-6):
            raise BoundaryError("SCHEMA_MISMATCH", "reward mapping differs")
        if raw[2] > 0 or raw[3] > 0:
            raise BoundaryError("SCHEMA_MISMATCH", "engine costs must remain nonpositive")
        self.raw_rewards.append(raw.tolist())
        return result


def rollout(env, policy, contract, seed, initial_obs=None, cancelled=None):
    enable_subrep()
    from env.skill_executor import SkillExecutor
    # initial_obs is only allowed after the caller really reset this env, or
    # for operational continuation. It is never used as simulator restoration.
    obs = env.reset(seed=seed)[0] if initial_obs is None else initial_obs
    vector(obs, 8)
    watched = ObservedEnvironment(env, cancelled)
    executor = SkillExecutor(watched, policy, gamma=contract.gamma, max_steps=contract.horizon)
    try:
        payoff, objectives, terminated = executor.run_episode(initial_obs=obs)
    except Exception as exc:
        exc.partial_outcome = {"seed": seed, "steps": len(watched.raw_rewards),
                               "raw_rewards": watched.raw_rewards, "gamma": contract.gamma,
                               "horizon": contract.horizon,
                               "stop_reason": "cancelled" if isinstance(exc, ExecutionCancelled) else "execution_error"}
        raise
    info = executor.last_run_info
    return {"seed": seed, "initial_observation": np.asarray(obs).tolist(),
            "payoff": float(payoff), "objectives": vector(objectives, 2, "INVALID_EVIDENCE").tolist(),
            "terminated": bool(terminated), "truncated": bool(info["truncated"]),
            "stop_reason": info["stop_reason"], "steps": info["steps"],
            "landing_success": bool(watched.last_info.get("landing_success", False)),
            "crashed": bool(watched.last_info.get("crashed", False)),
            "raw_rewards": watched.raw_rewards}


def collect(env, registry, candidate, contract):
    episodes = [rollout(env, registry.policy(candidate, seed), contract, seed) for seed in contract.seeds]
    evidence = {"contract": asdict(contract), "behavior": candidate.metadata(), "episodes": episodes}
    evidence.update(summarize(episodes))
    validate_evidence(evidence, contract)
    return evidence


def summarize(episodes):
    payoffs = np.array([e["payoff"] for e in episodes])
    objectives = np.array([e["objectives"] for e in episodes])
    return {"mean_payoff": float(payoffs.mean()), "std_payoff": float(payoffs.std()),
            "mean_objectives": objectives.mean(axis=0).tolist(),
            "std_objectives": objectives.std(axis=0).tolist()}


def validate_evidence(evidence, contract):
    contract.require(MeasurementContract(**evidence["contract"]))
    episodes = evidence["episodes"]
    if [e["seed"] for e in episodes] != list(contract.seeds):
        raise BoundaryError("SCHEMA_MISMATCH", "episode reset seeds differ")
    for e in episodes:
        vector(e["initial_observation"], 8, "INVALID_EVIDENCE")
        raw = np.asarray(e["raw_rewards"], dtype=float)
        if raw.shape != (e["steps"], 4) or not 0 < e["steps"] <= contract.horizon or not np.isfinite(raw).all():
            raise BoundaryError("INVALID_EVIDENCE", "invalid raw episode rewards")
        if (raw[:, 2:] > 0).any():
            raise BoundaryError("SCHEMA_MISMATCH", "positive engine cost")
        if any(type(e[k]) is not bool for k in ("terminated", "truncated", "landing_success", "crashed")):
            raise BoundaryError("INVALID_EVIDENCE", "terminal diagnostics must be booleans")
        expected_reason = "terminated" if e["terminated"] else ("truncated" if e["truncated"] else "max_steps")
        if e["stop_reason"] != expected_reason or (expected_reason == "max_steps" and e["steps"] != contract.horizon):
            raise BoundaryError("INVALID_EVIDENCE", "inconsistent rollout stop reason")
        success = bool(e["terminated"] and not e["truncated"] and raw[-1, 0] >= 100)
        crashed = bool(e["terminated"] and raw[-1, 0] <= -100)
        if e["landing_success"] != success or e["crashed"] != crashed:
            raise BoundaryError("INVALID_EVIDENCE", "terminal reward disagrees with success diagnostics")
        mapped = np.column_stack((raw[:, 0] + raw[:, 1], raw[:, 2] + raw[:, 3]))
        totals = np.sum(mapped * (contract.gamma ** np.arange(len(raw)))[:, None], axis=0)
        if not np.allclose(totals, e["objectives"], atol=1e-3, rtol=1e-5) or not np.isclose(totals.sum(), e["payoff"], atol=1e-3, rtol=1e-5):
            raise BoundaryError("INVALID_EVIDENCE", "raw rewards disagree with episode returns")
    if digest(summarize(episodes)) != digest({key: evidence[key] for key in summarize(episodes)}):
        raise BoundaryError("INVALID_EVIDENCE", "aggregate differs from episode evidence")


def matched_improvements(baseline, candidate, contract):
    enable_subrep()
    from baseline.improvement_calculator import ImprovementCalculator
    validate_evidence(baseline, contract)
    validate_evidence(candidate, contract)
    for base, skill in zip(baseline["episodes"], candidate["episodes"]):
        if base["initial_observation"] != skill["initial_observation"]:
            raise BoundaryError("SCHEMA_MISMATCH", "matched reset did not reproduce observation")
    calculator = ImprovementCalculator({"baseline_payoff": baseline["mean_payoff"],
                                        "baseline_motives": baseline["mean_objectives"]})
    return calculator.compute_improvements(candidate["mean_payoff"], candidate["mean_objectives"])
