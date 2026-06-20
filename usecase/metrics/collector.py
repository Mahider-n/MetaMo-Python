"""
Logs and computes all evaluation metrics.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class EpisodeLog:
    minerals_collected: int  = 0
    minerals_spawned:   int  = 1
    total_steps:        int  = 0
    total_reward:       float = 0.0
    lava_steps:         int  = 0       # steps spent in lava
    srv_flags:          list = field(default_factory=list)  # agent-specific safe-region violations
    unsafe_flags:       list = field(default_factory=list)  # environment danger/lava band
    arousal_log:        list = field(default_factory=list)  # MetaMo only
    safety_log:         list = field(default_factory=list)  # MetaMo only
    individuation_log:  list = field(default_factory=list)  # MetaMo only
    transcendence_log:  list = field(default_factory=list)  # MetaMo only
    energy_log:         list = field(default_factory=list)
    survived:           bool = True

    # completion rate (CR) = minerals_collected / minerals_spawned
    def completion_rate(self) -> float:
        if self.minerals_spawned == 0:
            return 0.0
        return self.minerals_collected / self.minerals_spawned

    def srv_rate(self) -> float:
        if not self.srv_flags:
            return 0.0
        return sum(self.srv_flags) / len(self.srv_flags)

    def unsafe_rate(self) -> float:
        if not self.unsafe_flags:
            return 0.0
        return sum(self.unsafe_flags) / len(self.unsafe_flags)

    def recovery_time(self) -> float:
        """
        RT(t0) = min{τ ≥ 0 : m_{t0+τ} ∈ S for L consecutive steps}
        We use L=3.
        Returns average RT over all violation bouts.
        """
        L = 3
        flags = self.srv_flags or self.unsafe_flags
        if not flags:
            return 0.0

        rts = []
        i = 0
        while i < len(flags):
            if flags[i]:                     # violation starts
                t0 = i
                j  = i + 1
                consec_safe = 0
                while j < len(flags):
                    if not flags[j]:
                        consec_safe += 1
                        if consec_safe >= L:
                            rts.append(j - t0)
                            break
                    else:
                        consec_safe = 0
                    j += 1
                i = j + 1
            else:
                i += 1
        return float(np.mean(rts)) if rts else float(len(flags))

    def lava_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.lava_steps / self.total_steps


class MetricsCollector:
    """Accumulates EpisodeLogs across episodes and computes summary stats."""

    def __init__(self, label: str):
        self.label   = label
        self.episodes: list[EpisodeLog] = []

    def add(self, ep: EpisodeLog):
        self.episodes.append(ep)

    def summary(self) -> dict:
        n = len(self.episodes)
        if n == 0:
            return {}

        cr   = [e.completion_rate()  for e in self.episodes]
        lr   = [e.lava_rate()        for e in self.episodes]
        tr   = [e.total_reward       for e in self.episodes]
        srv  = [e.srv_rate()         for e in self.episodes]
        unsafe = [e.unsafe_rate()    for e in self.episodes]
        rt   = [e.recovery_time()    for e in self.episodes]

        return {
            "label":               self.label,
            "n_episodes":          n,
            "completion_rate":     {"mean": np.mean(cr),  "std": np.std(cr)},
            "lava_rate":           {"mean": np.mean(lr),  "std": np.std(lr)},
            "total_reward":        {"mean": np.mean(tr),  "std": np.std(tr)},
            "srv_rate":            {"mean": np.mean(srv), "std": np.std(srv)},
            "unsafe_rate":         {"mean": np.mean(unsafe), "std": np.std(unsafe)},
            "recovery_time":       {"mean": np.mean(rt),  "std": np.std(rt)},
        }
