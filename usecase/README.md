# MetaMo GridWorld — Comparative RL Simulation

Side-by-side pygame simulation comparing a **Baseline Q-learning agent** against a
**MetaMo-enhanced Q-learning agent** in a 10×10 hazard gridworld.

## Project Structure

```
metamo_gridworld/
│
├── simulation.py            ← Main entry-point (run this)
│
├── environment/
│   └── gridworld.py         ← 10×10 GridWorld (random red hazards, mixed mineral zones)
│
├── agents/
│   ├── baseline_agent.py    ← Tabular Q-learning, no motivational layer
│   └── metamo_agent.py      ← Q-learning + MetaMo motivational regulation
│
├── metamo/
│   ├── state.py             ← MotivationalState vector + safe-region S
│   └── core.py              ← A(), D(), F() — appraisal / decision / update
│
├── metrics/
│   └── collector.py         ← EpisodeLog, MetricsCollector, SRV, RT
│
├── assets/
│   ├── cat.jpg           ← Agent sprite
│   └── clank.wav            ← Hazard sound effect
│
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          
pip install -r requirements.txt
python simulation.py
```

## Controls

| Key      | Action              |
|----------|---------------------|
| `SPACE`  | Pause / Resume      |
| `R`      | Reset episode       |
| `F/UP`   | Speed up            |
| `S/DOWN` | Slow down           |
| `Q/ESC`  | Quit                |

## Environment

```
(0,0)  [Agent Start]
  │    ... Safe Open Space ...
  │
  └───[Random red lava hazards each episode]
      [Minerals sometimes spawn near danger and sometimes in safe cells]
```

Each training and evaluation episode is capped at 100 environment steps.

## MetaMo Motivational State

The MetaMo agent uses the root `MotivationalState(G, M)`.

The dashboard shows the current and consensus-target values for:

- `G_IND`: individuation / safety preservation
- `G_TRANS`: transcendence / exploratory growth
- `M_THRESHOLD`: displayed as safety threshold
- `M_AROUSAL`: displayed as arousal

**Internal safe region S:**  `G_IND ≥ THETA_SAFE  AND  ||G|| ≤ G_MAX`

| Variable          | Description                              |
|-------------------|------------------------------------------|
| `energy_drive`    | Homeostatic hunger for minerals          |
| `safety_threshold`| Structural integrity / risk-avoidance    |
| `arousal`         | Affective stress — spikes near lava      |

## Metrics

| Metric            | Formula                                        |
|-------------------|------------------------------------------------|
| Completion Rate   | minerals_collected / minerals_spawned          |
| SRV Rate          | MetaMo: (1/T) Σ I[m_t ∉ S]; baseline: environment danger-band proxy |
| Recovery Time     | min{τ ≥ 0 : m_{t0+τ} ∈ S for 3 steps}          |
| Lava Rate         | lava_steps / total_steps                       |
| Unsafe-Zone Rate  | steps in lava or the lava-adjacent danger band / total_steps |
