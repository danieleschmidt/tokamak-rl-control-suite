# Tokamak RL Control Suite

Reinforcement learning for plasma control in tokamak fusion reactors, with a focus on MHD stability, disruption avoidance, and safe policy optimization.

---

## The Problem

Tokamaks confine hot plasma in a magnetic field to achieve nuclear fusion. The plasma is inherently unstable — small deviations in pressure, density, or magnetic configuration can trigger a **disruption**: a catastrophic loss of plasma confinement that can damage the vessel wall in milliseconds.

Control is hard because:
- The state space is high-dimensional and partially observable
- Safety constraints are hard (disruption = episode over, potentially hardware damage)
- The physics couples temperature, density, pressure, and magnetic field in nonlinear ways
- Control must operate in real-time (~10 ms timesteps)

This suite provides a simplified but physics-grounded environment and a PPO-based safe RL controller.

---

## Architecture

```
TokamakEnv          — Plasma dynamics + MHD stability model
SafePolicyNetwork   — Actor-critic MLP with per-actuator safety projection
PPOAgent            — Proximal Policy Optimization training loop
DisruptionPredictor — Physics-based pre-disruption state classifier
```

### State Space (5D)

| Variable | Units | Description |
|----------|-------|-------------|
| `plasma_current` (Iₚ) | MA | Total toroidal plasma current |
| `temperature` (Tₑ) | keV | Electron temperature |
| `density` (nₑ) | 10²⁰ m⁻³ | Electron density |
| `beta` (β) | — | Normalized plasma pressure (β = p / (B²/2μ₀)) |
| `q_profile` (q) | — | Safety factor on axis |

### Action Space (3D, normalized to [-1, 1])

| Actuator | Physical scale | Effect |
|----------|---------------|--------|
| `heating_power` | ±5 MW | NBI/ICRH — raises temperature, increases beta |
| `gas_injection` | ±0.3 ×10²⁰ m⁻³/s | Fueling — increases density |
| `magnetic_field_trim` | ±0.05 T | Fine B-field adjustment — modulates q and beta |

### Physics (Simplified MHD Stability)

Disruption risk is a weighted combination of three instability indicators:

```
risk = 0.50 × β_risk + 0.35 × q_risk + 0.15 × density_risk
```

- **β_risk**: proximity to the Troyon limit (β > 0.10 → Ballooning instability)
- **q_risk**: proximity to q = 1 on axis (q < 1.2 → Kink instability / sawteeth)
- **density_risk**: proximity to the Greenwald density limit

**Reward:**
```
r = −disruption_risk − 0.3 × |Tₑ − 10 keV| / 10
```

**Hard constraint:** if `disruption_risk > 0.9`, episode terminates (disruption event).

---

## Installation

```bash
git clone https://github.com/danieleschmidt/tokamak-rl-control-suite
cd tokamak-rl-control-suite
pip install -e .
```

Requires Python ≥ 3.9, PyTorch ≥ 1.12, NumPy ≥ 1.21.

---

## Quick Start

```python
from tokamak_rl import TokamakEnv, PPOAgent, DisruptionPredictor

# Create environment
env = TokamakEnv(max_steps=200, seed=42)

# Train PPO agent
agent = PPOAgent(env, lr_actor=3e-4, n_epochs=4)
stats = agent.train(n_episodes=100)

# Classify plasma states
predictor = DisruptionPredictor(risk_threshold=0.5)
risky_state = [2.0, 14.0, 2.2, 0.095, 1.05]
label, score = predictor.predict(risky_state)
# → label=1 (pre-disruption), score=0.74
```

Run the full demo:

```bash
python demo.py
```

---

## SafePolicyNetwork

The actor-critic network includes a **safety projection layer** that hard-clips actions to per-actuator bounds before they're sent to the environment:

```python
# Prevents beta runaway from excessive heating
heating_max =  0.9   # normalized

# Prevents Greenwald limit violation from over-fueling
gas_max     =  0.7   # normalized
```

This is a form of **constrained policy optimization** — the policy network learns within a pre-defined safe action manifold rather than learning the constraints from scratch.

---

## DisruptionPredictor

Two prediction modes:

1. **Physics-based (default):** threshold classifier on beta, q, and density proximity to known limits — no training required, interpretable

2. **Neural classifier:** small MLP trained on labeled rollout data — learns correlated disruption signatures

```python
# Train neural predictor on rollout data
predictor = DisruptionPredictor(use_neural=True)
predictor.train_neural(states, labels, n_epochs=50)
metrics = predictor.evaluate(test_states, test_labels)
# → {'accuracy': 0.91, 'precision': 0.88, 'recall': 0.93, 'f1': 0.90}
```

---

## Tests

```bash
python -m pytest tests/ -v
```

38 tests covering environment physics, policy safety bounds, PPO training, and disruption prediction.

---

## Background: Why RL for Tokamaks?

Classical PID controllers struggle with tokamak plasma because:
- The plant is **nonlinear** (MHD equations)
- Constraints are **hard** (disruption causes hardware damage)
- Optimal operating points are **multi-dimensional** (current, shape, profiles)
- Real machines like ITER and SPARC will push to plasma regimes where classical models break down

Recent work (DeepMind/EPFL 2022 on TCV, Google/TAE 2023) has demonstrated RL successfully learning tokamak shape control directly from simulation. This suite provides a self-contained research sandbox for experimenting with constrained RL methods in this domain.

---

## Project Structure

```
src/tokamak_rl/
├── __init__.py          # Public API
├── environment.py       # TokamakEnv — plasma dynamics
├── policy.py            # SafePolicyNetwork — actor-critic + safety projection
├── ppo.py               # PPOAgent — training loop
└── disruption.py        # DisruptionPredictor — pre-disruption classifier
tests/
├── test_environment.py
├── test_policy.py
├── test_ppo.py
└── test_disruption.py
demo.py                  # End-to-end training demo
```

---

## License

MIT
