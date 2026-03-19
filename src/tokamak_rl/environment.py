"""
TokamakEnv: Simplified plasma control environment for RL research.

State space (5D):
  - plasma_current [MA]:        target ~2.0 MA
  - temperature [keV]:          target ~10 keV
  - density [1e20 m^-3]:        target ~1.0
  - beta (plasma pressure ratio): critical stability parameter; disruption if too high
  - q_profile [dimensionless]:  safety factor; q < 1 at core → sawteeth/disruption risk

Action space (3D, normalized [-1, 1]):
  - heating_power:   additional NBI/ICRH power (affects temperature & beta)
  - gas_injection:   fueling (affects density; too high → disruption)
  - magnetic_field_trim: fine-tuning B-field (affects q-profile and beta)

Physics model (simplified MHD stability):
  - Disruption risk combines high beta and low q (Greenwald-like)
  - Episode terminates on disruption (disruption_risk > DISRUPTION_THRESHOLD)
  - Reward = -disruption_risk - energy_loss_penalty
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


DISRUPTION_THRESHOLD = 0.9

# State indices
IDX_CURRENT = 0
IDX_TEMP = 1
IDX_DENSITY = 2
IDX_BETA = 3
IDX_Q = 4

# Action indices
IDX_HEATING = 0
IDX_GAS = 1
IDX_BTRIM = 2


@dataclass
class PlasmaState:
    plasma_current: float   # MA
    temperature: float       # keV
    density: float           # 1e20 m^-3
    beta: float              # dimensionless (normalized plasma pressure)
    q_profile: float         # safety factor (q on axis)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.plasma_current,
            self.temperature,
            self.density,
            self.beta,
            self.q_profile,
        ], dtype=np.float32)


class TokamakEnv:
    """
    Simplified tokamak plasma control environment.

    Observation: 5D continuous state (plasma_current, temperature, density, beta, q)
    Action:      3D continuous [-1, 1] (heating_power, gas_injection, mag_field_trim)

    Episode terminates when:
      - disruption_risk > DISRUPTION_THRESHOLD (plasma disruption)
      - max_steps reached
    """

    # Physical operating bounds
    STATE_LOW  = np.array([0.5, 2.0, 0.2, 0.01, 0.8], dtype=np.float32)
    STATE_HIGH = np.array([4.0, 25.0, 3.0, 0.15, 5.0], dtype=np.float32)

    # Nominal operating point
    STATE_NOMINAL = np.array([2.0, 10.0, 1.0, 0.05, 1.8], dtype=np.float32)

    # Action scale (physical units per normalized unit)
    ACTION_SCALE = np.array([5.0, 0.3, 0.05], dtype=np.float32)  # MW, 1e20/m3/s, T

    def __init__(self, max_steps: int = 200, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state: Optional[np.ndarray] = None
        self._step_count = 0
        self._episode_disruptions = 0

        # Spaces (bounds)
        self.obs_dim = 5
        self.act_dim = 3
        self.observation_space_low = self.STATE_LOW.copy()
        self.observation_space_high = self.STATE_HIGH.copy()
        self.action_space_low = np.full(self.act_dim, -1.0, dtype=np.float32)
        self.action_space_high = np.full(self.act_dim, 1.0, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset to nominal state with small perturbation."""
        noise = self.rng.normal(0, 0.05, size=5).astype(np.float32)
        self.state = np.clip(
            self.STATE_NOMINAL * (1 + noise),
            self.STATE_LOW,
            self.STATE_HIGH,
        )
        self._step_count = 0
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action and advance plasma state by one control timestep (~10ms).

        Returns:
            obs:     next state observation
            reward:  scalar reward
            done:    episode termination flag
            info:    diagnostics dict
        """
        assert self.state is not None, "Call reset() before step()"
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Unpack state
        I_p  = self.state[IDX_CURRENT]    # plasma current (MA)
        T_e  = self.state[IDX_TEMP]       # electron temperature (keV)
        n_e  = self.state[IDX_DENSITY]    # electron density
        beta = self.state[IDX_BETA]       # normalized beta
        q    = self.state[IDX_Q]          # q on axis

        # Unpack action (scaled to physical units)
        P_heat  = action[IDX_HEATING]  * self.ACTION_SCALE[0]   # MW additional heating
        Gamma_n = action[IDX_GAS]      * self.ACTION_SCALE[1]   # density source
        dB      = action[IDX_BTRIM]    * self.ACTION_SCALE[2]   # field trim (T)

        dt = 0.01  # 10 ms control timestep

        # --- Simplified MHD dynamics ---

        # Temperature evolves with heating and radiation losses
        # dT/dt ∝ P_heat/n - T^0.5 (energy confinement τ_E ~ T^0.5 scaling)
        tau_E = 0.3 * T_e**0.5 / max(n_e, 0.1)   # energy confinement time (s)
        dT = dt * (P_heat / max(n_e, 0.1) - T_e / tau_E) * 0.05
        T_e = T_e + dT + self.rng.normal(0, 0.05)

        # Density: gas injection plus recycling; radiative collapse if too high
        dn = dt * (Gamma_n - 0.1 * n_e) + self.rng.normal(0, 0.02)
        n_e = n_e + dn

        # Beta: ∝ n*T / B^2 (simplified); field trim lowers beta
        # Equilibrium beta from pressure; slow towards new equilibrium
        beta_eq = 0.04 * n_e * T_e / (I_p + 0.5)  # simplified pressure balance
        beta = beta + dt * (beta_eq - beta) / 0.1 - 0.001 * dB + self.rng.normal(0, 0.003)

        # q profile: safety factor decreases with current, increases with B trim
        q_eq = 0.9 + 1.5 / max(I_p, 0.5) + 0.3 * dB
        q = q + dt * (q_eq - q) / 0.2 + self.rng.normal(0, 0.02)

        # Plasma current: slow ohmic decay + correction; bounded by external coils
        I_p = I_p + dt * (-0.05 * I_p + 0.1 * np.tanh(action[IDX_BTRIM])) + self.rng.normal(0, 0.01)

        # Clip to physical bounds
        I_p  = np.clip(I_p,  self.STATE_LOW[IDX_CURRENT], self.STATE_HIGH[IDX_CURRENT])
        T_e  = np.clip(T_e,  self.STATE_LOW[IDX_TEMP],    self.STATE_HIGH[IDX_TEMP])
        n_e  = np.clip(n_e,  self.STATE_LOW[IDX_DENSITY], self.STATE_HIGH[IDX_DENSITY])
        beta = np.clip(beta, self.STATE_LOW[IDX_BETA],    self.STATE_HIGH[IDX_BETA])
        q    = np.clip(q,    self.STATE_LOW[IDX_Q],       self.STATE_HIGH[IDX_Q])

        self.state = np.array([I_p, T_e, n_e, beta, q], dtype=np.float32)

        # --- Disruption risk model ---
        # Risk increases with high beta (Troyon limit ~0.10), low q (q < 1 ⟹ kink instability),
        # and high density (Greenwald limit analogue)
        troyon_limit = 0.10
        beta_risk    = np.clip((beta / troyon_limit - 0.5) * 2.0, 0.0, 1.0)
        q_risk       = np.clip((1.2 - q) * 2.0, 0.0, 1.0)  # high risk when q < 1.2
        density_risk = np.clip((n_e / 2.5 - 0.5) * 2.0, 0.0, 1.0)
        disruption_risk = np.clip(
            0.5 * beta_risk + 0.35 * q_risk + 0.15 * density_risk,
            0.0, 1.0
        )

        # Energy loss (deviation from nominal temperature = wasted energy)
        energy_loss = abs(T_e - 10.0) / 10.0

        # --- Reward ---
        reward = float(-disruption_risk - 0.3 * energy_loss)

        # --- Termination ---
        disrupted = bool(disruption_risk > DISRUPTION_THRESHOLD)
        self._step_count += 1
        timeout = self._step_count >= self.max_steps
        done = disrupted or timeout

        info = {
            "disrupted": disrupted,
            "disruption_risk": float(disruption_risk),
            "energy_loss": float(energy_loss),
            "beta": float(beta),
            "q_profile": float(q),
            "step": self._step_count,
        }
        return self.state.copy(), reward, done, info

    def disruption_risk(self, state: Optional[np.ndarray] = None) -> float:
        """Compute disruption risk for a given state (or current state)."""
        s = state if state is not None else self.state
        if s is None:
            return 0.0
        beta = s[IDX_BETA]
        q    = s[IDX_Q]
        n_e  = s[IDX_DENSITY]
        beta_risk    = np.clip((beta / 0.10 - 0.5) * 2.0, 0.0, 1.0)
        q_risk       = np.clip((1.2 - q) * 2.0, 0.0, 1.0)
        density_risk = np.clip((n_e / 2.5 - 0.5) * 2.0, 0.0, 1.0)
        return float(np.clip(0.5 * beta_risk + 0.35 * q_risk + 0.15 * density_risk, 0.0, 1.0))

    def render(self) -> str:
        """Text representation of plasma state."""
        if self.state is None:
            return "TokamakEnv (not initialized)"
        s = self.state
        risk = self.disruption_risk()
        return (
            f"Ip={s[IDX_CURRENT]:.2f}MA  Te={s[IDX_TEMP]:.1f}keV  "
            f"ne={s[IDX_DENSITY]:.2f}  β={s[IDX_BETA]:.4f}  q={s[IDX_Q]:.2f}  "
            f"risk={risk:.3f}"
        )
