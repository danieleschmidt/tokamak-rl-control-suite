"""
DisruptionPredictor: Classifies pre-disruption plasma states.

Uses threshold-based rules on key MHD stability indicators:
  - beta proximity to Troyon limit
  - q-profile approach to q=1 (sawtooth/kink instability)
  - density proximity to Greenwald limit

Can also be trained as a lightweight neural classifier on rollout data.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


# Default thresholds based on MHD physics
BETA_TROYON_FRACTION_WARN   = 0.7   # warn at 70% of Troyon limit (β = 0.07)
Q_KINK_THRESHOLD            = 1.3   # q < 1.3 → approaching kink instability zone
DENSITY_GREENWALD_FRACTION  = 0.8   # warn at 80% of simplified Greenwald density


class DisruptionPredictor:
    """
    Two-tier disruption prediction:

    1. Threshold classifier (physics-based, always available):
       - Computes disruption risk score from beta, q, and density
       - Returns binary label: 0 (safe) / 1 (pre-disruption)
       - Threshold at risk_threshold (default 0.5)

    2. Neural classifier (optional, trained from data):
       - Small MLP trained on labeled rollout states
       - Useful for learning subtle correlated signatures
    """

    def __init__(
        self,
        risk_threshold: float = 0.5,
        beta_troyon_limit: float = 0.10,
        use_neural: bool = False,
        obs_dim: int = 5,
    ):
        self.risk_threshold    = risk_threshold
        self.beta_troyon_limit = beta_troyon_limit
        self.use_neural        = use_neural

        self.neural_model: Optional[nn.Module] = None
        if use_neural:
            self.neural_model = self._build_neural_model(obs_dim)

    def _build_neural_model(self, obs_dim: int) -> nn.Module:
        """Lightweight binary classifier MLP."""
        model = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        return model

    def physics_risk_score(self, state: np.ndarray) -> float:
        """
        Physics-based disruption risk score in [0, 1].

        Args:
            state: (plasma_current, temperature, density, beta, q_profile)
        """
        # State indices
        beta = state[3]
        q    = state[4]
        n_e  = state[2]

        # Beta proximity to Troyon limit
        beta_frac = beta / self.beta_troyon_limit
        beta_risk = np.clip((beta_frac - BETA_TROYON_FRACTION_WARN) /
                            (1.0 - BETA_TROYON_FRACTION_WARN), 0.0, 1.0)

        # q approaching kink instability threshold
        q_risk = np.clip((Q_KINK_THRESHOLD - q) / (Q_KINK_THRESHOLD - 0.8), 0.0, 1.0)

        # Density approaching Greenwald limit (simplified: 2.0 = limit)
        density_limit = 2.0
        density_risk  = np.clip(
            (n_e / density_limit - DENSITY_GREENWALD_FRACTION) /
            (1.0 - DENSITY_GREENWALD_FRACTION),
            0.0, 1.0
        )

        # Weighted combination
        risk = 0.50 * beta_risk + 0.35 * q_risk + 0.15 * density_risk
        return float(np.clip(risk, 0.0, 1.0))

    def predict(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Predict whether plasma state is pre-disruptive.

        Returns:
            label:  0 (safe) or 1 (pre-disruption)
            score:  continuous risk score in [0, 1]
        """
        if self.use_neural and self.neural_model is not None:
            with torch.no_grad():
                obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                score = float(self.neural_model(obs_t).item())
        else:
            score = self.physics_risk_score(state)

        label = int(score >= self.risk_threshold)
        return label, score

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple states.

        Args:
            states: (N, 5) array of plasma states

        Returns:
            labels: (N,) binary array
            scores: (N,) float array
        """
        scores = np.array([self.physics_risk_score(s) for s in states])
        labels = (scores >= self.risk_threshold).astype(int)
        return labels, scores

    def train_neural(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> List[float]:
        """
        Train the neural classifier on labeled state data.

        Args:
            states: (N, obs_dim) array
            labels: (N,) binary labels (0=safe, 1=pre-disruption)
            n_epochs: training epochs
            lr: learning rate

        Returns:
            losses per epoch
        """
        if self.neural_model is None:
            self.neural_model = self._build_neural_model(states.shape[1])
            self.use_neural = True

        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        losses = []
        for _ in range(n_epochs):
            optimizer.zero_grad()
            preds = self.neural_model(X)
            loss  = criterion(preds, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    def evaluate(self, states: np.ndarray, labels: np.ndarray) -> dict:
        """Compute accuracy, precision, recall on labeled data."""
        pred_labels, scores = self.predict_batch(states)
        tp = int(((pred_labels == 1) & (labels == 1)).sum())
        fp = int(((pred_labels == 1) & (labels == 0)).sum())
        fn = int(((pred_labels == 0) & (labels == 1)).sum())
        tn = int(((pred_labels == 0) & (labels == 0)).sum())

        accuracy  = (tp + tn) / max(len(labels), 1)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)

        return {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
