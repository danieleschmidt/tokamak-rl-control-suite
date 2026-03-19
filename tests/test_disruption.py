"""Tests for DisruptionPredictor."""

import sys
import pytest
import numpy as np

sys.path.insert(0, "src")

from tokamak_rl.disruption import DisruptionPredictor


SAFE_STATE      = np.array([1.8, 8.0,  0.7, 0.03,  2.5], dtype=np.float32)
NOMINAL_STATE   = np.array([2.0, 10.0, 1.0, 0.05,  1.8], dtype=np.float32)
RISKY_STATE     = np.array([2.0, 14.0, 2.2, 0.095, 1.05], dtype=np.float32)
DISRUPTION_STATE = np.array([2.5, 16.0, 2.5, 0.13, 0.85], dtype=np.float32)


class TestDisruptionPredictor:

    def setup_method(self):
        self.predictor = DisruptionPredictor(risk_threshold=0.5)

    def test_safe_state_low_risk(self):
        _, score = self.predictor.predict(SAFE_STATE)
        assert score < 0.5, f"Safe state has high risk score: {score}"

    def test_disruption_state_high_risk(self):
        _, score = self.predictor.predict(DISRUPTION_STATE)
        assert score > 0.5, f"Disruption state has low risk score: {score}"

    def test_risky_state_classified_correctly(self):
        label, score = self.predictor.predict(RISKY_STATE)
        # Risky state should have higher score than nominal
        _, nominal_score = self.predictor.predict(NOMINAL_STATE)
        assert score > nominal_score, "Risky state should have higher risk than nominal"

    def test_risk_score_range(self):
        states = [SAFE_STATE, NOMINAL_STATE, RISKY_STATE, DISRUPTION_STATE]
        for s in states:
            _, score = self.predictor.predict(s)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_batch_predict_shape(self):
        states = np.stack([SAFE_STATE, NOMINAL_STATE, RISKY_STATE])
        labels, scores = self.predictor.predict_batch(states)
        assert labels.shape == (3,)
        assert scores.shape == (3,)
        assert labels.dtype == int
        assert scores.dtype == np.float64

    def test_batch_predict_ordering(self):
        """Disruption state should have higher risk than safe state in batch."""
        states = np.stack([SAFE_STATE, DISRUPTION_STATE])
        _, scores = self.predictor.predict_batch(states)
        assert scores[1] > scores[0], "Disruption state should score higher than safe"

    def test_evaluate_returns_metrics(self):
        states = np.stack([SAFE_STATE, SAFE_STATE, DISRUPTION_STATE, DISRUPTION_STATE])
        labels = np.array([0, 0, 1, 1])
        metrics = self.predictor.evaluate(states, labels)
        assert "accuracy"  in metrics
        assert "precision" in metrics
        assert "recall"    in metrics
        assert "f1"        in metrics
        assert metrics["accuracy"] >= 0.0
        assert metrics["accuracy"] <= 1.0

    def test_evaluate_perfect_labels(self):
        """Should achieve high metrics on clearly separated examples."""
        safe_states = np.tile(SAFE_STATE, (10, 1))
        risky_states = np.tile(DISRUPTION_STATE, (10, 1))
        states = np.vstack([safe_states, risky_states])
        labels = np.array([0]*10 + [1]*10)
        metrics = self.predictor.evaluate(states, labels)
        assert metrics["accuracy"] >= 0.7, f"Accuracy too low: {metrics['accuracy']}"

    def test_custom_threshold(self):
        """Higher threshold → more conservative (fewer pre-disruption calls)."""
        pred_strict  = DisruptionPredictor(risk_threshold=0.8)
        pred_lenient = DisruptionPredictor(risk_threshold=0.3)

        states = np.stack([NOMINAL_STATE, RISKY_STATE])
        labels_strict,  _ = pred_strict.predict_batch(states)
        labels_lenient, _ = pred_lenient.predict_batch(states)

        # Lenient should flag more states as pre-disruptive
        assert labels_lenient.sum() >= labels_strict.sum()

    def test_neural_model_build(self):
        """Neural predictor should build without error."""
        predictor = DisruptionPredictor(risk_threshold=0.5, use_neural=True, obs_dim=5)
        assert predictor.neural_model is not None
        label, score = predictor.predict(NOMINAL_STATE)
        assert 0.0 <= score <= 1.0

    def test_neural_training(self):
        """Neural classifier should train without error."""
        predictor = DisruptionPredictor(use_neural=True, obs_dim=5)
        states = np.vstack([
            np.tile(SAFE_STATE, (20, 1)),
            np.tile(DISRUPTION_STATE, (20, 1)),
        ])
        labels = np.array([0]*20 + [1]*20)
        losses = predictor.train_neural(states, labels, n_epochs=20)
        assert len(losses) == 20
        assert losses[-1] < losses[0] + 0.1  # Loss should not blow up
