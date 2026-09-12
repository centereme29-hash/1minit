"""Step 21 — ensemble: weighted average of the top models' probabilities."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import CLASS_NAMES, MODELS_DIR  # noqa: E402
from src.baselines import MLP, predict_proba  # noqa: E402


class TransformerWrapper:
    """Wrapper that makes TransformerPredictor compatible with predict_proba interface.

    The transformer's predict_proba returns (n - context + 1, n_classes), but the
    baseline models return (n, n_classes). This wrapper handles single-row prediction
    by padding with context from the input array when needed, and handles feature
    mismatch by selecting only the features the transformer was trained on.
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self.context_len = predictor.cfg.context_len
        self.required_features = set(predictor.feature_cols)

    def predict_proba(self, X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        """Return probabilities with shape (n, n_classes).

        For single-row input, uses context from the provided array (pads with
        the last row if insufficient context is available).

        Args:
            X: Feature matrix of shape (n, n_features)
            feature_names: Optional list of feature names corresponding to X columns.
                          If provided, only the features the transformer was trained on
                          will be used. If not provided, assumes X already has the
                          correct features.
        """
        if feature_names is not None:
            # Select only the features the transformer was trained on
            feat_idx = [feature_names.index(f) for f in self.predictor.feature_cols
                       if f in feature_names]
            if len(feat_idx) != len(self.predictor.feature_cols):
                missing = set(self.predictor.feature_cols) - set(feature_names)
                raise ValueError(f"Transformer requires features not in input: {missing}")
            X = X[:, feat_idx]

        n = len(X)
        n_classes = self.predictor.cfg.n_classes
        
        if n < self.context_len:
            # Not enough history for the transformer.
            # Return uniform probabilities to indicate uncertainty.
            # This ensures the ensemble still works, just with a weak transformer signal.
            return np.full((n, n_classes), 1.0 / n_classes, dtype=np.float32)
        
        # For batch processing, return predictions for all rows where possible
        # The transformer can only predict for rows from context_len-1 onwards
        probs = self.predictor.predict_proba(X)  # shape (n - context_len + 1, n_classes)
        # Pad with uniform probabilities for rows that can't be predicted
        pad_rows = self.context_len - 1
        if pad_rows > 0:
            padding = np.full((pad_rows, n_classes), 1.0 / n_classes, dtype=np.float32)
            probs = np.concatenate([padding, probs], axis=0)
        return probs  # shape (n, n_classes)


def load_models() -> dict:
    models: dict = {}
    models["xgboost"] = joblib.load(MODELS_DIR / "xgboost.joblib")
    models["lightgbm"] = joblib.load(MODELS_DIR / "lightgbm.joblib")

    scaler = joblib.load(MODELS_DIR / "mlp_scaler.joblib")
    in_dim = int(getattr(scaler, "n_features_in_", 53))
    net = MLP(in_dim=in_dim)
    net.load_state_dict(torch.load(MODELS_DIR / "mlp.pt", map_location="cpu", weights_only=True))
    net.eval()
    models["mlp"] = (net, scaler)

    # Load transformer as 4th model (if available)
    try:
        predictor = load_transformer_predictor()
        models["transformer"] = TransformerWrapper(predictor)
    except Exception:
        pass  # Transformer not trained yet

    return models


def ensemble_proba(models: dict, X: np.ndarray, weights: dict | None = None,
                  feature_names: list[str] | None = None) -> np.ndarray:
    """Weighted average of model probabilities.

    Works with all model types: XGBoost/LightGBM (sklearn), MLP (tuple of net+scaler),
    and Transformer (TransformerWrapper).
    """
    if weights is None:
        weights = {k: 1.0 for k in models}
    probs = [weights.get(k, 1.0) * predict_proba(m, X, feature_names=feature_names)
             for k, m in models.items()]
    wsum = sum(weights.get(k, 1.0) for k in models)
    return np.sum(np.stack(probs), axis=0) / wsum


def ensemble_predict(models: dict, X: np.ndarray, weights: dict | None = None,
                     feature_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted class indices, confidence) for each row in X."""
    probs = ensemble_proba(models, X, weights, feature_names=feature_names)
    return probs.argmax(axis=1), probs.max(axis=1)


def load_transformer_predictor():
    """Load the trained multi-timeframe transformer (if present)."""
    from src.train import TransformerPredictor  # local import avoids a cycle
    return TransformerPredictor.from_disk()


def gated_action(probs: np.ndarray, gate: float) -> tuple[np.ndarray, np.ndarray]:
    """Map direction probabilities to LONG / SHORT / NO_TRADE.

    Works for 2-class (DOWN/UP) and 3-class (DOWN/NO_MOVE/UP).  Emits NO_TRADE
    when confidence < gate or the argmax class is NO_MOVE (3-class only).
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    actions: list[str] = []
    for p, c in zip(pred, conf):
        cls = CLASS_NAMES.get(int(p), "")
        if c < gate or cls == "NO_MOVE":
            actions.append("NO_TRADE")
        elif cls == "UP":
            actions.append("LONG")
        elif cls == "DOWN":
            actions.append("SHORT")
        else:
            actions.append("NO_TRADE")
    return np.asarray(actions), conf
