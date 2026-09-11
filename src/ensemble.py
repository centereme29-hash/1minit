"""Step 21 — ensemble: weighted average of the top models' probabilities."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import MODELS_DIR  # noqa: E402
from src.baselines import MLP, predict_proba  # noqa: E402


def load_models() -> dict:
    models: dict = {}
    models["xgboost"] = joblib.load(MODELS_DIR / "xgboost.joblib")
    models["lightgbm"] = joblib.load(MODELS_DIR / "lightgbm.joblib")

    scaler = joblib.load(MODELS_DIR / "mlp_scaler.joblib")
    in_dim = int(getattr(scaler, "n_features_in_", 53))
    net = MLP(in_dim=in_dim)
    net.load_state_dict(torch.load(MODELS_DIR / "mlp.pt", map_location="cpu"))
    net.eval()
    models["mlp"] = (net, scaler)
    return models


def ensemble_proba(models: dict, X: np.ndarray, weights: dict | None = None) -> np.ndarray:
    if weights is None:
        weights = {k: 1.0 for k in models}
    probs = [weights.get(k, 1.0) * predict_proba(m, X) for k, m in models.items()]
    wsum = sum(weights.get(k, 1.0) for k in models)
    return np.sum(np.stack(probs), axis=0) / wsum


def ensemble_predict(models: dict, X: np.ndarray, weights: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    probs = ensemble_proba(models, X, weights)
    return probs.argmax(axis=1), probs.max(axis=1)
