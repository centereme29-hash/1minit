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
