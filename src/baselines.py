"""Step 12 — baseline models: XGBoost, LightGBM and a small MLP.

All three are CPU-friendly.  The goal is a first, honest signal that the data
is predictive — if these baselines do no better than chance, a giant
transformer won't magically help (and if they do well, we have a ceiling to
beat and a reference for leakage checks).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import FEATURES_DIR, MODELS_DIR, NUM_CLASSES, RANDOM_STATE, RESULTS_DIR  # noqa: E402

# label -> class index (sklearn / xgb / lgb expect 0..K-1)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
CLASS_NAMES = {0: "DOWN", 1: "NO_MOVE", 2: "UP"}

NON_FEATURE = {
    "time", "future_ret_1", "future_ret_2", "future_ret_3", "future_ret_5",
    "future_ret_10", "normalized_move", "label", "future_max_up",
    "future_max_down", "future_vol",
}


def load_split(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(FEATURES_DIR / f"{name}.csv")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].map(LABEL_MAP).to_numpy()
    return df, X, y, feature_cols


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {"model": name, "accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f1),
            "cm_down": int(cm[0].sum()), "cm_nomove": int(cm[1].sum()), "cm_up": int(cm[2].sum())}


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def train_xgb(Xtr, ytr, Xval, yval) -> object:
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="multi:softprob", num_class=NUM_CLASSES,
        eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE,
        early_stopping_rounds=30, tree_method="hist",
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
def train_lgb(Xtr, ytr, Xval, yval) -> object:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=800, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        n_jobs=-1, random_state=RANDOM_STATE, objective="multiclass", num_class=NUM_CLASSES,
        verbosity=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


# ---------------------------------------------------------------------------
# MLP (PyTorch, CPU)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3, n_class: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xval, yval, epochs: int = 30, batch_size: int = 512):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xval_s = scaler.transform(Xval).astype(np.float32)

    model = MLP(in_dim=Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(Xtr_s)
    yt = torch.from_numpy(ytr.astype(np.int64))
    Xv = torch.from_numpy(Xval_s)
    yv = torch.from_numpy(yval.astype(np.int64))

    best_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(Xv).argmax(1) == yv).float().mean().item()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def predict(model, X) -> np.ndarray:
    """Return predicted class indices for a model (xgb/lgb/mlp-or-tuple)."""
    if isinstance(model, tuple):  # (MLP, scaler)
        net, scaler = model
        Xs = scaler.transform(X.astype(np.float32))
        with torch.no_grad():
            return net(torch.from_numpy(Xs)).argmax(1).numpy()
    return model.predict(X)


def predict_proba(model, X) -> np.ndarray:
    if isinstance(model, tuple):
        net, scaler = model
        Xs = scaler.transform(X.astype(np.float32))
        with torch.no_grad():
            return torch.softmax(net(torch.from_numpy(Xs)), dim=1).numpy()
    return model.predict_proba(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 12 — baselines: XGBoost / LightGBM / MLP\n")

    _, Xtr, ytr, feature_cols = load_split("train")
    _, Xval, yval, _ = load_split("val")
    _, Xtest, ytest, _ = load_split("test")
    print(f"train={len(Xtr)}  val={len(Xval)}  test={len(Xtest)}  features={len(feature_cols)}")

    results = []
    models = {}

    for name, trainer in [("xgboost", train_xgb), ("lightgbm", train_lgb), ("mlp", train_mlp)]:
        t0 = time.time()
        print(f"\n[{name}] training ...")
        model = trainer(Xtr, ytr, Xval, yval)
        models[name] = model
        dt = time.time() - t0

        # Validation metrics (the split used for selection)
        yv_pred = predict(model, Xval)
        v_rep = evaluate(name, yval, yv_pred)
        v_rep["seconds"] = round(dt, 1)

        # Held-out test metrics (only reported once, for honesty)
        yt_pred = predict(model, Xtest)
        t_acc = float(accuracy_score(ytest, yt_pred))
        v_rep["test_accuracy"] = t_acc
        print(f"    val_acc={v_rep['accuracy']:.4f}  val_f1={v_rep['f1']:.4f}  "
              f"test_acc={t_acc:.4f}  time={dt:.1f}s")
        results.append(v_rep)

        # Persist
        if name == "mlp":
            torch.save(model[0].state_dict(), MODELS_DIR / f"mlp.pt")
            joblib.dump(model[1], MODELS_DIR / f"mlp_scaler.joblib")
        else:
            joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "baselines.csv", index=False)
    print("\nbaseline results (macro-averaged):")
    print(res_df.to_string(index=False))
    print(f"\nsaved -> {RESULTS_DIR / 'baselines.csv'}")
    print("STEP 12 DONE.")


if __name__ == "__main__":
    main()

