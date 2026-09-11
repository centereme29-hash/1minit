"""Step 6D/6E — trainer for the multi-timeframe transformer.

Loads the synchronized feature/target matrix, standardizes features, builds
sliding multi-timeframe windows, and trains ``MultiTimeframeTransformer`` with a
weighted multi-head loss.  The training loop is factored so ``evolution_gpu.py``
can reuse it.

Usage::

    python src/train.py --epochs 5 --batch-size 128 --d-model 256 --layers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    FEATURES_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_RETURNS,
)
from src.dataset import build_matrix  # noqa: E402
from src.baselines import LABEL_MAP  # noqa: E402
from src.model import (  # noqa: E402
    ModelConfig,
    MultiTimeframeTransformer,
    count_params,
    multihead_loss,
)

RETURN_COLS = [f"future_ret_{r}" for r in TARGET_RETURNS]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_training_data() -> tuple[np.ndarray, dict, list[str], pd.Series]:
    """Return standardized-ready feature matrix + multi-head target dict."""
    m, feature_cols = build_matrix()
    X = m[feature_cols].to_numpy(dtype=np.float32)
    targets = {
        "direction": m["label"].map(LABEL_MAP).to_numpy(dtype=np.int64),
        "returns": m[RETURN_COLS].to_numpy(dtype=np.float32),
        "vol": m["future_vol"].to_numpy(dtype=np.float32),
        "max_up": m["future_max_up"].to_numpy(dtype=np.float32),
        "max_down": m["future_max_down"].to_numpy(dtype=np.float32),
    }
    return X, targets, feature_cols, m["time"]


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, targets: dict, context_len: int, indices: np.ndarray) -> None:
        # Keep everything as numpy to avoid holding multiple full-size torch tensors
        # (torch.from_numpy(X) + all targets + indices previously OOMed at ~268k windows).
        self.X = X
        self.targets = targets
        self.context = context_len
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        end = int(self.indices[i])
        start = end - self.context + 1
        x = torch.from_numpy(self.X[start:end + 1].copy())
        tgt = {k: torch.as_tensor(v[end], dtype=torch.int64 if k == "direction" else torch.float32)
               for k, v in self.targets.items()}
        return x, tgt


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_model(
    cfg: ModelConfig,
    X: np.ndarray,
    targets: dict,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_name: str = "adamw",
    device: torch.device | None = None,
    patience: int = 0,
) -> tuple[MultiTimeframeTransformer, StandardScaler, dict]:
    device = device or get_device()
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    scaler = StandardScaler().fit(X[train_indices])
    Xs = scaler.transform(X).astype(np.float32)

    train_ds = SequenceDataset(Xs, targets, cfg.context_len, train_indices)
    val_ds = SequenceDataset(Xs, targets, cfg.context_len, val_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiTimeframeTransformer(cfg).to(device)
    opt = _make_optimizer(optimizer_name, model.parameters(), lr, weight_decay)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, tgt in train_loader:
            x = x.to(device)
            tgt = {k: v.to(device) for k, v in tgt.items()}
            opt.zero_grad()
            out = model(x)
            loss, _ = multihead_loss(out, tgt, cfg)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.size(0)

        model.eval()
        correct = 0
        n = 0
        with torch.no_grad():
            for x, tgt in val_loader:
                x = x.to(device)
                preds = model(x)["direction"].argmax(1).cpu()
                correct += int((preds == tgt["direction"]).sum())
                n += len(preds)
        val_acc = correct / max(1, n)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if patience and epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    metrics = run_inference(model, cfg, Xs, targets, val_indices, device, batch_size)
    metrics["val_acc"] = best_val_acc
    return model, scaler, metrics


def run_inference(model: MultiTimeframeTransformer, cfg: ModelConfig, Xs: np.ndarray,
                  targets: dict, indices: np.ndarray, device: torch.device,
                  batch_size: int = 256) -> dict:
    """Return probabilities, predictions and regression MAEs over ``indices``."""
    ds = SequenceDataset(Xs, targets, cfg.context_len, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs_list, preds_list, true_list = [], [], []
    mae_ret, mae_vol, mae_up, mae_down = 0.0, 0.0, 0.0, 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, tgt in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out["direction"], dim=1).cpu().numpy()
            probs_list.append(p)
            preds_list.append(p.argmax(1))
            true_list.append(tgt["direction"].numpy())
            r = tgt["returns"].numpy()
            mae_ret += float(np.abs(out["returns"].cpu().numpy() - r).sum())
            mae_vol += float(np.abs(out["vol"].cpu().numpy() - tgt["vol"].numpy()).sum())
            mae_up += float(np.abs(out["max_up"].cpu().numpy() - tgt["max_up"].numpy()).sum())
            mae_down += float(np.abs(out["max_down"].cpu().numpy() - tgt["max_down"].numpy()).sum())
            n += len(p)

    probs = np.concatenate(probs_list)
    preds = np.concatenate(preds_list)
    ytrue = np.concatenate(true_list)
    acc = float(accuracy_score(ytrue, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(ytrue, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "return_mae": float(mae_ret / max(1, n)),
        "vol_mae": float(mae_vol / max(1, n)),
        "max_up_mae": float(mae_up / max(1, n)),
        "max_down_mae": float(mae_down / max(1, n)),
        "probs": probs,
        "preds": preds,
        "ytrue": ytrue,
    }


def chrono_indices(n: int, train_ratio: float = 0.6, val_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Return train / validation row indices for a chronological split."""
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return np.arange(0, train_end), np.arange(train_end, val_end)


class TransformerPredictor:
    """Loads a trained transformer and runs sliding-window direction inference."""

    def __init__(self, model: MultiTimeframeTransformer, cfg: ModelConfig,
                 mean: np.ndarray, scale: np.ndarray, feature_cols: list[str],
                 device: torch.device | None = None) -> None:
        self.model = model
        self.cfg = cfg
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self.feature_cols = list(feature_cols)
        self.device = device or get_device()
        self.model.to(self.device).eval()

    @classmethod
    def from_disk(cls, device: torch.device | None = None) -> "TransformerPredictor":
        with open(MODELS_DIR / "transformer_config.json") as fh:
            meta = json.load(fh)
        cfg = ModelConfig(**meta["cfg"])
        model = MultiTimeframeTransformer(cfg)
        model.load_state_dict(torch.load(MODELS_DIR / "transformer.pt", map_location="cpu", weights_only=True))
        mean = np.load(MODELS_DIR / "transformer_mean.npy")
        scale = np.load(MODELS_DIR / "transformer_scaler.npy")
        return cls(model, cfg, mean, scale, meta["feature_cols"], device)

    def standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) / self.scale

    def predict_proba(self, X_contig: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Return direction probabilities for each row with a full context window.

        ``X_contig`` is a chronological, standardized feature matrix.  Output has
        shape ``(len(X_contig) - context_len + 1, n_classes)``.
        """
        X = np.asarray(X_contig, dtype=np.float32)
        ctx = self.cfg.context_len
        n_out = len(X) - ctx + 1
        if n_out <= 0:
            return np.zeros((0, self.cfg.n_classes), dtype=np.float32)
        probs = np.zeros((n_out, self.cfg.n_classes), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n_out, batch_size):
                idx = np.arange(start, min(start + batch_size, n_out))
                windows = np.stack([X[t:t + ctx] for t in idx])
                out = self.model(torch.from_numpy(windows).to(self.device))
                probs[idx] = torch.softmax(out["direction"], dim=1).cpu().numpy()
        return probs


def _model_defaults() -> dict:
    """Model hyperparameters from config.yaml (fall back to ModelConfig defaults)."""
    from src.params import load_config  # local import
    m = load_config().get("model") or {}
    return {
        "d_model": int(m.get("d_model", 512)),
        "layers": int(m.get("layers", 8)),
        "heads": int(m.get("heads", 8)),
        "dropout": float(m.get("dropout", 0.15)),
        "lr": float(m.get("lr", 1e-4)),
        "weight_decay": float(m.get("weight_decay", 0.01)),
        "seq_len": int(m.get("seq_len", 30)),
        "branches": m.get("branches", {"1m": 30, "5m": 36, "15m": 32}),
    }


def main() -> None:
    d = _model_defaults()
    parser = argparse.ArgumentParser(description="Train multi-timeframe transformer (6D/6E)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-train-windows", type=int, default=20000,
                        help="cap on training windows (speed vs coverage)")
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    args = parser.parse_args()

    device = get_device()
    print(f"device = {device}")
    X, targets, feature_cols, _ = build_training_data()
    cfg = ModelConfig(
        n_features=len(feature_cols),
        d_model=args.d_model if args.d_model is not None else d["d_model"],
        layers=args.layers if args.layers is not None else d["layers"],
        heads=args.heads if args.heads is not None else d["heads"],
        dropout=args.dropout if args.dropout is not None else d["dropout"],
        seq_len=d["seq_len"],
        branches=d["branches"],
        n_returns=len(RETURN_COLS),
        n_classes=NUM_CLASSES,
    )
    lr = args.lr if args.lr is not None else d["lr"]
    weight_decay = args.weight_decay if args.weight_decay is not None else d["weight_decay"]
    print(f"features={len(feature_cols)}  context_len={cfg.context_len}  "
          f"params={count_params(MultiTimeframeTransformer(cfg)):,}")

    n = len(X)
    first = cfg.context_len - 1
    train_idx = np.arange(first, int(n * 0.6))
    val_idx = np.arange(int(n * 0.6), int(n * 0.8))
    if len(train_idx) > args.max_train_windows:
        train_idx = np.linspace(first, train_idx[-1], args.max_train_windows).astype(int)

    model, scaler, metrics = train_model(
        cfg, X, targets, train_idx, val_idx,
        epochs=args.epochs, batch_size=args.batch_size, lr=lr,
        weight_decay=weight_decay, optimizer_name=args.optimizer, device=device,
    )

    print(f"val acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
          f"return_mae={metrics['return_mae']:.6f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "transformer.pt")
    np.save(MODELS_DIR / "transformer_scaler.npy", scaler.scale_)
    np.save(MODELS_DIR / "transformer_mean.npy", scaler.mean_)
    with open(MODELS_DIR / "transformer_config.json", "w") as fh:
        json.dump({"cfg": cfg.__dict__, "feature_cols": feature_cols}, fh, indent=2)
    with open(RESULTS_DIR / "transformer_metrics.json", "w") as fh:
        json.dump({k: v for k, v in metrics.items() if k not in ("probs", "preds", "ytrue")},
                  fh, indent=2)
    print(f"saved -> {MODELS_DIR / 'transformer.pt'}")


if __name__ == "__main__":
    main()

