"""Step 6F — GPU evolution engine for transformer architectures.

Evolves the multi-timeframe transformer.  Generation 0 samples a population of
small models, scores each on VALIDATION-A, keeps a diverse top-K, then produces
the next generation via mutation + crossover.  Every individual is logged to
``results/generation_NNN.csv``.

Rules (from the handoff):
  * score models on net-of-fee simulated PnL, not raw accuracy;
  * keep diverse top models, drop near-duplicate predictions (corr > threshold);
  * never touch FINAL TEST;
  * start small, scale winners later.

Usage::

    python src/evolution_gpu.py --population 12 --generations 3 --epochs 2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    CONFIDENCE_GATE,
    FEE_RATE,
    NUM_CLASSES,
    RANDOM_STATE,
    RESULTS_DIR,
    SLIPPAGE,
    TARGET_HORIZON,
)
from src.model import ModelConfig, MultiTimeframeTransformer, multihead_loss  # noqa: E402
from src.train import (  # noqa: E402
    RETURN_COLS,
    SequenceDataset,
    build_training_data,
    get_device,
    run_inference,
)
from src.backtest import simulate  # noqa: E402

SEARCH_SPACE = {
    "d_model": [128, 256, 512],
    "layers": [2, 4, 6],
    "heads": [2, 4, 8],
    "dropout": [0.1, 0.15, 0.2, 0.3],
    "lr": [1e-4, 5e-5, 1e-5],
    "weight_decay": [0.0, 0.01, 0.02],
    "seq_len": [15, 30, 60],
    "optimizer": ["adam", "adamw"],
    "return_weight": [0.2, 0.5, 1.0],
    "vol_weight": [0.1, 0.3, 0.5],
    "max_weight": [0.1, 0.3, 0.5],
}

SCORE_WEIGHTS = {"acc": 0.2, "f1": 0.15, "precision": 0.15,
                 "return": 0.15, "pnl": 0.2, "calibration": 0.15}


def sample_genes(rng: np.random.Generator) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def mutate(parent: dict, rng: np.random.Generator) -> dict:
    child = dict(parent)
    keys = list(SEARCH_SPACE)
    n_perturb = int(rng.integers(1, 4))
    for k in rng.choice(keys, size=n_perturb, replace=False):
        child[k] = rng.choice(SEARCH_SPACE[k])
    return child


def crossover(a: dict, b: dict, rng: np.random.Generator) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in SEARCH_SPACE}


def subset_features(Xs: np.ndarray, feature_cols: list[str], rng: np.random.Generator):
    frac = float(rng.choice([0.5, 0.75, 1.0]))
    if frac >= 1.0:
        indices = np.arange(len(feature_cols))
        return Xs, feature_cols, indices
    k = max(8, int(len(feature_cols) * frac))
    indices = np.sort(rng.choice(len(feature_cols), size=k, replace=False))
    return Xs[:, indices], [feature_cols[i] for i in indices], indices


def expected_calibration_error(probs: np.ndarray, preds: np.ndarray,
                               ytrue: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    correct = (preds == ytrue).astype(float)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_score(metrics: dict, sim: dict, ece: float) -> float:
    ret = 1.0 / (1.0 + metrics["return_mae"])
    pnl = (math.tanh(sim["total_return"]) + 1.0) / 2.0
    calib = 1.0 - ece
    w = SCORE_WEIGHTS
    return (w["acc"] * metrics["accuracy"] + w["f1"] * metrics["f1"]
            + w["precision"] * metrics["precision"] + w["return"] * ret
            + w["pnl"] * pnl + w["calibration"] * calib)


def _pred_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def train_individual(
    genes: dict,
    model_id: int,
    Xs: np.ndarray,
    targets: dict,
    times: np.ndarray,
    feature_cols: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE + model_id)
    Xsub, cols, feature_indices = subset_features(Xs, feature_cols, rng)
    cfg = ModelConfig(
        n_features=len(cols),
        d_model=genes["d_model"],
        layers=genes["layers"],
        heads=genes["heads"],
        dropout=genes["dropout"],
        seq_len=genes["seq_len"],
        n_returns=len(RETURN_COLS),
        n_classes=NUM_CLASSES,
        return_weight=genes["return_weight"],
        vol_weight=genes["vol_weight"],
        max_weight=genes["max_weight"],
    )

    model = MultiTimeframeTransformer(cfg).to(device)
    if genes["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=genes["lr"],
                               weight_decay=genes["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=genes["lr"],
                                weight_decay=genes["weight_decay"])

    train_ds = SequenceDataset(Xsub, targets, cfg.context_len, train_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=(device.type == "cuda"))

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        for x, tgt in train_loader:
            x = x.to(device)
            tgt = {k: v.to(device) for k, v in tgt.items()}
            opt.zero_grad()
            loss, _ = multihead_loss(model(x), tgt, cfg)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach())

    metrics = run_inference(model, cfg, Xsub, targets, val_idx, device, batch_size)
    ret5 = targets["returns"][val_idx, 3]
    df_val = pd.DataFrame({"time": times[val_idx], "future_ret_5": ret5})
    conf = metrics["probs"].max(axis=1)
    sim = simulate(df_val, metrics["preds"], conf, CONFIDENCE_GATE)
    ece = expected_calibration_error(metrics["probs"], metrics["preds"], metrics["ytrue"])
    score = compute_score(metrics, sim, ece)

    return {
        "model_id": model_id,
        "genes": genes,
        "n_features": len(cols),
        "feature_indices": feature_indices,
        "cfg": cfg,
        "model": model,
        "metrics": metrics,
        "sim": sim,
        "ece": ece,
        "score": score,
        "final_loss": final_loss,
        "preds": metrics["preds"],
        "diversity_flag": False,
        "val_b_acc": None,
    }


def diverse_topk(scored: list[dict], topk: int, threshold: float) -> list[dict]:
    scored = sorted(scored, key=lambda x: -x["score"])
    survivors: list[dict] = []
    for item in scored:
        if any(_pred_corr(item["preds"], s["preds"]) > threshold for s in survivors):
            item["diversity_flag"] = True
            continue
        survivors.append(item)
        if len(survivors) >= topk:
            break
    return survivors


def next_population(survivors: list[dict], population: int, rng: np.random.Generator) -> list[dict]:
    children: list[dict] = []
    for i in range(population):
        parent = survivors[i % len(survivors)]["genes"]
        if rng.random() < 0.5 and len(survivors) > 1:
            other = survivors[int(rng.integers(len(survivors)))]["genes"]
            children.append(crossover(parent, other, rng))
        else:
            children.append(mutate(parent, rng))
    return children


def _log_row(gen: int, item: dict) -> dict:
    m = item["metrics"]
    return {
        "generation": gen,
        "model_id": item["model_id"],
        "score": item["score"],
        "val_loss": item["final_loss"],
        "acc": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "mae": m["return_mae"],
        "simulated_return": item["sim"]["total_return"],
        "profit_factor": item["sim"]["profit_factor"],
        "drawdown": item["sim"]["max_drawdown"],
        "ece": item["ece"],
        "n_features": item["n_features"],
        "val_b_acc": item["val_b_acc"],
        "diversity_flag": int(item["diversity_flag"]),
        **item["genes"],
    }


def run(population: int = 12, generations: int = 3, topk: int = 4,
        epochs: int = 2, batch_size: int = 64, max_train: int = 15000,
        max_val: int = 4000, diversity_threshold: float = 0.9,
        check_val_b: bool = True) -> None:
    device = get_device()
    print(f"device = {device}")
    X, targets, feature_cols, time_series = build_training_data()
    times = time_series.to_numpy()
    n = len(X)

    # Required 1m context: the 15m branch (32 points x 15m) dominates.
    from src.model import BRANCH_TIMEFRAMES  # noqa: E402
    max_seq = max(SEARCH_SPACE["seq_len"])
    branch_len = {"1m": max_seq, "5m": 36, "15m": 32}
    context = max((nbr - 1) * BRANCH_TIMEFRAMES[name] + 1
                  for name, nbr in branch_len.items())
    first = context - 1

    from src.params import TRAIN_RATIO, VAL_RATIO, VAL_B_RATIO  # noqa: E402
    train_end = int(n * TRAIN_RATIO)
    val_a_end = train_end + int(n * VAL_RATIO)
    val_b_end = val_a_end + int(n * VAL_B_RATIO)

    train_idx = np.arange(first, train_end)
    val_idx = np.arange(train_end, val_a_end)
    val_b_idx = np.arange(val_a_end, val_b_end)
    if len(train_idx) > max_train:
        train_idx = np.linspace(first, train_end - 1, max_train).astype(int)
    if len(val_idx) > max_val:
        val_idx = np.linspace(train_end, val_a_end - 1, max_val).astype(int)

    from sklearn.preprocessing import StandardScaler  # local import
    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X).astype(np.float32)

    rng = np.random.default_rng(RANDOM_STATE)
    population_genes = [sample_genes(rng) for _ in range(population)]
    model_id = 0

    for gen in range(generations):
        print(f"\n=== generation {gen} ===")
        scored: list[dict] = []
        for genes in population_genes:
            item = train_individual(genes, model_id, Xs, targets, times, feature_cols,
                                    train_idx, val_idx, device, epochs, batch_size)
            model_id += 1
            scored.append(item)
            print(f"  id={item['model_id']:3d} score={item['score']:.4f} "
                  f"acc={item['metrics']['accuracy']:.4f} pnl={item['sim']['total_return']:+.4f} "
                  f"params={sum(p.numel() for p in item['model'].parameters()):,}")

        survivors = diverse_topk(scored, topk, diversity_threshold)
        if check_val_b and len(val_b_idx):
            for s in survivors:
                Xsub = Xs[:, s["feature_indices"]]
                m = run_inference(s["model"], s["cfg"], Xsub, targets, val_b_idx,
                                  device, batch_size)
                s["val_b_acc"] = float(m["accuracy"])

        pd.DataFrame([_log_row(gen, it) for it in scored]).to_csv(
            RESULTS_DIR / f"generation_{gen:03d}.csv", index=False)

        if gen == generations - 1:
            break
        population_genes = next_population(survivors, population, rng)

    best = max(scored, key=lambda x: x["score"])
    print(f"\nbest: id={best['model_id']} score={best['score']:.4f} genes={best['genes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU evolution engine (6F)")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=15000)
    parser.add_argument("--max-val", type=int, default=4000)
    parser.add_argument("--diversity-threshold", type=float, default=0.9)
    parser.add_argument("--no-val-b", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run(args.population, args.generations, args.topk, args.epochs, args.batch_size,
        args.max_train, args.max_val, args.diversity_threshold,
        check_val_b=not args.no_val_b)


if __name__ == "__main__":
    main()

