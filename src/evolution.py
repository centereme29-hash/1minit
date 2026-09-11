"""Steps 13-18 — CPU-feasible evolutionary search over baseline hyperparameters.

A full 100-model transformer population is not practical on CPU, so this module
evolves **LightGBM** hyperparameters instead: sample a population (generation 0),
keep the top-K by validation F1, then produce the next generation via mutation +
crossover.  It demonstrates the evolution loop end-to-end and is fully runnable.

    python src/evolution.py --population 12 --generations 3 --models lightgbm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import RESULTS_DIR, RANDOM_STATE  # noqa: E402
from src.baselines import evaluate, load_split  # noqa: E402

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [3, 5, 7, 10, -1],
    "min_child_samples": [5, 10, 20, 50],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}


def sample_config(rng: np.random.Generator) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def mutate(parent: dict, rng: np.random.Generator) -> dict:
    child = dict(parent)
    for k, v in SEARCH_SPACE.items():
        if rng.random() < 0.4:  # 40% chance to mutate each gene
            child[k] = rng.choice(v)
    return child


def crossover(a: dict, b: dict, rng: np.random.Generator) -> dict:
    child = {}
    for k in SEARCH_SPACE:
        child[k] = a[k] if rng.random() < 0.5 else b[k]
    return child


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=40000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    _, Xtr, ytr, _ = load_split("train")
    _, Xval, yval, _ = load_split("val")

    # Subsample for speed (still chronological order is preserved by index).
    idx = np.linspace(0, len(Xtr) - 1, min(args.subsample, len(Xtr))).astype(int)
    Xtr, ytr = Xtr[idx], ytr[idx]

    population = [sample_config(rng) for _ in range(args.population)]
    log: list[dict] = []

    for gen in range(args.generations):
        print(f"\n=== Generation {gen} ===")
        scored = []
        for cfg in population:
            model = _fit(cfg, Xtr, ytr, Xval, yval)
            pred = model.predict(Xval)
            rep = evaluate("lgb", yval, pred)
            rep.update(cfg)
            rep["generation"] = gen
            scored.append((rep["f1"], rep, model))
            print(f"  f1={rep['f1']:.4f}  {cfg}")
            log.append(rep)

        scored.sort(key=lambda x: -x[0])
        top = [c for _, _, c in scored[: args.topk]]

        if gen == args.generations - 1:
            best = scored[0][1]
            break
        population = [top[i % len(top)] for i in range(args.population)]
        population = [mutate(m, rng) if rng.random() < 0.7 else crossover(m, top[rng.integers(len(top))], rng)
                      for m in population]

    pd.DataFrame(log).to_csv(RESULTS_DIR / "evolution.csv", index=False)
    print(f"\nbest config: {best}")
    print(f"saved -> {RESULTS_DIR / 'evolution.csv'}")


def _fit(cfg: dict, Xtr, ytr, Xval, yval):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=400, objective="multiclass", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **cfg,
    )
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


if __name__ == "__main__":
    run()
