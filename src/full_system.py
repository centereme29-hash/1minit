"""Step 7 - FULL SYSTEM run (Stage A): regenerate the whole stack at full power.

Runs every modelling stage in dependency order with artifact-verification gates:

  data? -> clean -> synchronize -> features -> targets -> dataset
        -> baselines (2-class XGB/LGB/MLP) -> backtest
        -> transformer (GPU) -> evolution (GPU) -> walk-forward
        -> event_detector -> event_m23 -> event_calibrate -> event_viz

If a gate is not satisfied the run STOPS before cascading downstream.  Every
stage is a standalone script, so a failed stage can be re-run on its own.

Usage:
    python src/full_system.py                     # everything (download skipped if raw exists)
    python src/full_system.py --with-download     # also re-fetch historical candles
    python src/full_system.py --skip-evolution    # skip the GPU evolutionary search
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

RAW_OK = (ROOT / "data" / "raw" / "DOGEUSDT_1m_raw.csv").exists()


def run(script: str, *args, tag: str) -> None:
    cmd = [sys.executable, str(SRC / script), *args]
    print("\n===== [%s] $ %s =====" % (tag, " ".join(cmd)), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def mark(name: str, path: str) -> None:
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        raise SystemExit("GATE FAILED: %s -> %s missing/empty" % (name, p))
    print("GATE OK: %s -> %s (%s bytes)" % (name, p, p.stat().st_size), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full system run (Stage A)")
    ap.add_argument("--with-download", action="store_true")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--skip-evolution", action="store_true")
    args = ap.parse_args()

    if args.with_download:
        run("download_bybit.py", "--days", str(args.days), tag="download")
    elif not RAW_OK:
        print("WARN: raw candles missing - use --with-download", flush=True)

    for script, tag in (("clean_data.py", "clean"),
                        ("synchronize.py", "synchronize"),
                        ("features.py", "features"),
                        ("targets.py", "targets"),
                        ("dataset.py", "dataset")):
        run(script, tag=tag)
    mark("synchronized wide", "data/synchronized/all_1m.csv")
    mark("features matrix", "data/features/features.csv")
    mark("targets", "data/features/targets.csv")
    mark("chronological splits", "data/features/train.csv")

    run("baselines.py", tag="baselines")
    run("backtest.py", tag="backtest")
    mark("xgboost baseline", "models/xgboost.joblib")
    mark("baseline csv", "results/baselines.csv")
    mark("backtest csv", "results/backtest.csv")

    run("train.py", "--epochs", str(args.epochs), "--batch-size", "96",
        "--max-train-windows", "20000", tag="transformer")
    mark("transformer", "models/transformer.pt")
    mark("transformer metrics", "results/transformer_metrics.json")

    if not args.skip_evolution:
        run("evolution_gpu.py", "--population", "12", "--generations", "3",
            "--epochs", "1", tag="evolution_gpu")
    run("walk_forward.py", "--model", "lightgbm", "--n-windows", "8",
        "--subsample", "50000", tag="walk_forward")
    mark("walk-forward csv", "results/walk_forward.csv")

    run("event_detector.py", tag="event_detector")
    mark("event lgb model", "models/event_lgb.joblib")
    run("event_m23.py", tag="event_m23")
    mark("event metrics", "results/event_m23_summary.json")
    run("event_calibrate.py", tag="event_calibrate")   # must run AFTER event_detector
    mark("calibration summary", "results/event_retrain_summary.json")
    run("event_viz.py", tag="event_viz")
    mark("dataset view html", "data/dataset_view.html")

    print("\nSTAGE A COMPLETE.", flush=True)


if __name__ == "__main__":
    main()