"""Verify the live feature-strip fix is in place and working."""
import sys
from pathlib import Path

ROOT = Path(r"c:\Users\user\Desktop\1minit")
sys.path.insert(0, str(ROOT))

import numpy as np

# 1. Check live.py:predict_baseline strips micro columns
from src.live import predict_baseline
import inspect
src = inspect.getsource(predict_baseline)
has_strip = "_micro_" in src
print(f"[live.py:predict_baseline] strips _micro_: {has_strip}")

# 2. Check paper.py:latest_features strips micro columns
from src.paper import latest_features
src2 = inspect.getsource(latest_features)
has_strip2 = "_micro_" in src2
print(f"[paper.py:latest_features] strips _micro_: {has_strip2}")

# 3. Check live_dashboard.py run_once strips micro columns
import live_dashboard
src3 = inspect.getsource(live_dashboard.run_once)
has_strip3 = "_micro_" in src3
print(f"[live_dashboard.py:run_once] strips _micro_: {has_strip3}")

# 4. Check what build_features currently produces (with available micro data)
from src.features import build_features
from src.live import fetch_wide
try:
    wide = fetch_wide(minutes=200)
    feats = build_features(wide)
    n_cols = len([c for c in feats.columns if c != "time"])
    n_micro = len([c for c in feats.columns if "_micro_" in c])
    print(f"[build_features] total feature cols (excl time): {n_cols}  micro cols: {n_micro}")
    print(f"[build_features] feature sample: {list(feats.columns)[:5]} ... {list(feats.columns)[-3:]}")
except Exception as e:
    print(f"[build_features] could not fetch/run: {type(e).__name__}: {e}")

# 5. Check ensemble_proba / ensemble_predict signatures
from src.ensemble import ensemble_predict, ensemble_proba
from src.baselines import predict_proba, load_models
print(f"[ensemble] ensemble_predict={ensemble_predict}  ensemble_proba={ensemble_proba}")
print(f"[baselines] predict_proba={predict_proba}  load_models={load_models}")

print("\nDONE")
