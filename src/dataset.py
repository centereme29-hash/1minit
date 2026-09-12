"""Step 11 — chronological train / validation / test split.

Builds the final modelling matrix (features + targets, NaN rows dropped) and
splits it strictly by time: 60% train, 20% validation, 20% test.  The test set
is the most recent data and must never be touched during model development.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.params import (  # noqa: E402
    FEATURES_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_B_RATIO,
    VAL_RATIO,
)

TARGET_COLS = [
    "future_ret_1", "future_ret_2", "future_ret_3", "future_ret_5",
    "future_ret_10", "normalized_move", "label",
    "future_max_up", "future_max_down", "future_vol",
]


def build_matrix() -> pd.DataFrame:
    feats = pd.read_csv(FEATURES_DIR / "features.csv")
    targets = pd.read_csv(FEATURES_DIR / "targets.csv")
    feats["time"] = pd.to_datetime(feats["time"], utc=True)
    targets["time"] = pd.to_datetime(targets["time"], utc=True)

    m = feats.merge(targets, on="time", how="inner")
    m = m.sort_values("time").reset_index(drop=True)

    feature_cols = [c for c in feats.columns if c != "time"]
    # Drop warm-up / tail rows with missing features or missing targets.
    m = m.dropna(subset=feature_cols + TARGET_COLS).reset_index(drop=True)
    if len(m) == 0:
        micro_cols = [c for c in feature_cols if "_micro_" in c]
        raise RuntimeError(
            "dataset is empty after NaN-drop. This usually means sparse "
            "microstructure features are being merged but cover only a few minutes, "
            f"so every historical row has NaN micro columns ({len(micro_cols)} micro "
            "columns present). Either collect more micro data (run "
            "`python src/microstructure.py` for several days) or clear "
            "`data/microstructure/` to build the candle-only matrix."
        )
    return m, feature_cols


def chrono_split(m: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.DataFrame]:
    n = len(m)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)          # Validation-A
    val_b_end = val_end + int(n * VAL_B_RATIO)        # Validation-B

    keep = ["time", *feature_cols, *TARGET_COLS]
    splits = {
        "train": m.iloc[:train_end][keep].reset_index(drop=True),
        "val": m.iloc[train_end:val_end][keep].reset_index(drop=True),
    }
    if VAL_B_RATIO > 0 and val_b_end > val_end:
        splits["val_b"] = m.iloc[val_end:val_b_end][keep].reset_index(drop=True)
    splits["test"] = m.iloc[val_b_end:][keep].reset_index(drop=True)
    return splits


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 11 — chronological split (train / val-a / val-b / test)")

    m, feature_cols = build_matrix()
    splits = chrono_split(m, feature_cols)

    for name, df in splits.items():
        path = FEATURES_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name}: {len(df)} rows  "
              f"{df['time'].iloc[0]:%Y-%m-%d %H:%M} → {df['time'].iloc[-1]:%Y-%m-%d %H:%M}")

    print(f"feature columns = {len(feature_cols)}")
    print("label distribution (train):")
    print(splits["train"]["label"].value_counts().sort_index().to_string())
    print("\nSTEP 11 DONE.")


if __name__ == "__main__":
    main()
