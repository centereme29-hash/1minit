"""CSV persistence helpers: append new rows and de-duplicate on a key column.

Keeps files growing (live collection) instead of overwriting, while never
storing the same candle twice.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_csv(path: str | Path, df: pd.DataFrame, key: str) -> Path:
    """Append ``df`` to ``path``, de-duplicating on ``key`` (newest wins).

    Column lists are unioned so a growing schema never drops old columns.
    Callers pass already time-sorted frames; the merge keeps chronological order
    because a fresh fetch overlaps the tail of the existing file and
    ``drop_duplicates(keep="last")`` replaces the old copies in place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    old = pd.DataFrame()
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()

    if old.empty:
        merged = df.copy()
    else:
        cols = list(dict.fromkeys([*old.columns, *df.columns]))
        merged = pd.concat(
            [old.reindex(columns=cols), df.reindex(columns=cols)],
            ignore_index=True,
        )

    if key in merged.columns and not merged.empty:
        merged["_key"] = merged[key].astype(str)
        merged = merged.drop_duplicates(subset="_key", keep="last").drop(columns="_key")

    if key in merged.columns and not merged.empty:
        try:
            merged = merged.sort_values(key)
        except (TypeError, ValueError):
            pass
    merged = merged.reset_index(drop=True)

    merged.to_csv(path, index=False)
    return path
