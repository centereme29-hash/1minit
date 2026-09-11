"""End-to-end pipeline orchestrator.

Runs every stage in order (skipping the long historical download unless
``--with-download`` is passed).  Each stage is a standalone script, so a stage
can also be re-run on its own.

    python src/pipeline.py               # clean → sync → features → ... → backtest
    python src/pipeline.py --with-download --days 730
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

STAGES = [
    ("clean", SRC / "clean_data.py"),
    ("synchronize", SRC / "synchronize.py"),
    ("features", SRC / "features.py"),
    ("targets", SRC / "targets.py"),
    ("dataset", SRC / "dataset.py"),
    ("baselines", SRC / "baselines.py"),
    ("backtest", SRC / "backtest.py"),
]


def run_script(script: Path, *extra) -> None:
    cmd = [sys.executable, str(script), *extra]
    print("\n" + "=" * 72 + f"\n$ {' '.join(cmd)}\n" + "=" * 72, flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-download", action="store_true")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--symbols", nargs="+", default=["DOGEUSDT", "BTCUSDT", "XAUTUSDT"])
    args = parser.parse_args()

    if args.with_download:
        run_script(SRC / "download_bybit.py", "--days", str(args.days),
                   "--symbols", *args.symbols)

    for name, script in STAGES:
        run_script(script)

    print("\nPIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
