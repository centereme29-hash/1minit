from __future__ import annotations
import datetime
import subprocess
import sys
import time
from pathlib import Path

root = Path(r"c:\Users\user\Desktop\1minit")
snap = root / "_chain_snapshot.txt"
snap.write_text(f"CHAIN START {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

PYTHON = sys.executable
steps = [
    ("baselines", [str(root / "src" / "baselines.py")]),
    ("train_3epochs", [str(root / "src" / "train.py"), "--epochs", "3"]),
    ("backtest", [str(root / "src" / "backtest.py")]),
    ("train_eval", [str(root / "src" / "train.py"), "--eval"]),
]

log_map = {
    "baselines": root / "_baselines.log",
    "train_3epochs": root / "_train_chain.log",
    "backtest": root / "_backtest.log",
    "train_eval": root / "_train_eval.log",
}

def snap_append(text: str) -> None:
    with snap.open("a", encoding="utf-8") as f:
        f.write(text)

for name, args in steps:
    log = log_map[name]
    snap_append(f"===== {name} ({' '.join(args)}) =====\n")
    snap_append(f"started_at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    with log.open("wb") as lf:
        proc = subprocess.run(
            [PYTHON, *args],
            cwd=str(root),
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=7200,
        )
    rc = int(proc.returncode)
    snap_append(f"ended_at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    snap_append(f"RC = {rc}\n")
    snap_append(f"log bytes = {log.stat().st_size}\n")
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = "".join(lines[-120:]) or "(empty)"
    snap_append(f"--- tail ---\n{tail}\n\n")
    print(name, "RC", rc, "bytes", log.stat().st_size, flush=True)

snap_append(f"CHAIN END {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
print("done", flush=True)
