from __future__ import annotations
import datetime
import subprocess
import sys
import time
from pathlib import Path

root = Path(r"c:\Users\user\Desktop\1minit")

# 1) kill all python.exe tree
subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/T"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

# 2) snapshot current chain_snapshot
sn = root / "_chain_snapshot.txt"
snap_now = root / "_snap_now.txt"
txt = sn.read_text(encoding="utf-8", errors="ignore") if sn.exists() else "(missing)"
with open(snap_now, "wb") as fh:
    fh.write(f"TIME {datetime.datetime.now():%Y-%m-%d %H:%M:%S} SIZE={sn.stat().st_size if sn.exists() else -1}\n\n".encode())
    fh.write(txt.encode())

# 3) reimplemented rerun: sequential logs, no chain_snapshot
def run_one(name: str, args: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as lf:
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=str(root),
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=7200,
        )
    rc = int(proc.returncode)
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = "".join(lines[-80:]) or "(empty)"
    with open(root / f"_{name}_result.txt", "w", encoding="utf-8") as f:
        f.write(f"RC = {rc}\nlog bytes = {log.stat().st_size}\n--- tail ---\n{tail}\n")
    return rc

rc_b = run_one("baselines", [str(root / "src" / "baselines.py")], root / "_baselines.log")
rc_t = run_one("train", [str(root / "src" / "train.py"), "--epochs", "3"], root / "_train_rerun.log")
rc_bt = run_one("backtest", [str(root / "src" / "backtest.py")], root / "_backtest.log")
rc_e = run_one("eval", [str(root / "src" / "train.py"), "--eval"], root / "_eval.log")

with open(root / "_rerun_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"baselines RC={rc_b}\n")
    f.write(f"train RC={rc_t}\n")
    f.write(f"backtest RC={rc_bt}\n")
    f.write(f"eval RC={rc_e}\n")
    tp = root / "models" / "transformer.pt"
    f.write(f"transformer.pt exists={tp.exists()} size={tp.stat().st_size if tp.exists() else -1}\n")
print("rerun done")
