from __future__ import annotations
import subprocess, sys, datetime
from pathlib import Path

ROOT = Path(r"c:\Users\user\Desktop\1minit")
PYTHON = r"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"
SNAP = ROOT / "_train_chain_snapshot.txt"
SNAP.write_text("CHAIN START " + datetime.datetime.now().isoformat() + "\n")

def run(label, script, extras=(), logname="_chain.log"):
    SNAP.write_text(SNAP.read_text() + f"\n===== {label} ({datetime.datetime.now().isoformat()}) =====\n")
    logp = ROOT / logname
    cmd = [PYTHON, str(ROOT / "src" / script), *extras]
    with logp.open("wb") as logf:
        p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    rc = p.returncode
    SNAP.write_text(SNAP.read_text() + f"{label} RC={rc}\n")
    SNAP.write_text(SNAP.read_text() + f"{label} log_bytes={logp.stat().st_size if logp.exists() else 0}\n")
    if logp.exists():
        data = logp.read_bytes()
        SNAP.write_text(SNAP.read_text() + data.decode("utf-8", "replace")[-4000:] + "\n")
    else:
        SNAP.write_text(SNAP.read_text() + "(no log)\n")
    return rc

# 1. Train transformer (3 epochs)
rc1 = run("train_transformer", "train.py", ("--epochs", "3"), "_train.log")
SNAP.write_text(SNAP.read_text() + f"\n=== train RC={rc1} ===\n")

# 2. Ensemble backtest (uses baselines + transformer if present)
rc2 = run("backtest", "backtest.py", (), "_backtest.log")
SNAP.write_text(SNAP.read_text() + f"\n=== backtest RC={rc2} ===\n")

# 3. Transformer evaluate on held-out test
rc3 = run("transformer_eval", "train.py", ("--eval",), "_eval.log")
SNAP.write_text(SNAP.read_text() + f"\n=== eval RC={rc3} ===\n")

SNAP.write_text(SNAP.read_text() + "\nCHAIN DONE " + datetime.datetime.now().isoformat() + "\n")
print("DONE snapshot_bytes=" + str(SNAP.stat().st_size))
