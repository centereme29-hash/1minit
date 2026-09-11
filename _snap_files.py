from pathlib import Path
import datetime

root = Path(r"c:\Users\user\Desktop\1minit")
files = [
    "_bl_check.txt",
    "_bl_direct.log",
    "_chain_snapshot.txt",
    "_train.log",
    "_backtest.log",
    "_eval.log",
    "models/transformer.pt",
    "results/backtest.csv",
    "results/evaluate.json",
]
bits = []
bits.append("time=" + datetime.datetime.now().strftime("%H:%M:%S"))
for fn in files:
    p = root / fn
    if p.exists():
        bits.append(fn + " exists=True bytes=" + str(p.stat().st_size))
    else:
        bits.append(fn + " exists=False")
out = root / "_snap_files.txt"
out.write_text("\n".join(bits))
print("snapshot bytes=" + str(out.stat().st_size))
