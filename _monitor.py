import os, time, glob, pandas as pd

ROOT = r"c:\Users\user\Desktop\1minit"
log = os.path.join(ROOT, "_run_all.log")
marker = os.path.join(ROOT, "_DONE.marker")
out = os.path.join(ROOT, "_status.txt")

with open(out, "w", encoding="utf-8") as f:
    if os.path.exists(marker):
        f.write("STATUS: DONE\n")
    else:
        f.write("STATUS: RUNNING\n")
    # tail the log with robust encoding detection
    try:
        raw = open(log, "rb").read()
        for enc in ("utf-16", "utf-16-le", "utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        lines = [l.rstrip("\r\n") for l in text.splitlines() if l.strip()]
        f.write("--- log tail ---\n")
        for l in lines[-12:]:
            f.write(l + "\n")
    except Exception as e:
        f.write(f"log read error: {e}\n")
    # raw csv row counts
    f.write("--- raw csv sizes ---\n")
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*", "*_1m.csv"))):
        try:
            sz = os.path.getsize(p)
            n = sum(1 for _ in open(p)) - 1
            f.write(f"{os.path.basename(p)}: {n} rows, {sz} bytes\n")
        except Exception as e:
            f.write(f"{p}: err {e}\n")
