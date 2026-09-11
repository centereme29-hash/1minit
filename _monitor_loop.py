import os, time, glob, datetime

ROOT = r"c:\Users\user\Desktop\1minit"
log = os.path.join(ROOT, "_run_all.log")
marker = os.path.join(ROOT, "_DONE.marker")
out = os.path.join(ROOT, "_status.txt")

while True:
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"checked {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z\n")
            if os.path.exists(marker):
                f.write("STATUS: DONE\n")
            else:
                f.write("STATUS: RUNNING\n")
            try:
                raw = open(log, "rb").read()
                for enc in ("utf-16", "utf-16-le", "utf-8", "latin-1"):
                    try:
                        text = raw.decode(enc); break
                    except Exception:
                        continue
                lines = [l.rstrip("\r\n") for l in text.splitlines() if l.strip()]
                f.write("--- log tail ---\n")
                for l in lines[-10:]:
                    f.write(l + "\n")
            except Exception as e:
                f.write(f"log read error: {e}\n")
            f.write("--- raw csv sizes ---\n")
            for p in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*", "*_1m.csv"))):
                try:
                    n = sum(1 for _ in open(p)) - 1
                    f.write(f"{os.path.basename(p)}: {n} rows, {os.path.getsize(p)} bytes\n")
                except Exception as e:
                    f.write(f"{p}: err {e}\n")
    except Exception:
        pass
    if os.path.exists(marker):
        break
    time.sleep(60)
