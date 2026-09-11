import os, re, glob, datetime

ROOT = r"c:\Users\user\Desktop\1minit"
log = os.path.join(ROOT, "_run_all.log")
marker = os.path.join(ROOT, "_DONE.marker")
out = os.path.join(ROOT, "_status.txt")


def decode_log(raw: bytes) -> str:
    # Log mixes UTF-16 (PowerShell *>> wrapper) and UTF-8 (python child output).
    # Drop UTF-16 null bytes, then decode latin-1 so every byte survives.
    return raw.replace(b"\x00", b"").decode("latin-1", errors="replace")


with open(out, "w", encoding="utf-8") as f:
    f.write(f"checked {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z\n")
    f.write("STATUS: DONE (marker present)\n" if os.path.exists(marker) else "STATUS: RUNNING\n")
    stage = sym = page = None
    saved = []
    errors = []
    try:
        text = decode_log(open(log, "rb").read())
        for l in text.splitlines():
            l = l.strip()
            if not l:
                continue
            m = re.search(r"={5,}\s*(.+?)\s*={5,}", l)
            if m:
                stage = m.group(1)
            m = re.search(r"\$ python .*[\\/]src[\\/](\w+\.py)", l)
            if m:
                stage = "pipeline: " + m.group(1)
            m = re.search(r"(STEP \d+[^\n]*)", l)
            if m:
                stage = m.group(1)
            m = re.search(r"\[([A-Z]+USDT)\]\s+page\s+(\d+):", l)
            if m:
                sym, page = m.group(1), int(m.group(2))
            m = re.search(r"\[([A-Z]+USDT)\]\s+saved\s+(\d+)\s+rows", l)
            if m:
                saved.append((m.group(1), int(m.group(2))))
            if "ERROR" in l:
                errors.append(l)
            if "PIPELINE COMPLETE" in l or "FINISHED" in l:
                stage = l
        f.write(f"stage: {stage}\n")
        f.write(f"current symbol: {sym}  page: {page}\n")
        if saved:
            f.write("saved: " + ", ".join(f"{s}={n}" for s, n in saved) + "\n")
        if errors:
            f.write("errors (last 5):\n")
            for e in errors[-5:]:
                f.write("  " + e + "\n")
    except Exception as e:
        f.write(f"log read error: {e}\n")
    f.write("--- raw csv rows ---\n")
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*", "*_1m.csv"))):
        try:
            n = sum(1 for _ in open(p)) - 1
            f.write(f"{os.path.basename(p)}: {n} rows\n")
        except Exception as e:
            f.write(f"{p}: err {e}\n")
    f.write("--- models ---\n")
    md = os.path.join(ROOT, "models")
    if os.path.isdir(md):
        for name in sorted(os.listdir(md)):
            f.write(f"{name}: {os.path.getsize(os.path.join(md, name))} bytes\n")
    else:
        f.write("(no models dir)\n")
