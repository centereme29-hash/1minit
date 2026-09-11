import time
time.sleep(5)
from pathlib import Path
import datetime

root = Path(r"c:\Users\user\Desktop\1minit")
log = root / "_train_stdout.txt"
err = root / "_train_stderr.txt"
m = root / "models" / "transformer.pt"
snap = root / "_train_snapshot.txt"

out = open(snap, "w", encoding="utf-8")
out.write("TIME " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
out.write("stdout_bytes=" + str(log.stat().st_size if log.exists() else -1) + "\n")
out.write("stderr_bytes=" + str(err.stat().st_size if err.exists() else -1) + "\n")
out.write("transformer_exists=" + str(m.exists()) + "\n")
out.write("transformer_size=" + str(m.stat().st_size if m.exists() else -1) + "\n")
out.write("\n=== LAST 120 STDOUT ===\n")
out.write("".join(log.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-120:]) if log.exists() else "no log")
out.write("\n=== LAST 120 STDERR ===\n")
out.write("".join(err.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-120:]) if err.exists() else "no err")
out.close()
print("done", snap.stat().st_size)
