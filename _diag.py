import sys
import subprocess
import datetime
from pathlib import Path

root = Path(r"c:\Users\user\Desktop\1minit")
diag = root / "_diag.txt"
diag.write_text("TIME " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

proc = subprocess.run(
    [sys.executable, "-c", "\n".join([
        "import sys",
        "sys.path.insert(0, r'c:\\Users\\user\\Desktop\\1minit')",
        "from src.train import build_training_data",
        "X, tgt, fc, t = build_training_data()",
        "print('OK features=', len(fc), 'rows=', X.shape[0])",
    ])],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="ignore",
    timeout=180,
)

diag.write_text(
    diag.read_text(encoding="utf-8", errors="ignore")
    + "\n=== RETURN CODE ===\n"
    + str(proc.returncode)
    + "\n=== STDOUT ===\n"
    + proc.stdout
    + "\n=== STDERR ===\n"
    + proc.stderr
    + "\n",
    encoding="utf-8",
)
print("diag written", diag.stat().st_size)
