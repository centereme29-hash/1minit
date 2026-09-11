import subprocess
from pathlib import Path

root = Path(r"c:\Users\user\Desktop\1minit")
out = root / "_procs_wmi.txt"
proc = subprocess.run(
    ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="ignore",
)
out.write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
print("wrote", out.stat().st_size, "chars; rc=", proc.returncode)
