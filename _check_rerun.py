from pathlib import Path
import datetime
import subprocess

ROOT = Path(r'c:\Users\user\Desktop\1minit')
log = ROOT / '_rerun.txt'
err = ROOT / '_rerun_err.txt'
proc_txt = ROOT / '_rerun_processes.txt'
out = ROOT / '_rerun_tail.txt'

lines = []
lines.append('TIME ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
lines.append('LOG bytes=' + str(log.stat().st_size if log.exists() else -1))
lines.append('ERR bytes=' + str(err.stat().st_size if err.exists() else -1))
if log.exists():
    txt = log.read_text(encoding='utf-8', errors='ignore').splitlines(True)
    lines.append(''.join(txt[-160:]))
lines.append('\n--- STDERR ---\n')
if err.exists():
    lines.append(err.read_text(encoding='utf-8', errors='ignore'))

lines.append('\n--- PROCESSES ---\n')
try:
    p = subprocess.run([
        'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
        r'Get-CimInstance Win32_Process -Filter "Name=''python.exe''" | Select-Object ProcessId,CommandLine | Format-List'
    ], capture_output=True, text=True, timeout=60)
    lines.append(p.stdout)
    lines.append('RC=' + str(p.returncode))
except Exception as e:
    lines.append('ERR procs: ' + repr(e))

out.write_text('\n'.join(lines), encoding='utf-8')
print('ok', out.stat().st_size)
