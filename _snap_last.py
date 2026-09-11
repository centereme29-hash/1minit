from pathlib import Path
import datetime, subprocess

root = Path(r'c:\Users\user\Desktop\1minit')
log = root / '_rerun.txt'
err = root / '_rerun_err.txt'
out = root / '_rerun_tail.txt'
proc = root / '_rerun_procs.txt'

with proc.open('w', encoding='utf-8') as f:
    f.write('TIME ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    p = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
                       capture_output=True, text=True, timeout=60)
    f.write(p.stdout)
    f.write('RC=' + str(p.returncode) + '\n')

out.write_text('SNAP ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n', encoding='utf-8')
if log.exists():
    lines = log.read_text(encoding='utf-8', errors='ignore').splitlines(True)
    with out.open('a', encoding='utf-8') as f:
        f.write('LOG lines=' + str(len(lines)) + '\n')
        f.write(''.join(lines[-200:]))
else:
    with out.open('a', encoding='utf-8') as f:
        f.write('LOG missing\n')

if err.exists():
    el = err.read_text(encoding='utf-8', errors='ignore')
    with out.open('a', encoding='utf-8') as f:
        f.write('ERR bytes=' + str(len(el)) + '\n')
        f.write(el)

with out.open('a', encoding='utf-8') as f:
    f.write('\n--- DONE ---\n')
print('snap ok', out.stat().st_size, proc.stat().st_size)
