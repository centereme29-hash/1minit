from pathlib import Path
import datetime, time

ROOT = Path(r'c:\Users\user\Desktop\1minit')
log = ROOT / '_rerun.txt'
err = ROOT / '_rerun_err.txt'
out = ROOT / '_rerun_tail.txt'

for _ in range(12):
    time.sleep(15)
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
    out.write_text('\n'.join(lines), encoding='utf-8')
    if log.stat().st_size > 200:
        break

print('done', out.stat().st_size)
