from pathlib import Path
import datetime

root = Path(r'c:\Users\user\Desktop\1minit')
log = root / '_rerun.txt'
err = root / '_rerun_err.txt'
out = root / '_rerun_tail.txt'

lines = []
lines.append('TIME ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
lines.append('LOG bytes=' + str(log.stat().st_size if log.exists() else -1))
lines.append('ERR bytes=' + str(err.stat().st_size if err.exists() else -1))
if log.exists():
    txt = log.read_text(encoding='utf-8', errors='ignore').splitlines(True)
    lines.append(''.join(txt[-220:]))
lines.append('\n--- STDERR ---\n')
if err.exists():
    lines.append(err.read_text(encoding='utf-8', errors='ignore'))
out.write_text('\n'.join(lines), encoding='utf-8')
print('ok', out.stat().st_size)
