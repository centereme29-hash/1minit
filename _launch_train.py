from pathlib import Path
import subprocess, sys, datetime

ROOT = Path(r'c:\Users\user\Desktop\1minit')
log = ROOT / '_train_stdout.txt'
err = ROOT / '_train_stderr.txt'
start = ROOT / '_train_started.txt'

log.write_text('LAUNCH ' + datetime.datetime.now().isoformat() + '\n', encoding='utf-8')
err.write_text('LAUNCH ' + datetime.datetime.now().isoformat() + '\n', encoding='utf-8')

cmd = [sys.executable, str(ROOT / 'src' / 'train.py'), '--epochs', '3']
with open(log, 'a', encoding='utf-8') as fo, open(err, 'a', encoding='utf-8') as fe:
    p = subprocess.run(cmd, stdout=fo, stderr=fe, cwd=str(ROOT))

start.write_text('RC=' + str(p.returncode) + ' ' + datetime.datetime.now().isoformat() + '\n', encoding='utf-8')
print('done RC', p.returncode)
