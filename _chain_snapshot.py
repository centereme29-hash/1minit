from pathlib import Path
import datetime, subprocess

root = Path(r'c:\Users\user\Desktop\1minit')
out = root / '_chain_snapshot.txt'

t0 = datetime.datetime.now()
with out.open('w', encoding='utf-8') as f:
    f.write('TIME ' + t0.strftime('%Y-%m-%d %H:%M:%S') + '\n\n')

    tp = root / 'models' / 'transformer.pt'
    f.write('transformer.pt exists=' + str(tp.exists()) + '\n')
    if tp.exists():
        f.write('transformer.pt bytes=' + str(tp.stat().st_size) + '\n')
    tn = root / 'models' / 'transformer_norms.json'
    f.write('transformer_norms.json exists=' + str(tn.exists()) + '\n')
    f.write('\n')

    # Step 1: baselines (fast)
    logpath = root / '_baselines.log'
    if logpath.exists():
        logpath.unlink()
    cmd = ['python', str(root / 'src' / 'baselines.py')]
    f.write('=== BASELINES ===\n')
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(root))
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        rc = -999
        p = None
    f.write('RC=' + str(rc) + '\n')
    if p and p.stdout:
        f.write('STDOUT:\n' + p.stdout[-3000:] + '\n')
    if p and p.stderr:
        f.write('STDERR:\n' + p.stderr[-3000:] + '\n')
    f.write('log bytes=' + str(logpath.stat().st_size if logpath.exists() else -1) + '\n\n')

print('step1 done', out.stat().st_size)
