from pathlib import Path
import subprocess

ROOT = Path(r'c:\Users\user\Desktop\1minit')
out = ROOT / '_tlist.txt'

try:
    p = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
         r'tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH'],
        capture_output=True, text=True, timeout=60,
    )
    lines = ['RC=' + str(p.returncode), 'out_len=' + str(len(p.stdout)), p.stdout, 'err_len=' + str(len(p.stderr))]
    if p.stderr.strip():
        lines.append('STDERR:')
        lines.append(p.stderr)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('ok', out.stat().st_size)
except Exception as e:
    out.write_text('ERR ' + repr(e), encoding='utf-8')
    print('err', repr(e))
