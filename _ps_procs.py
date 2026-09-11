from pathlib import Path
import subprocess

ROOT = Path(r'c:\Users\user\Desktop\1minit')
out = ROOT / '_ps_out.txt'

try:
    p = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
         r'Get-CimInstance Win32_Process -Filter "Name=''python.exe''" | '
         r'Select-Object ProcessId,CommandLine | Format-List'],
        capture_output=True, text=True, timeout=120,
    )
    txt = []
    txt.append('RC=' + str(p.returncode))
    txt.append('STDOUT len=' + str(len(p.stdout)))
    txt.append(p.stdout)
    txt.append('STDERR len=' + str(len(p.stderr)))
    txt.append(p.stderr)
    out.write_text('\n'.join(txt), encoding='utf-8')
    print('ok', out.stat().st_size, 'RC', p.returncode)
except Exception as e:
    out.write_text('ERR ' + repr(e), encoding='utf-8')
    print('err', repr(e))
