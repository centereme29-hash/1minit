from pathlib import Path
import subprocess

ROOT = Path(r'c:\Users\user\Desktop\1minit')
out = ROOT / '_cmdline.txt'

script = r"""
$x = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId,CommandLine
foreach ($o in $x) {
    $o.ProcessId
    $o.CommandLine
    '---'
}
"""
try:
    p = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script],
        capture_output=True, text=True, timeout=120,
    )
    lines = []
    lines.append('RC=' + str(p.returncode))
    lines.append('out_len=' + str(len(p.stdout)))
    lines.append(p.stdout)
    lines.append('err_len=' + str(len(p.stderr)))
    if p.stderr.strip():
        lines.append('STDERR:')
        lines.append(p.stderr)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('ok', out.stat().st_size)
except Exception as e:
    out.write_text('ERR ' + repr(e), encoding='utf-8')
    print('err', repr(e))
