$ErrorActionPreference = 'Stop'
$root = 'c:\Users\user\Desktop\1minit'
$log = "$root\_rerun_faillog.txt"
Remove-Item $log -ErrorAction SilentlyContinue
"`nRUN BEGIN $(Get-Date -UtcNow)`n" | Out-File $log -Encoding utf8
python "$root\src\train.py" --epochs 1 *>> $log
$rc = $LASTEXITCODE
"`nEXIT CODE $rc`nLOG BYTES: $($log | Get-Item).Length" | Out-File $log -Encoding utf8 -Append
