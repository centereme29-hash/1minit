$p = 'c:\Users\user\Desktop\1minit\src\run_live.py'
Remove-Item 'c:\Users\user\Desktop\1minit\logs\live_heartbeat.csv' -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath 'python' -ArgumentList '-u', $p -RedirectStandardOutput 'c:\Users\user\Desktop\1minit\logs\live_supervisor.log' -RedirectStandardError 'c:\Users\user\Desktop\1minit\logs\live_supervisor.err' -WindowStyle Hidden -PassThru
Write-Output "PID=$($proc.Pid)"