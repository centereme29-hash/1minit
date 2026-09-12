$p = 'c:\Users\user\Desktop\1minit\src\full_system.py'
Remove-Item 'c:\Users\user\Desktop\1minit\logs\full_system.log' -ErrorAction SilentlyContinue
Remove-Item 'c:\Users\user\Desktop\1minit\logs\full_system.err' -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath 'python' -ArgumentList '-u', $p -RedirectStandardOutput 'c:\Users\user\Desktop\1minit\logs\full_system.log' -RedirectStandardError 'c:\Users\user\Desktop\1minit\logs\full_system.err' -WindowStyle Hidden -PassThru
Write-Output "PID=$($proc.Pid)"