@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -Filter 'Name=\"python.exe\"' | Select-Object ProcessId,CommandLine | Format-List" > "c:\Users\user\Desktop\1minit\_ps_out.txt" 2>&1
echo RC=%ERRORLEVEL%
