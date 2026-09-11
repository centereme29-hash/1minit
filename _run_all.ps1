$ErrorActionPreference = 'Continue'
$ROOT = 'c:\Users\user\Desktop\1minit'
$log = "$ROOT\_run_all.log"
$marker = "$ROOT\_DONE.marker"
Remove-Item $marker -ErrorAction SilentlyContinue
Set-Content -Path $log -Value "START $([DateTime]::UtcNow)"

function Step($msg) { Add-Content -Path $log -Value "`n========== $msg ==========" }

Step 'Download BTCUSDT 730d'
python "$ROOT\src\download_bybit.py" --symbols BTCUSDT --days 730 *>> $log

Step 'Download XAUTUSDT 730d'
python "$ROOT\src\download_bybit.py" --symbols XAUTUSDT --days 730 *>> $log

Step 'Pipeline clean -> backtest'
python "$ROOT\src\pipeline.py" *>> $log

Step 'Train transformer'
python "$ROOT\src\train.py" --epochs 3 *>> $log

Add-Content -Path $log -Value "FINISHED $([DateTime]::UtcNow)"
Set-Content -Path $marker -Value "done $([DateTime]::UtcNow)"

