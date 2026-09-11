$ErrorActionPreference = 'Continue'
$ROOT = 'c:\Users\user\Desktop\1minit'
$log = "$ROOT\_train_trans.log"
$marker = "$ROOT\_TRAIN_DONE.marker"
Remove-Item $marker -ErrorAction SilentlyContinue
Set-Content -Path $log -Value "TRAIN START $([DateTime]::UtcNow)"
python "$ROOT\src\train.py" --epochs 3 *>> $log
Add-Content -Path $log -Value "TRAIN FINISHED $([DateTime]::UtcNow)"
Set-Content -Path $marker -Value "done $([DateTime]::UtcNow)"
