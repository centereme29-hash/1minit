$ErrorActionPreference = 'Continue'
Set-Location 'c:\Users\user\Desktop\1minit'
$env:PYTHONUNBUFFERED = '1'

$log = 'c:\Users\user\Desktop\1minit\_pipeline.log'
'' | Out-File $log -Encoding utf8

function Step($name, $cmd) {
    "`n===== START $name =====" | Out-File $log -Append -Encoding utf8
    Invoke-Expression $cmd *>> $log
    $code = $LASTEXITCODE
    "===== END $name (exit $code) =====" | Out-File $log -Append -Encoding utf8
    if ($code -ne 0) {
        "`n===== FATAL: $name failed (exit $code), stopping =====`n" | Out-File $log -Append -Encoding utf8
        exit $code
    }
}

Step 'features'     'python src/features.py'
Step 'targets'      'python src/targets.py'
Step 'dataset'      'python src/dataset.py'
Step 'baselines'    'python src/baselines.py'
Step 'backtest'     'python src/backtest.py'
Step 'walk_forward' 'python src/walk_forward.py --model lightgbm --n-windows 8'

"`n===== ALL DONE =====" | Out-File $log -Append -Encoding utf8
