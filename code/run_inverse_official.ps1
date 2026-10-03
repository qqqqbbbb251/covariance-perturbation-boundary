# run_inverse_official.ps1
$ErrorActionPreference = "Continue"
$py   = "python"
$code = "$PSScriptRoot"
$log  = Join-Path $code "inverse_official.log"
Set-Location $code
"=== inverse_official start $(Get-Date) ===" | Out-File -Encoding utf8 $log
& $py "$code\inverse_official.py" *>> $log
"=== inverse_official done $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
