# run_diff_id.ps1 -- feasibility: differential response + identity discriminability
$ErrorActionPreference = "Continue"
$py   = "python"
$code = "$PSScriptRoot"
$data = "$env:PERTURB_DATA"
$log  = Join-Path $code "diff_id.log"

$files = @(
  "ReplogleWeissman2022_K562_essential.h5ad",
  "NadigOConner2024_jurkat.h5ad",
  "FrangiehIzar2021_RNA.h5ad",
  "NormanWeissman2019_filtered.h5ad"
) | ForEach-Object { Join-Path $data $_ }

Set-Location $code
"=== diff_id start $(Get-Date) ===" | Out-File -Encoding utf8 $log
& $py "$code\differential_identity.py" @files *>> $log
"=== diff_id done $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
