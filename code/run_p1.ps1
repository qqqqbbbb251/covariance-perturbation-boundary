# run_p1.ps1 -- P1: composition/batch/guide control, residual decomposition,
# cross-dataset transfer.
$ErrorActionPreference = "Continue"
$py   = "python"
$code = "$PSScriptRoot"
$res  = "$(Join-Path (Split-Path $PSScriptRoot -Parent) "results")"
$data = "$env:PERTURB_DATA"
$log  = Join-Path $code "p1.log"

$files = @(
  "AissaBenevolenskaya2021.h5ad",
  "ChangYe2021.h5ad",
  "DatlingerBock2017.h5ad",
  "DatlingerBock2021.h5ad",
  "FrangiehIzar2021_RNA.h5ad",
  "NadigOConner2024_hepg2.h5ad",
  "NadigOConner2024_jurkat.h5ad",
  "NormanWeissman2019_filtered.h5ad",
  "PapalexiSatija2021_eccite_RNA.h5ad",
  "PapalexiSatija2021_eccite_arrayed_RNA.h5ad",
  "ReplogleWeissman2022_K562_essential.h5ad",
  "ReplogleWeissman2022_rpe1.h5ad",
  "TianKampmann2019_day7neuron.h5ad",
  "TianKampmann2019_iPSC.h5ad",
  "TianKampmann2021_CRISPRa.h5ad",
  "TianKampmann2021_CRISPRi.h5ad"
) | ForEach-Object { Join-Path $data $_ }

function Log($m) {
  $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  $line | Out-File -Append -Encoding utf8 $log
}

Set-Location $code
"=== p1 start $(Get-Date) ===" | Out-File -Encoding utf8 $log

Log "composition_control.py"
& $py "$code\composition_control.py" @files *>> $log

Log "residual_decomposition.py"
& $py "$code\residual_decomposition.py" @files *>> $log

Log "cross_dataset_transfer.py"
& $py "$code\cross_dataset_transfer.py" *>> $log

Log "p1 done"
"=== p1 done $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
