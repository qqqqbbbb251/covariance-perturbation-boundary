# run_p3.ps1 -- final revision: reproduction table with target-gene-excluded
# scoring, and transparent mixed model (ICC).
$ErrorActionPreference = "Continue"
$py   = "python"
$code = "$PSScriptRoot"
$res  = "$(Join-Path (Split-Path $PSScriptRoot -Parent) "results")"
$data = "$env:PERTURB_DATA"
$log  = Join-Path $code "p3.log"

$files = @(
  "AissaBenevolenskaya2021.h5ad","ChangYe2021.h5ad","DatlingerBock2017.h5ad",
  "DatlingerBock2021.h5ad","FrangiehIzar2021_RNA.h5ad","NadigOConner2024_hepg2.h5ad",
  "NadigOConner2024_jurkat.h5ad","NormanWeissman2019_filtered.h5ad",
  "PapalexiSatija2021_eccite_RNA.h5ad","PapalexiSatija2021_eccite_arrayed_RNA.h5ad",
  "ReplogleWeissman2022_K562_essential.h5ad","ReplogleWeissman2022_rpe1.h5ad",
  "TianKampmann2019_day7neuron.h5ad","TianKampmann2019_iPSC.h5ad",
  "TianKampmann2021_CRISPRa.h5ad","TianKampmann2021_CRISPRi.h5ad"
) | ForEach-Object { Join-Path $data $_ }

Set-Location $code
"=== p3 start $(Get-Date) ===" | Out-File -Encoding utf8 $log
& $py "$code\cipher_repro_table.py" @files *>> $log
& $py "$code\mixed_effects.py" *>> $log
"=== p3 done $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
