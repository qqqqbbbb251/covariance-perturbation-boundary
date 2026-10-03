# run_p0.ps1 -- P0 rerun: 10-split matched-random reproduction table,
# hierarchical bootstrap (+global-random), mixed model (+global-random), figures.
$ErrorActionPreference = "Continue"
$py   = "python"
$code = "$PSScriptRoot"
$res  = "$(Join-Path (Split-Path $PSScriptRoot -Parent) "results")"
$data = "$env:PERTURB_DATA"
$log  = Join-Path $code "p0.log"

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
"=== p0 start $(Get-Date) ===" | Out-File -Encoding utf8 $log

Log "cipher_repro_table.py (10 splits, matched random, dual metric)"
& $py "$code\cipher_repro_table.py" @files *>> $log

Log "hierarchical_stats.py (+global-random)"
Set-Location $res
& $py "$code\hierarchical_stats.py" @files *>> $log

Log "mixed_effects.py (+global-random)"
& $py "$code\mixed_effects.py" *>> $log

Log "make_all_figures.py"
Set-Location $code
& $py "$code\make_all_figures.py" *>> $log

Log "p0 done"
"=== p0 done $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
