# run_diff_id_all.ps1 -- Exp1/Exp2 on all 16 canonical datasets (3 spaces each)
$ErrorActionPreference = "Continue"
$py   = "python"
$code = "$PSScriptRoot"
$data = "$env:PERTURB_DATA"
$log  = Join-Path $code "diff_id_all.log"

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
"=== diff_id_all start $(Get-Date) ===" | Out-File -Encoding utf8 $log
& $py "$code\differential_identity.py" @files *>> $log
"=== diff_id_all done $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
