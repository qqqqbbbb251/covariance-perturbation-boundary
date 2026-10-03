# run_overnight.ps1 -- overnight comprehensive pipeline.
# Order: new forward/reverse analyses on all datasets, then figures + verification.
$ErrorActionPreference = "Continue"
$py   = "python"
$code = "$PSScriptRoot"
$res  = "$(Join-Path (Split-Path $PSScriptRoot -Parent) "results")"
$data = "$env:PERTURB_DATA"
$log  = Join-Path $code "overnight.log"

$all22 = @(
  "AdamsonWeissman2016_GSM2406675_10X001.h5ad","AdamsonWeissman2016_GSM2406677_10X005.h5ad",
  "AissaBenevolenskaya2021.h5ad","ChangYe2021.h5ad","DatlingerBock2017.h5ad",
  "DatlingerBock2021.h5ad","DixitRegev2016_K562_TFs_7_days.h5ad","FrangiehIzar2021_RNA.h5ad",
  "GasperiniShendure2019_lowMOI.h5ad","JoungZhang2023_combinatorial.h5ad",
  "NadigOConner2024_hepg2.h5ad","NadigOConner2024_jurkat.h5ad",
  "NormanWeissman2019_filtered.h5ad","PapalexiSatija2021_eccite_arrayed_RNA.h5ad",
  "PapalexiSatija2021_eccite_RNA.h5ad","ReplogleWeissman2022_K562_essential.h5ad",
  "ReplogleWeissman2022_rpe1.h5ad","SrivatsanTrapnell2020_sciplex3.h5ad",
  "TianKampmann2019_day7neuron.h5ad","TianKampmann2019_iPSC.h5ad",
  "TianKampmann2021_CRISPRa.h5ad","TianKampmann2021_CRISPRi.h5ad"
) | ForEach-Object { Join-Path $data $_ }

$canon16 = @(
  "AissaBenevolenskaya2021.h5ad","ChangYe2021.h5ad","DatlingerBock2017.h5ad",
  "DatlingerBock2021.h5ad","FrangiehIzar2021_RNA.h5ad","NadigOConner2024_hepg2.h5ad",
  "NadigOConner2024_jurkat.h5ad","NormanWeissman2019_filtered.h5ad",
  "PapalexiSatija2021_eccite_RNA.h5ad","PapalexiSatija2021_eccite_arrayed_RNA.h5ad",
  "ReplogleWeissman2022_K562_essential.h5ad","ReplogleWeissman2022_rpe1.h5ad",
  "TianKampmann2019_day7neuron.h5ad","TianKampmann2019_iPSC.h5ad",
  "TianKampmann2021_CRISPRa.h5ad","TianKampmann2021_CRISPRi.h5ad"
) | ForEach-Object { Join-Path $data $_ }

function Log($m) {
  $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  $line | Out-File -Append -Encoding utf8 $log
}

Set-Location $code
"=== overnight start $(Get-Date) ===" | Out-File -Encoding utf8 $log

Log "differential_identity.py (all 22, 3 spaces)"
& $py "$code\differential_identity.py" @all22 *>> $log

Log "exp12_stats.py"
& $py "$code\exp12_stats.py" *>> $log

Log "spectrum_rank.py (all 22)"
& $py "$code\spectrum_rank.py" @all22 *>> $log

Log "inverse_official.py (canonical 16)"
& $py "$code\inverse_official.py" @canon16 *>> $log

Log "inverse_false_positive.py (all 22)"
& $py "$code\inverse_false_positive.py" @all22 *>> $log

Log "positive_controls_extra.py"
& $py "$code\positive_controls_extra.py" *>> $log

Log "fig_new.py"
& $py "$code\fig_new.py" *>> $log

Log "make_all_figures.py"
& $py "$code\make_all_figures.py" *>> $log

Log "fig15_16.py"
& $py "$code\fig15_16.py" *>> $log

Log "fig_summary_panel.py"
& $py "$code\fig_summary_panel.py" *>> $log

Log "verify_results.py"
& $py "$code\verify_results.py" *>> $log

Log "overnight done"
"=== overnight done $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
