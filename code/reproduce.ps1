<#
reproduce.ps1 -- one-command reproduction helper.

Layer 1 (default): verify every headline number against results/*.csv.
                   Fast, needs no raw data.

Layer 2:           rebuild results/*.csv and figures/*.png from the raw .h5ad files.

Usage:
  .\reproduce.ps1              # layer 1 (verify)
  .\reproduce.ps1 verify       # layer 1
  .\reproduce.ps1 all          # layer 2, full pipeline (slow)
  .\reproduce.ps1 skip-heavy   # layer 2, skip the slowest steps

Environment (both are honoured if already set):
  PERTURB_DATA   directory holding the .h5ad files
                 (default on this machine: %LOCALAPPDATA%\Temp\opencode)
  CIPHER_ROOT    path to the CIPHER source tree (default: <repo>\CIPHER)
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("verify", "all", "skip-heavy")]
    [string]$Mode = "verify"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$code = $PSScriptRoot
$py = "python"

# data dir: use the env var if set, else fall back to the local convention
if (-not $env:PERTURB_DATA) {
    $local = Join-Path $env:LOCALAPPDATA "Temp\opencode"
    if (Test-Path $local) { $env:PERTURB_DATA = $local }
}
if (-not $env:CIPHER_ROOT) {
    foreach ($cand in @((Join-Path $repo "CIPHER"),
                        (Join-Path $env:USERPROFILE "Downloads\CIPHER-main\CIPHER-main"))) {
        if (Test-Path $cand) { $env:CIPHER_ROOT = $cand; break }
    }
}

Write-Host "repo        : $repo"
Write-Host "PERTURB_DATA: $($env:PERTURB_DATA)"
Write-Host "CIPHER_ROOT : $($env:CIPHER_ROOT)"
Write-Host "mode        : $Mode"
Write-Host ""

switch ($Mode) {
    "verify" {
        & $py (Join-Path $code "verify_results.py")
    }
    "all" {
        if (-not $env:PERTURB_DATA) { throw "Set PERTURB_DATA to the folder containing the .h5ad files." }
        & $py (Join-Path $code "run_all.py")
    }
    "skip-heavy" {
        if (-not $env:PERTURB_DATA) { throw "Set PERTURB_DATA to the folder containing the .h5ad files." }
        & $py (Join-Path $code "run_all.py") --skip-heavy
    }
}
exit $LASTEXITCODE
