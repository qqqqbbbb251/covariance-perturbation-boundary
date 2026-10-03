# download_extra.ps1
# Background download of supplementary scPerturb datasets.
# Uses the /content URL form and connects to known Zenodo IPs (bypasses DNS).

$dir  = "$env:PERTURB_DATA"
$log  = Join-Path $dir "download_extra.log"
$base = "https://zenodo.org/api/records/13350497/files"
$ips  = @("188.185.48.75","137.138.52.235","137.138.153.219",
          "188.184.103.118","188.185.43.153","188.184.98.114")

# name -> size in MB (completion detected by size stability)
$files = [ordered]@{
  "DixitRegev2016_K562_TFs_7_days.h5ad"   = 244.8
  "GasperiniShendure2019_lowMOI.h5ad"     = 307.8
  "JoungZhang2023_combinatorial.h5ad"     = 795.4
  "SrivatsanTrapnell2020_sciplex3.h5ad"   = 2409.6
}

if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

function Log($m) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Add-Content -Path $log -Value $line
}

Log "=== download_extra start ==="

foreach ($name in $files.Keys) {
  $dst    = Join-Path $dir $name
  $url    = "$base/$name/content"
  $target = [int64]($files[$name] * 1MB)

  if ((Test-Path $dst) -and ((Get-Item $dst).Length -ge [int64]($target * 0.999))) {
    Log "already complete: $name"; continue
  }

  Log "downloading $name ($($files[$name]) MB)"
  $prev = -1; $stall = 0
  for ($i = 0; $i -lt 40; $i++) {
    $ip = $ips[$i % $ips.Count]
    & curl.exe --ssl-no-revoke -4 -L -C - -s -o $dst `
        --resolve "zenodo.org:443:$ip" --max-time 500 $url 2>$null | Out-Null
    $sz = if (Test-Path $dst) { (Get-Item $dst).Length } else { 0 }
    Log ("  {0} attempt {1}: {2:N1}/{3:N1} MB" -f $name, $i, ($sz/1MB), $files[$name])
    if ($sz -ge $target) { Log "COMPLETE $name"; break }
    if ($sz -eq $prev -and $sz -gt 0) { $stall++ } else { $stall = 0 }
    $prev = $sz
    if ($stall -ge 3) { Log "STALLED $name"; break }
    Start-Sleep -Seconds 2
  }
}

Log "=== download_extra finished ==="
Get-ChildItem $dir -Filter *.h5ad | ForEach-Object {
  Log ("have: {0,10:N1} MB  {1}" -f ($_.Length/1MB), $_.Name)
}
