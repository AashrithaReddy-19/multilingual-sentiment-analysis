# Waits for the Colab Model B result ZIP in Downloads, then: import -> final_report.md -> tests -> backend restart -> verified backup.
# Start:  powershell -ExecutionPolicy Bypass -File D:\NLP\sentiment_app\auto_import_model_b.ps1
# Log:    D:\NLP\backups\model_b_auto_import.log      Stop: end the "powershell" process running this script.
param(
  [string]$DownloadsDir = (Join-Path $env:USERPROFILE "Downloads"),
  [int]$WaitForFullMinutes = 45,          # after the small results ZIP appears, wait this long for the full ZIP (with model weights)
  [switch]$NoRestart,
  [switch]$NoBackup
)
$App = "D:\NLP\sentiment_app"
$Log = "D:\NLP\backups\model_b_auto_import.log"
function Log($m) { "$(Get-Date -Format s)  $m" | Out-File $Log -Append -Encoding utf8 }

function Get-StableZip($pattern) {
  $f = Get-ChildItem -Path $DownloadsDir -Filter $pattern -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $f) { return $null }
  $s1 = $f.Length; Start-Sleep -Seconds 10; $f.Refresh()
  if ($f.Length -eq $s1 -and $f.Length -gt 0) { return $f } else { return $null }   # still downloading
}

Log "watching $DownloadsDir for model_b_full_*.zip / model_b_results_*.zip"
$smallSeen = $null
while ($true) {
  if (Test-Path "$App\artifacts\multilingual\model_b\metrics_model_b.json") { Log "Model B results already present; nothing to import. Exiting."; exit 0 }
  $zip = Get-StableZip "model_b_full_*.zip"
  if (-not $zip) {
    $small = Get-StableZip "model_b_results_*.zip"
    if ($small) {
      if (-not $smallSeen) { $smallSeen = Get-Date; Log "found $($small.Name); waiting up to $WaitForFullMinutes min for the full ZIP with weights" }
      $downloading = Get-ChildItem -Path $DownloadsDir -Filter "model_b_full_*" -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -ne ".zip" }
      if (((Get-Date) - $smallSeen).TotalMinutes -ge $WaitForFullMinutes -and -not $downloading) { $zip = $small }
    }
  }
  if ($zip) { break }
  Start-Sleep -Seconds 30
}

Log "importing $($zip.FullName)"
Push-Location "$App\backend"
$out = & python -m pipeline.import_colab_results "$($zip.FullName)" 2>&1; $code = $LASTEXITCODE
$out | ForEach-Object { Log "  import: $_" }
if ($code -ne 0) { Log "IMPORT REFUSED/FAILED (exit $code). Nothing else was changed."; Pop-Location; exit 1 }

$out = & python -m pytest "$App\tests" -q -W ignore::DeprecationWarning 2>&1; $tcode = $LASTEXITCODE
$out | Select-Object -Last 3 | ForEach-Object { Log "  tests: $_" }
Pop-Location

if (-not $NoRestart) {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*uvicorn*app.main*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  Start-Process cmd.exe -ArgumentList "/k", "$App\start_backend.bat" -WindowStyle Minimized
  Log "backend restarted (the app now loads the multilingual model when its weights were imported)"
}
if (-not $NoBackup) {
  Push-Location "$App\backend"
  $out = & python -m pipeline.make_backup 2>&1; $bcode = $LASTEXITCODE
  $out | ForEach-Object { Log "  backup: $_" }
  Pop-Location
  if ($bcode -ne 0) { Log "BACKUP FAILED (exit $bcode)" }
}
Log "DONE (tests exit code $tcode)"
