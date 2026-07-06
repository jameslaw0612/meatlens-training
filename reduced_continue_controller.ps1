$ErrorActionPreference = 'Continue'
function Log([string]$msg) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path 'E:\Thesis Code\training_outputs\run_monitor\reduced_controller.log' -Value "[$ts] $msg"
}
Log 'Controller started.'
Log 'Baseline seed_metrics row count: 15'
Log 'Watching old PIDs: 23040, 11344'
while ($true) {
    $rowCount = 0
    if (Test-Path 'E:\Thesis Code\training_outputs\seed_metrics.csv') {
        try { $rowCount = (Import-Csv 'E:\Thesis Code\training_outputs\seed_metrics.csv').Count } catch { $rowCount = 0 }
    }
    $alive = @()
    foreach ($watchPid in @(23040,11344)) {
        $p = Get-Process -Id $watchPid -ErrorAction SilentlyContinue
        if ($p) { $alive += $watchPid }
    }
    if ($rowCount -gt 15) {
        Log "Detected new completed run boundary at row count $rowCount."
        break
    }
    if ($alive.Count -eq 0) {
        Log 'Old full-training processes already exited.'
        break
    }
    Start-Sleep -Seconds 30
}
foreach ($stopPid in @(23040,11344)) {
    $p = Get-Process -Id $stopPid -ErrorAction SilentlyContinue
    if ($p) {
        Log "Stopping old process PID=$stopPid"
        try { Stop-Process -Id $stopPid -Force -ErrorAction Stop } catch { Log "Failed to stop PID=$stopPid : $(.Exception.Message)" }
    }
}
Log 'Starting reduced MobileNetV3Small continuation run.'
$env:JUPYTER_ALLOW_INSECURE_WRITES = 'true'
& 'E:\Thesis Code\.venv-gpu\Scripts\python.exe' 'E:\Thesis Code\run_new3_notebook.py'
$exitCode = $LASTEXITCODE
Log "Reduced continuation finished with exit code $exitCode"
