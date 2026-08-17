# NephroScan AI - Expo Launcher
# Starts Flask backend and exposes via LocalTunnel

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "==========================================="
Write-Host "  NephroScan AI - Expo Launcher"
Write-Host "==========================================="
Write-Host ""

# Step 1: Enter project directory
Write-Host "[1/4] Entering project directory..."
Set-Location $ProjectDir
Write-Host "  $ProjectDir"

# Step 2: Python environment
Write-Host "[2/4] Setting up Python..."
$venvDir = Join-Path $ProjectDir "venv"
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    Write-Host "  Creating venv..."
    python -m venv venv
}
& "$venvDir\Scripts\Activate.ps1"
pip install -r requirements.txt --quiet 2>&1 | Out-Null
Write-Host "  Ready."

# Step 3: Start Flask backend
Write-Host "[3/4] Starting Flask backend on port 5000..."
$env:PORT = "5000"
$backendProc = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $ProjectDir -PassThru -NoNewWindow
Write-Host "  PID: $($backendProc.Id)"

$ready = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -TimeoutSec 3 -ErrorAction Stop
        if ($health.status -eq "online") {
            $ready = $true
            break
        }
    } catch {}
    Write-Host "  Waiting... ($i/30)"
}

if (-not $ready) {
    Write-Host "  ERROR: Backend did not start."
    $backendProc | Stop-Process -Force
    exit 1
}

Write-Host "  Backend online. All models: $($health.all_models_loaded)"
Write-Host ""

# Step 4: Start LocalTunnel
Write-Host "[4/4] Starting LocalTunnel..."
Write-Host ""
Write-Host "==========================================="
Write-Host "  Open from any device:"
Write-Host "  Local:  http://localhost:5000"
Write-Host "  Tunnel: (URL appears below)"
Write-Host "==========================================="
Write-Host ""

npx localtunnel --port 5000

Write-Host ""
Write-Host "Tunnel ended. Shutting down..."
$backendProc | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done."
