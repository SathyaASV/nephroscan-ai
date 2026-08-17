# NephroScan AI — Expo Launcher
# Starts the real Python Flask backend and exposes it via LocalTunnel
# Educational prototype only. Not a medical device.

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  NephroScan AI — Expo Launcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Enter project directory
Write-Host "[1/6] Entering project directory..." -ForegroundColor Yellow
Set-Location $ProjectDir
Write-Host "  -> $ProjectDir" -ForegroundColor Gray

# Step 2: Check Python
Write-Host "[2/6] Checking Python..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "  ERROR: Python not found. Install Python 3.10+ and add to PATH." -ForegroundColor Red
    exit 1
}
$pyVer = python --version 2>&1
Write-Host "  -> $pyVer" -ForegroundColor Gray

# Step 3: Create/activate virtual environment and install deps
Write-Host "[3/6] Setting up Python environment..." -ForegroundColor Yellow
$venvDir = Join-Path $ProjectDir "venv"
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
}
Write-Host "  Activating venv..." -ForegroundColor Gray
& "$venvDir\Scripts\Activate.ps1"

Write-Host "  Installing requirements..." -ForegroundColor Gray
pip install -r requirements.txt --quiet 2>&1 | Out-Null
Write-Host "  -> Dependencies ready" -ForegroundColor Gray

# Step 4: Start Flask backend
Write-Host "[4/6] Starting Flask backend on 0.0.0.0:5000..." -ForegroundColor Yellow
$env:PORT = "5000"
$backendProc = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $ProjectDir -PassThru -NoNewWindow
Write-Host "  -> Backend PID: $($backendProc.Id)" -ForegroundColor Gray

# Step 5: Wait for /api/health
Write-Host "[5/6] Waiting for backend to be ready..." -ForegroundColor Yellow
$ready = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -TimeoutSec 3 -ErrorAction Stop
        if ($health.status -eq "online") {
            $ready = $true
            break
        }
    } catch {
        # Not ready yet
    }
    Write-Host "  Waiting... ($i/30)" -ForegroundColor Gray
}

if (-not $ready) {
    Write-Host "  ERROR: Backend did not start in 30 seconds." -ForegroundColor Red
    $backendProc | Stop-Process -Force
    exit 1
}

Write-Host "  -> Backend online! Models loaded: $($health.all_models_loaded)" -ForegroundColor Green
Write-Host "  -> Health: http://127.0.0.1:5000/api/health" -ForegroundColor Gray

# Step 6: Start LocalTunnel
Write-Host "[6/6] Starting LocalTunnel..." -ForegroundColor Yellow

# Check if npx is available
$npx = Get-Command npx -ErrorAction SilentlyContinue
if (-not $npx) {
    Write-Host "  ERROR: npx not found. Install Node.js and npm first." -ForegroundColor Red
    Write-Host "  Download: https://nodejs.org/" -ForegroundColor Gray
    $backendProc | Stop-Process -Force
    exit 1
}

Write-Host "  Starting tunnel to port 5000..." -ForegroundColor Gray
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  EXPLORE FROM ANOTHER DEVICE:" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Local:    http://localhost:5000" -ForegroundColor White
Write-Host "  Tunnel:   (see URL below)" -ForegroundColor White
Write-Host ""
Write-Host "  The tunnel URL will appear below." -ForegroundColor Yellow
Write-Host "  Open it from any device on your network." -ForegroundColor Yellow
Write-Host "  The expo laptop must stay powered on." -ForegroundColor Yellow
Write-Host ""

# Start localtunnel
try {
    npx localtunnel --port 5000
} catch {
    Write-Host "  Tunnel ended." -ForegroundColor Yellow
}

# Cleanup
Write-Host ""
Write-Host "Shutting down backend..." -ForegroundColor Yellow
$backendProc | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Done." -ForegroundColor Green
