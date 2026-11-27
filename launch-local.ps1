$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'C:\Program Files\QGIS 3.44.1\apps\Python312\python.exe'
$url = 'http://127.0.0.1:8767/'
$healthUrl = "${url}api/health"

function Test-AetherMapServer {
    try {
        $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return $response.app -eq 'aethermap-road-demo'
    } catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "QGIS Python was not found at $python"
    exit 1
}

if (-not (Test-AetherMapServer)) {
    $listener = Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        Write-Error 'Port 8767 is occupied by another app. AetherMap was not opened.'
        exit 1
    }

    Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        ('"' + (Join-Path $root 'open-when-ready.ps1') + '"')
    )
    Set-Location -LiteralPath $root
    & $python sam_backend.py
    exit $LASTEXITCODE
}

Start-Process $url
