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

try {
    Invoke-RestMethod -Uri "${url}api/building-status" -TimeoutSec 3 | Out-Null
} catch {
    # A previous AetherMap process may still be serving the page without the
    # newer building endpoint. Keep its browser storage on 8767 and add a
    # separate loopback building service rather than replacing that process.
    $helperUrl = 'http://127.0.0.1:8768/api/building-status'
    try {
        Invoke-RestMethod -Uri $helperUrl -TimeoutSec 2 | Out-Null
    } catch {
        $occupied = Get-NetTCPConnection -LocalPort 8768 -State Listen -ErrorAction SilentlyContinue
        if ($occupied) {
            Write-Error 'Building helper port 8768 is occupied by another service.'
            exit 1
        }
        Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            ('"' + (Join-Path $root 'start-building-helper.ps1') + '"')
        )
        $ready = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            try {
                Invoke-RestMethod -Uri $helperUrl -TimeoutSec 2 | Out-Null
                $ready = $true
                break
            } catch { Start-Sleep -Milliseconds 500 }
        }
        if (-not $ready) {
            Write-Error 'The building model service did not start. Close the old AetherMap launcher and reopen this launcher.'
            exit 1
        }
    }
}

Start-Process $url
