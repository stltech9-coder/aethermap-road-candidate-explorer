$ErrorActionPreference = 'Stop'
$env:AETHERMAP_PORT = '8768'
Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)
& 'C:\Program Files\QGIS 3.44.1\apps\Python312\python.exe' sam_backend.py
