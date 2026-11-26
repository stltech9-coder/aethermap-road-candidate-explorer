@echo off
setlocal
set "PYTHON_EXE=C:\Program Files\QGIS 3.44.1\apps\Python312\python.exe"
if not exist "%PYTHON_EXE%" (
  echo QGIS Python was not found at %PYTHON_EXE%
  pause
  exit /b 1
)
cd /d "%~dp0"
start "" "http://127.0.0.1:8767/"
"%PYTHON_EXE%" sam_backend.py
if errorlevel 1 pause
