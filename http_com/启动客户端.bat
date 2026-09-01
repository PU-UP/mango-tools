@echo off
setlocal
cd /d "%~dp0"

where pythonw.exe >nul 2>nul
if not errorlevel 1 (
    start "" pythonw.exe "%~dp0client.py"
    exit /b 0
)

where pyw.exe >nul 2>nul
if not errorlevel 1 (
    start "" pyw.exe -3 "%~dp0client.py"
    exit /b 0
)

where python.exe >nul 2>nul
if not errorlevel 1 (
    python.exe "%~dp0client.py"
    if errorlevel 1 pause
    exit /b
)

where py.exe >nul 2>nul
if not errorlevel 1 (
    py.exe -3 "%~dp0client.py"
    if errorlevel 1 pause
    exit /b
)

echo Python was not found. Install Python 3 with Tcl/Tk and add it to PATH.
pause
exit /b 1
