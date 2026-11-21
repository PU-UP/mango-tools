@echo off
setlocal
echo Preparing ChatGPT Connection Helper...
echo If the helper is running, exit it from the tray first. ChatGPT can stay open.
if not exist "%~dp0build.ps1" goto failed
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1"
if errorlevel 1 goto failed
if not exist "%~dp0dist\ChatGPT-Connection-Helper.exe" goto failed
echo Ready. Copy ChatGPT-Connection-Helper.exe to your desktop or another PC.
explorer.exe "%~dp0dist"
exit /b 0

:failed
echo Preparation failed. Extract the entire tool folder and check the message above.
echo No application was launched.
pause
exit /b 1
