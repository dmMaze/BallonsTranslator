@echo off
cd /d "%~dp0"
echo Running BallonsTranslator Local Installer...
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
pause
