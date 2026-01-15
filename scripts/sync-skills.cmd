@echo off
setlocal

REM Windows PowerShell 5.1 compatible entrypoint
REM Usage:
REM   scripts\sync-skills.cmd
REM   scripts\sync-skills.cmd -Target roo -Content skills

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync-skills.ps1" %*

endlocal
