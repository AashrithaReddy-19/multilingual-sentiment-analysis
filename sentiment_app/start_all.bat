@echo off
rem Opens the backend and the frontend in two windows, then the browser.
start "Sentiment API" cmd /k "%~dp0start_backend.bat"
start "Sentiment Web" cmd /k "%~dp0start_frontend.bat"
timeout /t 8 /nobreak >nul
start http://localhost:5173
