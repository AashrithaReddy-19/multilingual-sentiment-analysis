@echo off
rem Starts the web app on http://localhost:5173 (installs packages on first run)
cd /d "%~dp0frontend"
if not exist node_modules call npm install
npm run dev
