@echo off
cd /d "%~dp0"
py -3.11 -m pytest tests -q || python -m pytest tests -q
cd frontend && npm run build
