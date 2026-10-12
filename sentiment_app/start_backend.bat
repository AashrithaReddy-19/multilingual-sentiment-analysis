@echo off
rem Starts the API on http://127.0.0.1:8000  (docs: http://127.0.0.1:8000/docs)
rem Uses Python 3.11 (the project's single environment); other installed Pythons lack torch/transformers.
cd /d "%~dp0backend"
where py >nul 2>nul && (py -3.11 -c "import fastapi" >nul 2>nul) && (
    py -3.11 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    goto :eof
)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
