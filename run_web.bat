@echo off
REM Windows: dobel klik atau jalankan run_web.bat lalu buka http://localhost:8000
cd /d %~dp0web
python -m uvicorn app:app --host 127.0.0.1 --port 8000
