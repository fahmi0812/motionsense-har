#!/usr/bin/env bash
# Satu perintah untuk menjalankan web: bash run_web.sh  -> buka http://localhost:8000
cd "$(dirname "$0")/web"
exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
