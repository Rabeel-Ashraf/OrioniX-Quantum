export CORS_ALLOW_ORIGIN="http://localhost:5173"
PORT="${PORT:-8080}"
uvicorn OrionIX Quantum.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
