# Moderation with Reasons — Teammate B (UI + API)

## Run API (mock mode)
```bash
cd api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run.sh    # or: uvicorn src.ui_api:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/health — should show `"mock": true`.

## Run UI
```bash
cd ../app
npm install
npm run dev
```
Open http://localhost:5173 and click **Predict**.

## Switch to real models later
Set env and start API using real artifacts from Teammate A:
```bash
export MOCK=0
export MODEL_DIR="/absolute/path/to/models/lr"
uvicorn src.ui_api:app --reload
```

## Docker (optional)
Use provided Dockerfiles to containerize `api` and `app`.
