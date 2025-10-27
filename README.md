# Moderation with Reasons — API (LR & BERT)

FastAPI service that classifies text into **violence / sexual / hate**, returning per-label **probabilities** and **thresholded predictions**, plus simple **keyword-based reasons**.

---

## Requirements

- Python 3.10+  
- (Optional, for BERT serving) PyTorch + Transformers

Install dependencies into a virtual environment:

**Windows (PowerShell)**
```powershell
cd api
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# If you plan to serve BERT:
pip install torch transformers python-multipart
```

**macOS/Linux**
```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If you plan to serve BERT:
pip install torch transformers python-multipart
```

---

## Start the API

### Option A — Mock mode (no model files, for quick wiring)
```bash
# Windows PowerShell
$env:MOCK="1"
python -m uvicorn src.ui_api:app --host 0.0.0.0 --port 8000
```
```bash
# macOS/Linux
export MOCK=1
python -m uvicorn src.ui_api:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/health` → should show `"mock": true`.

---

### Option B — Serve LR model
**Expected LR artifact layout**
```
models/lr/
  vectorizer.pkl
  model.joblib
  thresholds.json
  label_config.json
  (optional) eval_lr.json
```

**Run**
```bash
# Windows PowerShell
$env:MOCK="0"
$env:MODEL_DIR="C:\absolute\path\to\models\lr"
python -m uvicorn src.ui_api:app --host 0.0.0.0 --port 8000
```
```bash
# macOS/Linux
export MOCK=0
export MODEL_DIR="/absolute/path/to/models/lr"
python -m uvicorn src.ui_api:app --host 0.0.0.0 --port 8000
```

`GET /health` → `"models": ["lr"]`.

---

### Option C — Serve LR + BERT together
**Expected BERT artifact layout**
```
models/bert/
  config.json
  pytorch_model.bin  (or model.safetensors)
  tokenizer/         (tokenizer.json, vocab.txt, tokenizer_config.json, …)
  label_config.json
  (optional) eval_bert.json  ← used for thresholds/metrics if present
```

**Run**
```bash
# Windows PowerShell
$env:MOCK="0"
$env:MODEL_DIR="C:\absolute\path\to\models\lr"
$env:BERT_DIR="C:\absolute\path\to\models\bert"
python -m uvicorn src.ui_api:app --host 0.0.0.0 --port 8000
```
```bash
# macOS/Linux
export MOCK=0
export MODEL_DIR="/absolute/path/to/models/lr"
export BERT_DIR="/absolute/path/to/models/bert"
python -m uvicorn src.ui_api:app --host 0.0.0.0 --port 8000
```

`GET /health` → `"models": ["lr","bert"]`.

> Tip (Windows): avoid `--reload` if another Python (e.g., Anaconda) is being picked up by the reloader. Always start from your venv.

---

## Environment variables (summary)

- `MOCK`  
  - `"1"` (default) → no real models; returns mock probabilities + keyword reasons  
  - `"0"` → use real models from `MODEL_DIR` / `BERT_DIR`
- `MODEL_DIR` → absolute path to LR artifacts directory
- `BERT_DIR` → absolute path to BERT artifacts directory

---

## Health check

```
GET /health
```

Example response:
```json
{
  "ok": true,
  "mock": false,
  "model_dir": "C:\\...\\models\\lr",
  "bert_dir": "C:\\...\\models\\bert",
  "models": ["lr","bert"],
  "labels": ["violence","sexual","hate"]
}
```

---

## Endpoints

### 1) Predict (single text)
```
POST /predict
Content-Type: application/json
```
Body:
```json
{ "text": "Go back to your country or I will hurt you.", "model": "lr" }
```
- `model` can be `"lr"` or `"bert"` (default `"lr"` if omitted).
- Response:
```json
{
  "labels": ["violence","sexual","hate"],
  "probs": [0.92, 0.03, 0.81],
  "preds": [1, 0, 1]
}
```
> Thresholds: LR reads `thresholds.json`. BERT uses `eval_bert.json.thresholds` if present, otherwise 0.5.

### 2) Explain (keyword-based reasons)
```
POST /explain
Content-Type: application/json
```
Body:
```json
{ "text": "Go back to your country or I will hurt you." }
```
Returns spans for simple keyword matches per label and short reason strings. (Model-agnostic; not gradient-based.)

### 3) Metrics
```
GET /metrics?model=lr
GET /metrics?model=bert
```
Reads `eval_lr.json` or `eval_bert.json` **if present** and returns micro/macro F1, per-label AP, and thresholds. If missing, zeros/defaults are returned.

### 4) Batch Predict (CSV)
```
POST /batch_predict   (multipart/form-data)
  file=<csv with a "text" column>
  model=lr|bert
```
Returns rows with added columns: `p_violence`, `p_sexual`, `p_hate`, `pred_*`.

### 5) Batch Eval (CSV)
```
POST /batch_eval   (multipart/form-data)
  file=<csv with "text" and optional label columns: violence, sexual, hate>
  model=lr|bert
```
Computes micro/macro F1 if labels exist; also returns per-row probabilities/predictions.

---

## Curl examples

**LR**
```bat
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I will hurt you\",\"model\":\"lr\"}"
```

**BERT**
```bat
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"nsfw nude content\",\"model\":\"bert\"}"
```

**Metrics**
```bat
curl "http://localhost:8000/metrics?model=bert"
```

**Batch predict (CSV)**
```bat
curl -F "file=@C:\path\to\texts.csv" -F "model=lr" http://localhost:8000/batch_predict
```