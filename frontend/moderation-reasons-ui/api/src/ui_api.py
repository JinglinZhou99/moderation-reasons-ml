import os, io, json, traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import f1_score

from .schemas import (
    PredictRequest, PredictResponse,
    ExplainRequest, ExplainResponse,
    MetricsResponse, BatchPredictResponse, BatchEvalResponse
)
from .loader import ModelBundle
from .utils import mock_probs, find_spans, reasons_from_spans, KEYS

# -------------------- Env --------------------
MOCK = os.environ.get("MOCK", "1") == "1"
MODEL_DIR = os.environ.get("MODEL_DIR", "models/lr")
BERT_DIR  = os.environ.get("BERT_DIR",  "models/bert")

# -------------------- App --------------------
app = FastAPI(title="Moderation with Reasons API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- LR loader --------------------
bundle = ModelBundle(mock=MOCK, model_dir=MODEL_DIR)
LABELS = getattr(bundle, "labels", ["violence", "sexual", "hate"])

# -------------------- Optional BERT loader --------------------
bert_available = False
bert_tokenizer = None
bert_model = None
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch  # noqa: F401
    # Where are tokenizer files? (root or subfolder)
    tok_dir = (
        BERT_DIR if os.path.exists(os.path.join(BERT_DIR, "tokenizer_config.json"))
        else os.path.join(BERT_DIR, "tokenizer")
    )
    if os.path.exists(os.path.join(BERT_DIR, "config.json")) and os.path.exists(tok_dir):
        bert_tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
        bert_model.eval()
        bert_available = True
except Exception:
    traceback.print_exc()
    bert_available = False

# -------------------- Helpers --------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _load_eval_json(model_dir: str) -> Optional[dict]:
    for name in ("eval_lr.json", "eval.json", "eval_bert.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None

# -------------------- Health --------------------
@app.get("/health")
def health():
    models = ["lr", "bert"] if bert_available else ["lr"]
    return {
        "ok": True,
        "mock": MOCK,
        "model_dir": MODEL_DIR,
        "bert_dir": BERT_DIR,
        "models": models,
        "labels": LABELS,
    }

# -------------------- Predict --------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    model_choice = (getattr(req, "model", None) or "lr").lower()

    # ---- BERT path ----
    if model_choice == "bert":
        if not bert_available:
            # Graceful fallback: zeros (frontend will show 0%)
            return PredictResponse(labels=LABELS, probs=[0.0]*len(LABELS), preds=[0]*len(LABELS))
        import torch
        with torch.no_grad():
            enc = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            logits = bert_model(**enc).logits.detach().cpu().numpy()[0]  # (L,)
            probs = _sigmoid(logits)
        # thresholds: from eval_bert.json if present, else 0.5
        thr_map = {l: 0.5 for l in LABELS}
        eval_bert = _load_eval_json(BERT_DIR)
        if isinstance(eval_bert, dict) and isinstance(eval_bert.get("thresholds"), dict):
            for l in LABELS:
                thr_map[l] = float(eval_bert["thresholds"].get(l, thr_map[l]))
        preds = [int(probs[i] >= thr_map[LABELS[i]]) for i in range(len(LABELS))]
        return PredictResponse(labels=LABELS, probs=[float(x) for x in probs], preds=preds)

    # ---- LR / Mock path ----
    if MOCK or bundle.model is None:
        probs_dict = mock_probs(text)
        probs = [float(probs_dict[l]) for l in LABELS]
        preds = [int(p >= 0.5) for p in probs]
        return PredictResponse(labels=LABELS, probs=probs, preds=preds)

    P = bundle.predict_probs([text])[0]
    thresholds = bundle.thresholds
    preds = [int(p >= thresholds.get(l, 0.5)) for p, l in zip(P, LABELS)]
    return PredictResponse(labels=LABELS, probs=[float(x) for x in P], preds=preds)

# -------------------- Explain --------------------
@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    text = req.text or ""
    spans_by_label: Dict[str, List[dict]] = {}
    if MOCK or bundle.model is None:
        for label in LABELS:
            spans = find_spans(text, KEYS[label])
            spans_by_label[label] = [{"start": s, "end": e, "text": text[s:e]} for s, e in spans]
    else:
        # Placeholder: LR/BERT IG can be added later
        for label in LABELS:
            spans_by_label[label] = []
    reasons = reasons_from_spans(text, {k: [(d["start"], d["end"]) for d in v] for k, v in spans_by_label.items()})
    return ExplainResponse(spans=spans_by_label, reasons=reasons)

# -------------------- Metrics --------------------
@app.get("/metrics", response_model=MetricsResponse)
def metrics(model: str = "lr"):
    md = BERT_DIR if (model or "lr").lower() == "bert" else MODEL_DIR
    eval_data = _load_eval_json(md)
    per_label: Dict[str, dict] = {}
    micro = macro = 0.0

    if eval_data is not None:
        micro = float(eval_data.get("micro_f1", 0.0))
        macro = float(eval_data.get("macro_f1", 0.0))
        ap = eval_data.get("ap", {})
        # thresholds: from thresholds.json and/or eval thresholds
        thr = {}
        thr_path = os.path.join(md, "thresholds.json")
        if os.path.exists(thr_path):
            try:
                thr = json.load(open(thr_path, "r", encoding="utf-8"))
            except Exception:
                thr = {}
        if "thresholds" in eval_data and isinstance(eval_data["thresholds"], dict):
            thr = {**thr, **eval_data["thresholds"]}
        for l in LABELS:
            per_label[l] = {"ap": float(ap.get(l, 0.0)), "threshold": float(thr.get(l, 0.5))}
    else:
        for l in LABELS:
            per_label[l] = {"ap": 0.0, "threshold": 0.5}

    return MetricsResponse(model=model, labels=LABELS, micro_f1=micro, macro_f1=macro, per_label=per_label)

# -------------------- Batch helpers --------------------
def _predict_texts_lr(texts: List[str]) -> np.ndarray:
    if MOCK or bundle.model is None:
        out = []
        for t in texts:
            pdict = mock_probs(t)
            out.append([pdict[l] for l in LABELS])
        return np.array(out, dtype=float)
    return bundle.predict_probs(texts)

def _predict_texts_bert(texts: List[str]) -> np.ndarray:
    if not bert_available:
        return np.zeros((len(texts), len(LABELS)), dtype=float)
    import torch
    probs_all = []
    with torch.no_grad():
        for t in texts:
            enc = bert_tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
            logits = bert_model(**enc).logits.detach().cpu().numpy()[0]
            probs = _sigmoid(logits)
            probs_all.append(probs)
    return np.vstack(probs_all)

def _predict_texts(texts: List[str], model: str) -> np.ndarray:
    return _predict_texts_bert(texts) if (model or "lr").lower() == "bert" else _predict_texts_lr(texts)

# -------------------- Batch predict --------------------
@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(file: UploadFile = File(...), model: str = Form("lr")):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "text" not in df.columns:
        return BatchPredictResponse(model=model, labels=LABELS, rows=[])
    texts = df["text"].astype(str).tolist()
    probs = _predict_texts(texts, model)

    # thresholds
    thr_map = {l: 0.5 for l in LABELS}
    if (model or "lr").lower() == "lr":
        thr_map.update(bundle.thresholds)
    else:
        ev = _load_eval_json(BERT_DIR)
        if ev and isinstance(ev.get("thresholds"), dict):
            for l in LABELS:
                thr_map[l] = float(ev["thresholds"].get(l, thr_map[l]))

    thr = np.array([thr_map[l] for l in LABELS])
    preds = (probs >= thr).astype(int)

    for i, l in enumerate(LABELS):
        df[f"p_{l}"] = probs[:, i]
        df[f"pred_{l}"] = preds[:, i]

    return BatchPredictResponse(model=model, labels=LABELS, rows=df.to_dict(orient="records"))

# -------------------- Batch eval --------------------
@app.post("/batch_eval", response_model=BatchEvalResponse)
async def batch_eval(file: UploadFile = File(...), model: str = Form("lr")):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "text" not in df.columns:
        return BatchEvalResponse(model=model, labels=LABELS, micro_f1=0.0, macro_f1=0.0, per_label={}, rows=[])
    texts = df["text"].astype(str).tolist()
    probs = _predict_texts(texts, model)

    thr_map = {l: 0.5 for l in LABELS}
    if (model or "lr").lower() == "lr":
        thr_map.update(bundle.thresholds)
    else:
        ev = _load_eval_json(BERT_DIR)
        if ev and isinstance(ev.get("thresholds"), dict):
            for l in LABELS:
                thr_map[l] = float(ev["thresholds"].get(l, thr_map[l]))

    thr = np.array([thr_map[l] for l in LABELS])
    preds = (probs >= thr).astype(int)

    per_label: Dict[str, dict] = {}
    if all(col in df.columns for col in LABELS):
        Y = df[LABELS].fillna(0).astype(int).to_numpy()
        micro = float(f1_score(Y, preds, average="micro", zero_division=0))
        macro = float(f1_score(Y, preds, average="macro", zero_division=0))
        for j, l in enumerate(LABELS):
            per_label[l] = {"f1": float(f1_score(Y[:, j], preds[:, j], zero_division=0))}
    else:
        micro = macro = 0.0
        for l in LABELS:
            per_label[l] = {"f1": 0.0}

    for i, l in enumerate(LABELS):
        df[f"p_{l}"] = probs[:, i]
        df[f"pred_{l}"] = preds[:, i]
    return BatchEvalResponse(model=model, labels=LABELS, micro_f1=micro, macro_f1=macro, per_label=per_label, rows=df.to_dict(orient="records"))
