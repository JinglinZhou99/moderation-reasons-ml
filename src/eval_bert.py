# src/eval_bert.py
import argparse, json, os
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["violence","sexual","hate"]

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "text": obj["text"],
                "y": [int(obj["labels"].get(l,0)) for l in LABELS],
            })
    return rows

def sigmoid(x): return 1/(1+np.exp(-x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", required=True)     # e.g. data/splits/val.jsonl or test.jsonl
    ap.add_argument("--out", default=None)        # default: <model_dir>/eval_bert.json
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.model_dir, "eval_bert.json")

    # tokenizer may be in model_dir or model_dir/tokenizer
    tok_dir = (args.model_dir if os.path.exists(os.path.join(args.model_dir,"tokenizer_config.json"))
               else os.path.join(args.model_dir,"tokenizer"))
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    rows = load_jsonl(args.split)
    P, Y = [], []
    import torch
    with torch.no_grad():
        for r in rows:
            enc = tokenizer(r["text"], return_tensors="pt", truncation=True, max_length=256)
            logits = model(**enc).logits.detach().cpu().numpy()[0]  # (3,)
            P.append(sigmoid(logits))
            Y.append(r["y"])

    P = np.asarray(P, dtype=float)        # (N,3)
    Y = np.asarray(Y, dtype=int)          # (N,3)

    thr = np.array([args.threshold]*len(LABELS))
    preds = (P >= thr).astype(int)

    micro = float(f1_score(Y, preds, average="micro", zero_division=0))
    macro = float(f1_score(Y, preds, average="macro", zero_division=0))
    ap = {}
    for j,l in enumerate(LABELS):
        try:
            ap[l] = float(average_precision_score(Y[:,j], P[:,j]))
        except Exception:
            ap[l] = 0.0

    out = {
        "micro_f1": micro,
        "macro_f1": macro,
        "ap": ap,
        "thresholds": {l: float(args.threshold) for l in LABELS}
    }
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path)
if __name__ == "__main__":
    main()
