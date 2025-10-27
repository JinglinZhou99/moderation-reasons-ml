# src/train_lr.py
import argparse, os, json, joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from datasets import load_split, LABELS

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/splits/train.jsonl')
ap.add_argument('--val', default='data/splits/val.jsonl')
ap.add_argument('--vec', default='models/lr/vectorizer.pkl')
ap.add_argument('--outdir', default='models/lr')
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
vec = joblib.load(args.vec)

Xtr_text, Ytr = load_split(args.train)
Xva_text, Yva = load_split(args.val)
Ytr = np.array(Ytr); Yva = np.array(Yva)

Xtr = vec.transform(Xtr_text)
Xva = vec.transform(Xva_text)


estimators = []
for j, label in enumerate(LABELS):
    yj = Ytr[:, j]
    uniq = np.unique(yj)
    if len(uniq) < 2:

        const = int(uniq[0])

        p = 0.99 if const == 1 else 0.01
        estimators.append(("constant", p))
    else:
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        clf.fit(Xtr, yj)
        estimators.append(("lr", clf))


P = np.zeros((Xva.shape[0], len(LABELS)), dtype=float)
for j, (kind, est) in enumerate(estimators):
    if kind == "constant":
        P[:, j] = est
    else:
        proba = est.predict_proba(Xva)

        if proba.ndim == 1:

            P[:, j] = proba
        else:

            P[:, j] = proba[:, 1]


thr = {}
preds = np.zeros_like(P, dtype=int)
for j, label in enumerate(LABELS):
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        yhat = (P[:, j] >= t).astype(int)
        f1 = f1_score(Yva[:, j], yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thr[label] = round(float(best_t), 3)
    preds[:, j] = (P[:, j] >= best_t).astype(int)

micro = f1_score(Yva, preds, average='micro', zero_division=0)
macro = f1_score(Yva, preds, average='macro', zero_division=0)
print({'micro_f1': round(float(micro),3), 'macro_f1': round(float(macro),3), 'thresholds': thr})


joblib.dump({"estimators": estimators, "vectorizer_path": args.vec}, os.path.join(args.outdir, 'model.joblib'))
json.dump(thr, open(os.path.join(args.outdir, 'thresholds.json'), 'w'))
json.dump({'labels': LABELS}, open(os.path.join(args.outdir, 'label_config.json'), 'w'))
print('Saved LR model + thresholds to', args.outdir)
