# src/eval.py（只改加载和预测部分）
import argparse, json, numpy as np, joblib
from sklearn.metrics import f1_score, average_precision_score
from datasets import load_split, LABELS

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='data/splits/test.jsonl')
ap.add_argument('--vec', default='models/lr/vectorizer.pkl')
ap.add_argument('--model', default='models/lr/model.joblib')
ap.add_argument('--thresholds', default='models/lr/thresholds.json')
ap.add_argument('--out', default='models/lr/eval_lr.json')
args = ap.parse_args()

vec = joblib.load(args.vec)
bundle = joblib.load(args.model)  # 这里是我们保存的 dict
estimators = bundle.get("estimators", None)
if estimators is None:
    # 兼容老格式（直接是 OneVsRest clf）
    clf = bundle
else:
    clf = None

X_text, Y = load_split(args.split)
Xv = vec.transform(X_text)

# 计算概率矩阵
if clf is not None:
    # 旧格式：OneVsRest
    probs_list = clf.predict_proba(Xv)
    P = np.vstack([p[:,1] if p.ndim==2 else p for p in probs_list]).T
else:
    # 新格式：逐标签
    P = np.zeros((Xv.shape[0], len(LABELS)), dtype=float)
    for j, (kind, est) in enumerate(estimators):
        if kind == "constant":
            P[:, j] = est
        else:
            proba = est.predict_proba(Xv)
            P[:, j] = proba[:,1] if proba.ndim==2 else proba

thr = json.load(open(args.thresholds))
preds = np.zeros_like(P, dtype=int)
for j,l in enumerate(LABELS):
    preds[:,j] = (P[:,j] >= thr.get(l,0.5)).astype(int)

micro = f1_score(Y, preds, average='micro', zero_division=0)
macro = f1_score(Y, preds, average='macro', zero_division=0)
aps = {l: float(average_precision_score(np.array(Y)[:,j], P[:,j])) for j,l in enumerate(LABELS)}

out = {'micro_f1': float(micro), 'macro_f1': float(macro), 'ap': aps}
json.dump(out, open(args.out,'w'), indent=2)
print('Wrote', args.out, out)
