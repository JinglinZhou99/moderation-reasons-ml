import argparse, json, joblib, numpy as np
from datasets import LABELS

ap = argparse.ArgumentParser()
ap.add_argument('--text', required=True)
ap.add_argument('--vec', default='models/lr/vectorizer.pkl')
ap.add_argument('--model', default='models/lr/model.joblib')
ap.add_argument('--topk', type=int, default=5)
args = ap.parse_args()

vec = joblib.load(args.vec)
clf = joblib.load(args.model)
X = vec.transform([args.text])
explanations = {}
for j, label in enumerate(LABELS):
    est = clf.estimators_[j]
    coef = est.coef_.ravel()
    try:
        feats = vec.get_feature_names_out()
    except Exception:
        feats = np.arange(len(coef))
    x = X.toarray().ravel()
    contrib = coef * x
    top_idx = np.argsort(contrib)[-args.topk:][::-1]
    top = [(str(feats[i]), float(contrib[i])) for i in top_idx if contrib[i] > 0]
    explanations[label] = top

print(json.dumps(explanations, indent=2))
