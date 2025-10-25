import argparse, joblib
from sklearn.calibration import CalibratedClassifierCV
from datasets import load_split

ap = argparse.ArgumentParser()
ap.add_argument('--val', default='data/splits/val.jsonl')
ap.add_argument('--vec', default='models/lr/vectorizer.pkl')
ap.add_argument('--model', default='models/lr/model.joblib')
ap.add_argument('--out', default='models/lr/model_calibrated.joblib')
args = ap.parse_args()

vec = joblib.load(args.vec)
clf = joblib.load(args.model)
Xv, Yv = load_split(args.val)
Xv = vec.transform(Xv)
cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
cal.fit(Xv, Yv)
joblib.dump(cal, args.out)
print('Saved calibrated model to', args.out)
