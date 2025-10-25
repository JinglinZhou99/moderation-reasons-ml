import argparse, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_split

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/splits/train.jsonl')
ap.add_argument('--out', default='models/lr/vectorizer.pkl')
args = ap.parse_args()

vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, sublinear_tf=True)
X_train, _ = load_split(args.train)
Xv = vec.fit_transform(X_train)
joblib.dump(vec, args.out)
print('Saved vectorizer to', args.out, 'shape', Xv.shape)
