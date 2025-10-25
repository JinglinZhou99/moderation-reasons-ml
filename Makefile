.PHONY: toy lr bert explain

toy:
	python scripts/make_toy.py
	python scripts/make_splits.py --infile data/samples/toy.jsonl

lr: toy
	python src/featurize.py --train data/splits/train.jsonl --out models/lr/vectorizer.pkl
	python src/train_lr.py --train data/splits/train.jsonl --val data/splits/val.jsonl --vec models/lr/vectorizer.pkl --outdir models/lr
	python src/eval.py --split data/splits/test.jsonl --out models/lr/eval_lr.json

bert: toy
	python src/train_bert.py --train data/splits/train.jsonl --val data/splits/val.jsonl --outdir models/bert

explain:
	python src/explain_lr.py --text "Go back to your country or I will hurt you."
	python src/explain_ig.py --model_dir models/bert --text "Go back to your country or I will hurt you." --label hate
