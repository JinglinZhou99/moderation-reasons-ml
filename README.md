# Moderation with Reasons — Teammate A (ML)

## Setup
```bash
cd moderation-reasons-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Smoke test (toy data -> LR artifacts)
```bash
make lr
cat models/lr/eval_lr.json
```

Artifacts to handoff to Teammate B:
```
models/lr/
  vectorizer.pkl
  model.joblib
  thresholds.json
  label_config.json
```

## Optional: train BERT and check IG
```bash
make bert
make explain
```
