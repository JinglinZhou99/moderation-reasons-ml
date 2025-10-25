import jsonlines
from typing import List, Dict, Tuple
LABELS=["violence","sexual","hate"]

def read_jsonl(path: str):
    with jsonlines.open(path) as r:
        for obj in r: yield obj

def load_split(split_path: str):
    X, Y = [], []
    for ex in read_jsonl(split_path):
        X.append(ex['text'])
        y = [int(ex['labels'].get(l,0)) for l in LABELS]
        Y.append(y)
    return X, Y
