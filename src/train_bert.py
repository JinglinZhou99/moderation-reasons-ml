import argparse, os, json, numpy as np, torch
from datasets import load_split, LABELS
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import f1_score

class TextDs(Dataset):
    def __init__(self, X, Y, tok, max_len):
        self.X, self.Y, self.tok, self.max_len = X, Y, tok, max_len
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        enc = self.tok(self.X[i], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item['labels'] = torch.tensor(self.Y[i], dtype=torch.float)
        return item

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/splits/train.jsonl')
ap.add_argument('--val', default='data/splits/val.jsonl')
ap.add_argument('--outdir', default='models/bert')
ap.add_argument('--model_name', default='distilbert-base-uncased')
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--lr', type=float, default=3e-5)
ap.add_argument('--bsz', type=int, default=16)
ap.add_argument('--max_len', type=int, default=256)
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Xtr,Ytr = load_split(args.train)
Xva,Yva = load_split(args.val)

tok = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS), problem_type='multi_label_classification')
model.to(device)

tr_ds = TextDs(Xtr,Ytr,tok,args.max_len)
va_ds = TextDs(Xva,Yva,tok,args.max_len)
tr = DataLoader(tr_ds, batch_size=args.bsz, shuffle=True)
va = DataLoader(va_ds, batch_size=args.bsz)

opt = AdamW(model.parameters(), lr=args.lr)
t_total = len(tr)*args.epochs
sched = get_linear_schedule_with_warmup(opt, int(0.1*t_total), t_total)
bce = torch.nn.BCEWithLogitsLoss()

best_macro, best_state = -1, None
for ep in range(args.epochs):
    model.train()
    for batch in tr:
        batch = {k:v.to(device) for k,v in batch.items()}
        out = model(**batch)
        loss = bce(out.logits, batch['labels'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); model.zero_grad()
    model.eval(); preds=[]; gold=[]
    with torch.no_grad():
        for batch in va:
            y = batch['labels'].numpy()
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch)
            p = torch.sigmoid(out.logits).cpu().numpy()
            preds.append(p); gold.append(y)
    P = np.vstack(preds); G = np.vstack(gold)
    macro = f1_score(G, (P>=0.5).astype(int), average='macro', zero_division=0)
    print(f'Epoch {ep+1}: macro-F1={macro:.3f}')
    if macro>best_macro: best_macro, best_state = macro, model.state_dict()

if best_state is not None: model.load_state_dict(best_state)
model.save_pretrained(args.outdir)
tok.save_pretrained(os.path.join(args.outdir,'tokenizer'))
json.dump({'labels':LABELS}, open(os.path.join(args.outdir,'label_config.json'),'w'))
print('Saved BERT to', args.outdir)
