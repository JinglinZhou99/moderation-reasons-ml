import argparse, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from datasets import LABELS

ap = argparse.ArgumentParser()
ap.add_argument('--model_dir', default='models/bert')
ap.add_argument('--text', required=True)
ap.add_argument('--label', choices=LABELS, default='hate')
ap.add_argument('--steps', type=int, default=50)
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.to(device); model.eval()

tok = AutoTokenizer.from_pretrained(args.model_dir+'/tokenizer')
enc = tok(args.text, return_tensors='pt')
input_ids = enc['input_ids'].to(device)
attn = enc['attention_mask'].to(device)

label_idx = LABELS.index(args.label)

def forward_func(input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, label_idx]

ig = IntegratedGradients(lambda ids: forward_func(ids, attn))
attr = ig.attribute(input_ids, n_steps=args.steps, internal_batch_size=1)
attr = attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()

tokens = tok.convert_ids_to_tokens(input_ids[0])
idx = np.argsort(attr)[::-1][:10]
print('Top tokens:', [tokens[i] for i in idx])
