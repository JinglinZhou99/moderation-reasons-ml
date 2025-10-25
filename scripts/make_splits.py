import argparse, os, jsonlines, random
random.seed(42)

p = argparse.ArgumentParser()
p.add_argument('--infile', default='data/samples/toy.jsonl')
p.add_argument('--outdir', default='data/splits')
p.add_argument('--train', type=float, default=0.8)
p.add_argument('--val', type=float, default=0.1)
a = p.parse_args()

with jsonlines.open(a.infile) as r:
    data = list(r)
random.shuffle(data)
N = len(data)
tr = max(1, int(N * a.train))
va = max(1, int(N * a.val))

if tr + va >= N:
    va = 1
    tr = max(1, N - 2)
te = N - tr - va
splits = {'train': data[:tr], 'val': data[tr:tr+va], 'test': data[tr+va:]}
os.makedirs(a.outdir, exist_ok=True)
for k, v in splits.items():
    with jsonlines.open(os.path.join(a.outdir, f'{k}.jsonl'), 'w') as w:
        for e in v:
            w.write(e)
print('Wrote splits to', a.outdir)