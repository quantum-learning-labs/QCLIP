"""Step-4 recall diagnosis: which tensors carry the 512-dim 'out' leg, their shapes (MPS bond on the way to the output?) and how
many captions use each root type.   s4_roots.py <checkpoint.pt> <compiled data .pkl>"""
import sys, collections, pickle, torch
sd = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['text']
shapes = {k[len('params.'):]: tuple(v.shape) for k, v in sd.items() if k.startswith('params.')}
out_syms = {k: s for k, s in shapes.items() if 512 in s}
base = collections.Counter(k.split('_core')[0] for k in out_syms)
print(f"{len(shapes)} tensors; {len(out_syms)} carry a 512 leg ({len(base)} distinct types)")
for k, s in sorted(out_syms.items())[:12]: print('  ', k, s)
df = pickle.load(open(sys.argv[2], 'rb'))
print('data columns', list(df.columns)[:10], len(df))
