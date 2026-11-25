"""Step 6 checks of text.tokens: bpe on real COCO captions (CP128 + R1 + centre, the step-5 recipe):
 (a) compile: one-piece words keep 'w:<word>'; multi-piece words compile to root 'w:<piece>-' + (a:tok.in, k:<piece>,
     a:tok.out) per extra piece; share of word uses that are multi-piece
 (b) engine = reference: with random values everywhere (token factors and modifiers too), the levelwise output equals a
     per-caption opt_einsum contraction (centre: minus the same network with every root/word = word_base and every
     modifier = 1), then the head
 (c) identity init (tokens_reset): every chain equals its root vector, i.e. the reference with each chain replaced by
     its root gives the same output
 (d) test vocabulary: share of coco5k word types / uses that are unseen words, and of those, covered by trained pieces
  PYTHONPATH=. python tools/tn/s6_tok_check.py <config.yaml> [--set k=v ...]"""
import sys, copy, torch, collections
import opt_einsum as oe
import torch.nn.functional as F
import pandas as pd
from modules.utils.general import setup_exp
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

ov = [sys.argv[i + 1] for i in range(2, len(sys.argv)) if sys.argv[i] == '--set'] + ['text.tokens=bpe']
cfg, DEV, _ = setup_exp(sys.argv[1], seed=0, overrides=ov)
df = pd.read_pickle('data/mscoco/processed/tn/val_holdout.pkl').head(200)
torch.manual_seed(0)
ansatz, _, tm, _ = build_experiment(cfg, DEV)
comp = ansatz.compile_dataset(df.copy())
de = DataEngine(cfg, ansatz, DEV); de.text_init(tm, comp); tm.to(DEV).eval()
caps = [(e, s) for es, ss in zip(comp['captions_einsum'], comp['captions_symbols']) for e, s in zip(es, ss)]
diags = [d for ds in df['captions_diagram'] for d in ds]
# (a)
P = ansatz.pieces; bad = multi = uses = 0
for (e, syms), diag in zip(caps, diags):
    names = [s['name'] for s in syms]
    for w, _, _ in diag:
        uses += 1; ps = P(w)
        if len(ps) <= 1: bad += ('w:' + w) not in names
        else:
            multi += 1; bad += ('w:' + ps[0] + '-') not in names or any(('k:' + p) not in names for p in ps[1:])
print(f'(a) word uses {uses}, multi-piece {multi} ({100 * multi / uses:.1f}%), compile mismatches {bad}; '
      f'symbols {len(tm.symbols)}, banks {[tuple(k) for k in tm.bank_keys]}, token bank {tm._tok_bank}')

def reference(tm, caps, collapse=False):
    wb = tm.extra['word_base']; outs = []
    for e, syms in caps:
        ins, outs_sub = e.split('->'); subs = ins.split(',')
        def val(s, base):
            b, r = tm.sym_loc[s['name']]; row = tm._bank(b)[r]
            if s['name'].startswith('w:'): return wb if base else row + wb
            if s['name'].startswith('k:'): return torch.ones_like(row) if base else row
            return row
        def contract(base):
            ts, ss = [], []
            if not collapse:
                return oe.contract(e, *[val(s, base) for s in syms], backend='torch')
            # collapse chains: drop a:tok.* / k:* and give each root the index of its chain's end wire
            # chain wires: a:tok.in (x, r) -> k (r) -> a:tok.out (r, y): map x -> y
            nxt = {}
            for j, s in enumerate(syms):
                if s['name'] == 'a:tok.in': r = subs[j][1]; x = subs[j][0]; nxt[r] = [x, None]
            for j, s in enumerate(syms):
                if s['name'] == 'a:tok.out': nxt[subs[j][0]][1] = subs[j][1]
            step = {x: y for x, y in nxt.values()}
            for j, s in enumerate(syms):
                if s['name'].startswith(('a:tok.', 'k:')): continue
                sub = subs[j]
                if s['name'].startswith('w:') and sub in step:
                    while sub in step: sub = step[sub]
                ts.append(val(s, base)); ss.append(sub)
            return oe.contract(','.join(ss) + '->' + outs_sub, *ts, backend='torch')
        y = contract(False)
        if tm.centre: y = y - contract(True)
        outs.append(y)
    return F.normalize(tm.norm_head(torch.stack(outs)), dim=-1)

with torch.no_grad():
    for b in range(len(tm.bank_shapes)): tm._bank(b).add_(torch.randn_like(tm._bank(b)) * 0.1)
    tm.extra['word_base'].add_(torch.randn_like(tm.extra['word_base']) * 0.1)
    eng = tm(caps); ref = reference(tm, caps)
    print(f'(b) engine vs reference (random): max |diff| {(eng - ref).abs().max().item():.2e}, min cos {(eng * ref).sum(1).min().item():.6f}')
    n = tm.tokens_reset()
    eng = tm(caps); ref = reference(tm, caps); col = reference(tm, caps, collapse=True)
    print(f'(c) identity init ({n} rows reset): engine vs reference {(eng - ref).abs().max().item():.2e}, chain vs root-only {(ref - col).abs().max().item():.2e}')
# (d) test vocabulary against the training compile's words and pieces
tr = pd.read_pickle(cfg['dataset']['train']['text_path'])
trw = collections.Counter(w for ds in tr['captions_diagram'] for d in ds for w, _, _ in d)
pieces = {p for w in trw for p in P(w)}
import glob
te_path = cfg['dataset']['test']['coco5k']['text_path']
print('test text path', te_path)
if te_path:
    te = pd.read_pickle(te_path); tew = collections.Counter(w for ds in te['captions_diagram'] for d in ds for w, _, _ in d)
    oov = [w for w in tew if w not in trw]; cov = [w for w in oov if all(p in pieces for p in P(w))]
    print(f'(d) test word types {len(tew)}, unseen {len(oov)} ({sum(tew[w] for w in oov)} uses); covered by train pieces {len(cov)} '
          f'({sum(tew[w] for w in cov)} uses), multi-piece among unseen {sum(len(P(w)) > 1 for w in oov)}; e.g. {oov[:10]}')
