"""Step 5e checks of text.affine (homogeneous slots) on real COCO captions, CP128 + R1 + centre (the step-5 recipe):
 (a) engine = reference: with random nonzero biases, the levelwise output equals a direct per-caption torch.einsum over the
     masked tensors (word rows + word base, slot 1; centre = caption minus the all-base structure), then the head
 (b) pinned slots: every masked tensor satisfies its rule (word slot 1; CP column r* = e_h; output view row h = e_r*; dense
     output view: h-slice of the output leg = unit at all-h)
 (c) zero bias = multilinear model: after affine_reset_bias, copying the non-slot blocks into a plain (affine: false) model
     gives the same outputs up to float rounding
  PYTHONPATH=. python tools/tn/s5e_affine_check.py <config.yaml>"""
import sys, copy, torch
import opt_einsum as oe
import torch.nn.functional as F
import pandas as pd
from modules.utils.general import setup_exp
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine
from modules.models.text.affine import split_alias, row_mask, leg_kinds, is_cp_factor

cfg0, DEV, SEED = setup_exp(sys.argv[1])
df = pd.read_pickle('data/mscoco/processed/tn/val_holdout.pkl').head(150)

def build(affine):
    cfg = copy.deepcopy(cfg0); cfg['text']['affine'] = affine
    torch.manual_seed(0)
    ansatz, _, tm, _ = build_experiment(cfg, DEV)
    comp = ansatz.compile_dataset(df.copy())
    de = DataEngine(cfg, ansatz, DEV); de.text_init(tm, comp); tm.to(DEV).eval()
    caps = [(e, s) for es, ss in zip(comp['captions_einsum'], comp['captions_symbols']) for e, s in zip(es, ss)]
    return tm, caps

A, capsA = build(True)
names = [s['name'] for _, ss in capsA for s in ss]
print('affine model: symbols', len(A.symbols), 'banks', len(A.bank_shapes), 'uses', len(names), 'output-view uses', sum('^' in n for n in names), 'distinct aliases', len({n for n in names if '^' in n}))
with torch.no_grad():   # random biases everywhere (slot entries too; masks pin what must be pinned)
    for b in range(len(A.bank_shapes)): A._bank(b).add_(torch.randn_like(A._bank(b)) * 0.1)
    if A.extra is not None and 'word_base' in A.extra: A.extra['word_base'].add_(torch.randn_like(A.extra['word_base']) * 0.1)
A._masks = None

def masked(tm, name, base_word=False):
    base, view = split_alias(name)
    b, r = tm.sym_loc[base]; row = tm._bank(b)[r]
    if base.startswith('w:'):
        row = (tm.extra['word_base'] if base_word else row + tm.extra['word_base']) if tm.word_scale is not None else row
    M, C = row_mask(base, tuple(row.shape), view, tm.out_dim)
    return row * M.to(row.device) + C.to(row.device)

def reference(tm, caps, affine):
    outs = []
    for e, syms in caps:
        if affine:
            t = [masked(tm, s['name']) for s in syms]; tb = [masked(tm, s['name'], base_word=s['name'].startswith('w:')) for s in syms]
        else:
            t, tb = [], []
            for s in syms:
                b, r = tm.sym_loc[s['name']]; row = tm._bank(b)[r]
                if s['name'].startswith('w:') and tm.word_scale is not None:
                    t.append(row + tm.extra['word_base']); tb.append(tm.extra['word_base'])
                else:
                    t.append(row); tb.append(row)
        y = oe.contract(e, *t, backend='torch')
        if tm.centre: y = y - oe.contract(e, *tb, backend='torch')
        outs.append(y)
    return F.normalize(tm.norm_head(torch.stack(outs)), dim=-1)

with torch.no_grad():
    eng = A(capsA); ref = reference(A, capsA, True)
    print(f'(a) engine vs reference (random biases): max |diff| {(eng - ref).abs().max().item():.2e}, min cos {(eng * ref).sum(1).min().item():.6f}')
    bad = 0
    for name in {s['name'] for _, ss in capsA for s in ss}:
        base, view = split_alias(name); T = masked(A, name); k = leg_kinds(base, tuple(T.shape), A.out_dim)
        if base.startswith('w:'): bad += int(T[-1].item() != 1.0)
        elif is_cp_factor(base) and T.dim() == 2:
            if k[0] == 'h':
                bad += int(not torch.equal(T[:, -1], F.one_hot(torch.tensor(T.shape[0] - 1), T.shape[0]).to(T)))
                if view == 'o': bad += int(not torch.equal(T[-1], F.one_hot(torch.tensor(T.shape[1] - 1), T.shape[1]).to(T)))
        elif view not in (None, 'o') and all(x == 'h' for x in k):
            kk = int(view); sl = T.select(kk, T.shape[kk] - 1)
            unit = torch.zeros_like(sl); unit[tuple(-1 for _ in sl.shape)] = 1
            bad += int(not torch.equal(sl, unit))
    print(f'(b) pinned-slot violations: {bad}')
    # (c) zero bias = plain multilinear model
    A.affine_reset_bias(); A._masks = None
    P, capsP = build(False)
    import re
    def aff_name(n):   # shared factor names carry their shape: +1 on every non-sentence dim
        m = re.search(r'\.(\d+(?:x\d+)+)$', n)
        if not m: return n
        dims = [int(d) + (0 if int(d) == A.out_dim else 1) for d in m.group(1).split('x')]
        return n[:m.start(1)] + 'x'.join(map(str, dims))
    for name in P.symbols:
        bp, rp = P.sym_loc[name]; ba, ra = A.sym_loc[aff_name(name)]
        src = A._bank(ba)[ra]; dst = P._bank(bp)[rp]
        dst.copy_(src[tuple(slice(0, d) for d in dst.shape)])
    if P.extra is not None and 'word_base' in P.extra: P.extra['word_base'].copy_(A.extra['word_base'][:-1])
    P.norm_head.load_state_dict(A.norm_head.state_dict())
    oa, op = A(capsA), P(capsP)
    print(f'(c) zero-bias affine vs plain: max |diff| {(oa - op).abs().max().item():.2e}, min cos {(oa * op).sum(1).min().item():.6f}')
    rp = reference(P, capsP, False)
    print(f'    plain engine vs plain reference: max |diff| {(op - rp).abs().max().item():.2e}')
