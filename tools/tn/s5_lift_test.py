"""E2 lifting tests (GPU optional; run from qclip/ with PYTHONPATH=.):
 1. lift off: compile identical (einsum strings + symbols) to the compiler before E2 (5767954~1) on N captions;
 2. lift on: every compiled network is a valid einsum: each index has one dimension, the output index 0 is on exactly one
    tensor and every other index on >= 2 tensors; rule coverage (tokens per rule) on the sample;
 3. engines: levelwise == grouped == direct torch.einsum (fp64, raw TN output through the same head) on lifted captions;
 4. parameter counts with / without lifting, and a few lifted captions printed for reading.
   python tools/tn/s5_lift_test.py [N=20000] [--set k=v ...]   (base config: CP128, configs/tn/dec_t3.yaml)"""
import sys, subprocess, types, collections, random
import numpy as np, pandas as pd, torch, opt_einsum as oe
from modules.utils.general import setup_exp
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

N = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 20000
extra = [a.split('=', 1)[1] if a.startswith('--set=') else a for a in sys.argv[1:] if a not in (str(N),)]
sets = [extra[i + 1] for i, a in enumerate(extra) if a == '--set']
BASE = ['text.scale_free=false', 'text.head=no_bias', 'text.decomp=cp', 'text.cp_rank=128', 'text.share_args=leg_dist',
        'text.tn_engine=levelwise']
def build(lift, engine='levelwise'):
    ov = BASE + sets + ([f'text.lift={lift}'] if lift else []) + [f'text.tn_engine={engine}']
    config, DEV, SEED = setup_exp('configs/tn/dec_t3.yaml', overrides=ov)
    a, im, tm, _ = build_experiment(config, DEV)
    return config, DEV, SEED, a, tm

df = pd.read_pickle('data/mscoco/processed/tn/val_holdout.pkl')
dcol = [c for c in df.columns if c.endswith('_diagram')][0]
tns = [d for caps in df[dcol] for d in (caps or [])]
random.Random(0).shuffle(tns); tns = tns[:N]
print(f'{len(tns)} captions')

# 1. lift off == committed compiler
src = subprocess.run(['git', 'show', '5767954~1:modules/compilation/classical/tensor_network.py'], capture_output=True, text=True, check=True).stdout
old = types.ModuleType('old_tn'); exec(compile(src, 'old_tn', 'exec'), old.__dict__)
config, DEV, SEED, a_new, _ = build(None)
a_old = old.TNCompiler(obmap=a_new.obmap, decomp_fn=a_new.decomposition, decouple=True, orient=a_new.orient)
same = all(a_new.compile_tn(t) == a_old.compile_tn(t) for t in tns)
print('1. lift off == pre-E2 compiler:', same)

# 2. validity + coverage
config, DEV, SEED, a_lift, _ = build('frobenius')
bad, comp = 0, []
for t in tns:
    eq, syms = a_lift.compile_tn(t); comp.append((eq, syms))
    ins, out = eq.split('->'); ops = ins.split(',')
    dim, cnt = {}, collections.Counter()
    for sub, s in zip(ops, syms):
        assert len(sub) == len(s['shape']), (eq, s)
        for ch, d in zip(sub, s['shape']):
            if dim.setdefault(ch, d) != d: bad += 1
            cnt[ch] += 1
    if out != 'a' or cnt['a'] != 1 or any(c < 2 for ch, c in cnt.items() if ch != 'a'): bad += 1
st = a_lift.lift_stats
print('2. invalid networks:', bad, '| lift stats:', dict(st))
print('   share of word tokens lifted (R2-R6):', sum(v for k, v in st.items() if k.startswith(('R2', 'R3', 'R4', 'R5', 'R6'))) / st['words'],
      '| roots mapped (R1):', st['R1'] / st['captions'], '| roots kept:', st['root_kept'] / st['captions'])

# 3. engines on lifted captions (fp64, no centring / word_base: raw contraction + the same head)
torch.manual_seed(0)
res = {}
for eng in ('levelwise', 'grouped'):
    config, DEV, SEED, a, tm = build('frobenius', eng)
    de = DataEngine(config, a, DEV)
    cdf = pd.DataFrame({'captions_einsum': [[e for e, _ in comp[:2048]]], 'captions_symbols': [[s for _, s in comp[:2048]]]})
    de.text_init(tm, cdf)
    if eng == 'levelwise': state = {k: v.clone() for k, v in tm.state_dict().items()}
    else: tm.load_state_dict(state)
    tm.to(DEV).double().eval()
    batch = comp[:512]
    de.precompute_paths(tm, pd.DataFrame({'captions_einsum': [[e for e, _ in batch]], 'captions_symbols': [[s for _, s in batch]]}))
    with torch.no_grad(): res[eng] = tm(batch)
    if eng == 'levelwise':
        refs = []
        for eq, syms in batch[:64]:
            ts = []
            for s in syms:
                b, r = tm.sym_loc[s['name']]; ts.append(tm._bank(b)[r].reshape(s['shape']))
            refs.append(oe.contract(eq, *ts, backend='torch'))
        ref = torch.nn.functional.normalize(tm.norm_head(torch.stack(refs)), dim=-1)
        res['einsum'] = ref
cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1).min().item()
print('3. min cos levelwise/grouped:', cos(res['levelwise'], res['grouped']), '| levelwise/einsum (64):', cos(res['levelwise'][:64], res['einsum']))

# 4. params + examples
for lift in (None, 'frobenius'):
    config, DEV, SEED, a, tm = build(lift)
    de = DataEngine(config, a, DEV)
    cs = [a.compile_tn(t) for t in tns]
    de.text_init(tm, pd.DataFrame({'captions_einsum': [[e for e, _ in cs]], 'captions_symbols': [[s for _, s in cs]]}))
    print(f'4. params lift={lift}:', sum(p.numel() for p in tm.parameters()),
          '| distinct type tensors:', len({s["name"].split("_core")[0] for _, ss in cs for s in ss if s["name"].startswith("t")}))
shown = 0
for t, (eq, syms) in zip(tns, comp):
    names = [s['name'] for s in syms]
    if any(n.startswith('m:n->s') for n in names) and shown < 3:
        print('   caption:', ' '.join(w for w, _, _ in t)); print('     ', eq); print('     ', [n for n in names if not n.startswith('a:')]); shown += 1
