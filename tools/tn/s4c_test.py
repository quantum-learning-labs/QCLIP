"""Step-4c checks of the new composition options on a trained step-4 checkpoint (GPU):
 (1) text.centre under scale_free equals the float64 reference without scale_free: unit(TN(caption) - TN(caption, words = base));
 (2) word_base with every d_w = 0: the output equals the base-only pass (captions = their structure term);
 (3) training-mode backward reaches word_base, mix and the type rows; hybrid runs.
   PYTHONPATH=. python tools/tn/s4c_test.py <checkpoint.pt> [N=1024]"""
import sys, copy
from pathlib import Path
import torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

ck_path = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
config, DEV, SEED = setup_exp(str(ck_path.parent / 'config.yaml'))

ansatz, image_model, tm0, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
CheckpointManager.load_model(ck_path, image_model, tm0, DEV)
comp = de.compile_text('coco5k')
caps = [c for e, s in zip(comp['captions_einsum'], comp['captions_symbols']) for c in zip(e, s)]
names = set(tm0.sym_loc)
caps = [c for c in caps if all(d['name'] in names for d in c[1])][:N]
state = {k: v for k, v in tm0.state_dict().items()}

def model(**text):
    cfg = copy.deepcopy(config); cfg['text'].update(text)
    _, _, tm, _ = build_experiment(cfg, DEV)
    tm.from_state_dict({k: v for k, v in state.items() if k.startswith('params.')})
    missing, unexpected = tm.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith('extra.') or k.startswith('bow_proj') for k in missing), missing
    de.precompute_paths(tm, comp)
    return tm.to(DEV)

def groups_of(tm, batch):
    from collections import defaultdict
    if not hasattr(tm, '_plans'):
        tm._plans, tm._struct_ids, tm._structs = {}, {}, []
    g = defaultdict(list)
    for i, (eq, arr) in enumerate(batch):
        sid, rows = tm._plan(eq, arr); g[sid].append((i, rows))
    return g

ok = True
# (1) centre vs float64 reference
tm = model(scale_free=True, word_base=0.3, centre=True).eval()
torch.manual_seed(1)
with torch.no_grad():
    tm.extra['word_base'].copy_(torch.randn_like(tm.extra['word_base']) / 128 ** 0.5)
    b = tm._bank(tm._word_bank); b.mul_(0.3)   # d_w = 0.3 x the trained vectors
    tm.capture = []
    got = torch.cat([(tm(caps[j:j + 256]), tm.capture[-1])[1] for j in range(0, len(caps), 256)])
    tm.capture = None
    ref_m = copy.deepcopy(tm).double(); ref_m.scale_free = False
    # unit-norm type tensors: real and base pass share them, so both scale by the same positive factor (the centred direction
    # is unchanged) and the fp64 products of 20+ trained gains can no longer under/overflow
    for bk in range(len(ref_m.bank_shapes)):
        if bk != ref_m._word_bank:
            t = ref_m._bank(bk); t.div_(t.flatten(1).norm(dim=1).clamp_min(1e-300).view(-1, *([1] * (t.dim() - 1))))
    refs = []
    for j in range(0, len(caps), 256):
        batch = caps[j:j + 256]; g = groups_of(ref_m, batch)
        real = ref_m._levelwise(g)
        base = ref_m._levelwise(g, base_only=True)   # per sentence (no de-duplication), fp64, no scale_free
        refs.append(torch.nn.functional.normalize(real - base, dim=-1, eps=1e-300))   # (default eps 1e-12 > these norms)
    ref = torch.cat(refs)
cos = (got.double() * ref).sum(1)
print(f"(1) centre vs fp64 reference: cos min {cos.min():.6f} median {cos.median():.6f} n(cos<0.999) {(cos < 0.999).sum()} "
      f"non-finite ref {(~torch.isfinite(ref).all(1)).sum()}  (N={len(caps)})")
bad = cos.argsort()[:5].tolist()
with torch.no_grad():
    m64 = copy.deepcopy(tm).double()   # same code path (scale_free + log scales) in fp64
    wb_ = [caps[i] for i in bad]; g = groups_of(m64, wb_)
    c64 = m64._centred(g)
    r64 = ref_m._levelwise(groups_of(ref_m, wb_)); b64 = ref_m._levelwise(groups_of(ref_m, wb_), base_only=True)
    m32 = copy.deepcopy(tm); m32.scale_free = True
    r32, _ = m32._levelwise(groups_of(m32, wb_), track=True)
with torch.no_grad():
    for name, mm in (('trained', copy.deepcopy(tm).double()), ('unit-type', copy.deepcopy(ref_m))):
        mm.scale_free = True
        gg = groups_of(mm, caps[:512])
        _, lsr = mm._levelwise(gg, track=True); _, lsb = mm._levelwise(gg, track=True, base_only=True)
        nw = torch.tensor([sum(d['name'].startswith('w:') for d in c[1]) for c in caps[:512]], dtype=torch.float64)
        print(f"    log10 scale {name}: real min/med/max {lsr.min()/2.3026:.1f}/{lsr.median()/2.3026:.1f}/{lsr.max()/2.3026:.1f}  "
              f"base {lsb.min()/2.3026:.1f}/{lsb.median()/2.3026:.1f}/{lsb.max()/2.3026:.1f}  per word (median) {(lsr.cpu() / nw).median()/2.3026:.2f}")
for n, i in enumerate(bad):
    rel = ((r64[n] - b64[n]).norm() / r64[n].norm()).item()
    print(f"    worst {i}: cos {cos[i]:.4f} n_words {sum(d['name'].startswith('w:') for d in caps[i][1])} | fp64 same path vs ref "
          f"{(c64[n] * ref[i]).sum().item():.6f} | |real-base|/|real| {rel:.2e} | fp32 scale-free real vs fp64 unit-type real cos "
          f"{(torch.nn.functional.normalize(r32[n].double(), dim=0, eps=1e-300) @ torch.nn.functional.normalize(r64[n], dim=0, eps=1e-300)).item():.6f}")
ok &= bool(cos.min() > 0.999)

# (2) word_base with d = 0: output = base-only pass
tm = model(scale_free=True, word_base=0.3).eval()
with torch.no_grad():
    tm._bank(tm._word_bank).zero_()
    tm.extra['word_base'].copy_(torch.randn(tm.extra['word_base'].shape[0]) / 128 ** 0.5)
    batch = caps[:256]; g = groups_of(tm, batch)
    a = torch.nn.functional.normalize(tm._levelwise(g), dim=-1, eps=1e-30)
    c = torch.nn.functional.normalize(tm._levelwise(g, base_only=True), dim=-1, eps=1e-30)
d = (a - c).abs().max().item(); print(f"(2) d=0 vs base-only: max |diff| {d:.2e}"); ok &= d < 1e-5

# (3) gradients
for kw in (dict(word_base=0.3), dict(word_base=0.3, centre=True), dict(compose='hybrid')):
    tm = model(scale_free=True, **kw).train()
    if 'word_base' in tm.extra:
        with torch.no_grad(): tm.extra['word_base'].copy_(torch.randn(tm.extra['word_base'].shape[0]) / 128 ** 0.5)
    out = tm(caps[:256]); (out * torch.randn_like(out)).sum().backward()
    rows = tm.pop_row_grads()
    gx = {k: (float(v.grad.norm()) if v.grad is not None else None) for k, v in tm.extra.items()}
    gr = {tuple(bk.shape[1:]): float(gg.norm()) for bk, _, gg in rows}
    print(f"(3) {kw}: extra grads {gx}; row grads by shape {gr}")
    ok &= all(v is not None and v > 0 for v in gx.values()) and len(gr) >= 2
print('ALL OK' if ok else 'FAILED')
