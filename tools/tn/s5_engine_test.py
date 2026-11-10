"""Step-5 check of the levelwise engine rewrite (unique-row leaf pieces, broadcast single-row operands, freed pieces) against a
reference copy of the engine (tools/tn/.ref_einsum_<sha>.py = einsum_classical.py at that commit: its _schedule, _levelwise,
_centred and _Take are bound to the same model). Forward (eval) outputs, and in training the output, the RowAdam row
gradients per bank and the extra/head parameter gradients. GPU.
   PYTHONPATH=. python tools/tn/s5_engine_test.py <ref.py> (--ckpt <best.pt> | -cfg <yaml> --set k=v ...) [--batches 4] [--bs 512]"""
import argparse, types, importlib.util, copy
from pathlib import Path
import torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('--ckpt'); ap.add_argument('-cfg', '--config')
ap.add_argument('--set', action='append', default=[]); ap.add_argument('--batches', type=int, default=4); ap.add_argument('--bs', type=int, default=512); ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
spec = importlib.util.spec_from_file_location('ref_ec', args.ref); ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
cfg_path = str(Path(args.ckpt).parent / 'config.yaml') if args.ckpt else args.config
config, DEV, SEED = setup_exp(cfg_path, seed=args.seed, overrides=args.set)
ansatz, image_model, tm, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
comp = de.compile_text('coco5k')
if args.ckpt:
    CheckpointManager.load_model(args.ckpt, image_model, tm, DEV)
else:
    import pandas as pd
    de.text_init(tm, de.compile_text('val'))
caps = [c for e, s in zip(comp['captions_einsum'], comp['captions_symbols']) for c in zip(e, s)]
caps = [c for c in caps if all(d['name'] in tm.sym_loc for d in c[1])]
de.precompute_paths(tm, comp); tm = tm.to(DEV)
if not args.ckpt:   # a fresh model: give word_base / head non-trivial values like a trained one
    pass
NEW = {n: getattr(tm, n) for n in ('_schedule', '_levelwise', '_centred')}
OLD = {n: types.MethodType(getattr(ref.TNModel, n), tm) for n in NEW}
def use(d):
    for n, f in d.items(): setattr(tm, n, f)

params = {n: p for n, p in tm.named_parameters() if not n.startswith('bank')}
def run(batch, train):
    tm.capture = []
    if not train:
        with torch.no_grad():
            out = tm(batch)
        return tm.capture[-1], out, None
    tm.train(); tm._active = []
    for p in params.values(): p.grad = None
    g = torch.Generator(device='cpu').manual_seed(0)
    out = tm(batch)
    (out * torch.randn(out.shape, generator=g).to(out.device)).sum().backward()
    bid = {id(tm._bank(k)): k for k in range(len(tm.bank_shapes))}
    grads = {f'bank{bid[id(b)]}': (rows.cpu(), grad.detach().cpu()) for b, rows, grad in tm.pop_row_grads()}
    grads.update({n: (None, p.grad.detach().cpu()) for n, p in params.items() if p.grad is not None})
    tm.eval()
    return tm.capture[-1], out.detach(), grads

worst, exact = {}, True
def cmp(k, a, b):
    global exact
    d = (a.double() - b.double()).abs().max().item(); s = b.double().abs().max().item()
    worst[k] = max(worst.get(k, 0.0), d / max(s, 1e-300)); exact &= bool(torch.equal(a, b))
for j in range(args.batches):
    batch = caps[j * args.bs:(j + 1) * args.bs]
    for train in (False, True):
        use(NEW); pn, on, gn = run(batch, train)
        use(OLD); po, oo, go = run(batch, train)
        t = '/train' if train else ''
        cmp('pre-head' + t, pn, po); cmp('embedding' + t, on, oo)
        if train:
            assert sorted(gn) == sorted(go), (sorted(gn), sorted(go))
            for k in go:
                if go[k][0] is not None:
                    assert torch.equal(gn[k][0], go[k][0]), f'rows differ in {k}'
                cmp('grad ' + k, gn[k][1], go[k][1])
for k, v in worst.items():
    print(f'{k:24s} max |new - ref| / max |ref| = {v:.3e}')
print(config['text'].get('decomp', 'mps'), 'bit-identical' if exact else 'NOT bit-identical', '| rel. tolerance 1e-5:', 'PASS' if max(worst.values()) < 1e-5 else 'FAIL')
