"""Step-5 check: text.centre's merged levelwise pass (real + base rows in one schedule) vs the step-4c two-pass _centred
(imported from the tn-step4c worktree), on a trained checkpoint (GPU). Forward: pre-head outputs and final embeddings;
training: RowAdam row gradients per bank and the word_base gradient. Prints max |diff| and whether results are bit-identical.
   PYTHONPATH=. python tools/tn/s5_merge_test.py <checkpoint.pt> <old_einsum_classical.py> [batches=4]"""
import sys, types, importlib.util
from pathlib import Path
import torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

ck = Path(sys.argv[1]); nb = int(sys.argv[3]) if len(sys.argv) > 3 else 4
spec = importlib.util.spec_from_file_location('old_ec', sys.argv[2]); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
config, DEV, SEED = setup_exp(str(ck.parent / 'config.yaml'))
assert config['text'].get('centre'), 'needs a text.centre checkpoint'
ansatz, image_model, tm, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
CheckpointManager.load_model(ck, image_model, tm, DEV)
comp = de.compile_text('coco5k')
caps = [c for e, s in zip(comp['captions_einsum'], comp['captions_symbols']) for c in zip(e, s)]
caps = [c for c in caps if all(d['name'] in tm.sym_loc for d in c[1])]
de.precompute_paths(tm, comp); tm = tm.to(DEV)
new_c, old_c = tm._centred, types.MethodType(old.TNModel._centred, tm)

def run(batch, train):
    tm.capture = []
    if not train:
        with torch.no_grad():
            out = tm(batch)
        return tm.capture[-1], out, None
    tm.train(); tm._active = []; tm.extra['word_base'].grad = None
    g = torch.Generator(device='cpu').manual_seed(0)
    out = tm(batch)
    (out * torch.randn(out.shape, generator=g).to(out.device)).sum().backward()
    bid = {id(tm._bank(k)): k for k in range(len(tm.bank_shapes))}
    grads = {bid[id(b)]: (rows.cpu(), grad.detach().cpu()) for b, rows, grad in tm.pop_row_grads()}
    grads['word_base'] = (None, tm.extra['word_base'].grad.detach().cpu())
    tm.eval()
    return tm.capture[-1], out.detach(), grads

worst, exact = {}, True
def cmp(k, a, b):
    global exact
    d = (a.double() - b.double()).abs().max().item(); s = b.double().abs().max().item()
    worst[k] = max(worst.get(k, 0.0), d / max(s, 1e-300)); exact &= bool(torch.equal(a, b))

for j in range(nb):
    batch = caps[j * 512:(j + 1) * 512]
    for train in (False, True):
        tm._centred = new_c; pn, on, gn = run(batch, train)
        tm._centred = old_c; po, oo, go = run(batch, train)
        cmp('pre-head' + ('/train' if train else ''), pn, po); cmp('embedding' + ('/train' if train else ''), on, oo)
        if train:
            assert sorted(map(str, gn)) == sorted(map(str, go)), (list(gn), list(go))
            for k in go:
                if go[k][0] is not None:
                    assert torch.equal(gn[k][0], go[k][0]), f'rows differ in bank {k}'
                cmp(f'grad', gn[k][1], go[k][1])
for k, v in worst.items():
    print(f'{k:20s} max |new - old| / max |old| = {v:.3e}')
print('bit-identical' if exact else 'NOT bit-identical', '| rel. tolerance 1e-5:', 'PASS' if max(worst.values()) < 1e-5 else 'FAIL')
