"""Step-5 memory probe (GPU): where the training memory of a TN text model goes. Same setup as tools/tn/profile_train.py (same
seed -> same first batches), then N training steps under a saved-tensors hook and one no-grad forward of 2,560 val captions.
Reports: peak memory (train step / text forward / no-grad val forward), bytes autograd keeps for backward grouped by the op that
made them, and the per-use leaf copies of the levelwise schedule (rows per use vs unique rows, bytes) for the last step.
Works on the step-4c code too (run from that worktree with this file's path).
   PYTHONPATH=. python tools/tn/s5_mem.py -cfg configs/tn/dec_t3.yaml --set ... -o out.json"""
import argparse, json, math, socket
from collections import defaultdict
from pathlib import Path
import pandas as pd, torch
from modules.utils.factory import build_experiment, build_optimizer
from modules.utils.general import setup_exp
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import ContrastiveTrainer

ap = argparse.ArgumentParser()
ap.add_argument('-cfg', '--config', required=True); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=3); ap.add_argument('--set', action='append', default=[])
ap.add_argument('-o', '--out', required=True)
args = ap.parse_args()
out = Path(args.out)
if out.exists(): raise FileExistsError(out)
config, DEV, SEED = setup_exp(args.config, seed=args.seed, overrides=args.set)
ansatz, image_model, tm, loss_fn = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV)
ctr, cval = de.compile_text('train'), de.compile_text('val')
de.text_init(tm, pd.concat([ctr, cval], ignore_index=True)); de.image_init(image_model)
tm.to(DEV); loss_fn.to(DEV)
loader = de.get_loader(ctr, split='train')
opt, _ = build_optimizer(config, tm, image_model, loss_fn)
trainer = ContrastiveTrainer(image_model, tm, opt, loss_fn, DEV)
GB = 2 ** 30

# per-use leaf copies: wrap _schedule to keep the last plan's leaves (bank, shape, unique span, use span)
last = {}
orig = tm._schedule
def sched(*a, **k):
    r = orig(*a, **k); last['leaves'] = r[1]; return r
tm._schedule = sched
fwd = {}
def hook(_m, _i, o):
    torch.cuda.synchronize(); fwd['alloc_after_text_fwd'] = torch.cuda.memory_allocated() / GB
    fwd['peak_in_text_fwd'] = torch.cuda.max_memory_allocated() / GB
tm.register_forward_hook(hook)

saved = defaultdict(lambda: [0, 0]); seen = set()
def pack(t):
    if t.is_cuda:
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key); s = saved[type(t.grad_fn).__name__ if t.grad_fn is not None else ('param/leaf' if t.requires_grad else 'const')]
            s[0] += key[1]; s[1] += 1
    return t
res = {'overrides': args.set, 'gpu': torch.cuda.get_device_name(), 'host': socket.gethostname(),
       'params_text': sum(p.numel() for p in tm.parameters()), 'steps': []}
for k in range(args.steps):
    trainer.max_steps = 1
    saved.clear(); seen.clear()
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated() / GB; torch.cuda.reset_peak_memory_stats()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        trainer.train_epoch(loader, de.eval_mapper)
    torch.cuda.synchronize()
    st = {'alloc_before_gb': base, 'peak_step_gb': torch.cuda.max_memory_allocated() / GB, **fwd,
          'saved_for_backward_gb': sum(v[0] for v in saved.values()) / GB,
          'saved_by_op': {n: [round(b / GB, 3), c] for n, (b, c) in sorted(saved.items(), key=lambda x: -x[1][0])[:12]}}
    lv = last.get('leaves', [])
    st['leaf_pieces'] = [{'bank': str(x[0]), 'shape': list(x[1]), 'unique_rows': int(x[2][1] - x[2][0]),
                          'uses': int(x[3][1] - x[3][0]) if len(x) > 3 else None,
                          'unique_gb': (x[2][1] - x[2][0]) * math.prod(x[1]) * 4 / GB,
                          'per_use_copy_gb': (x[3][1] - x[3][0]) * math.prod(x[1]) * 4 / GB if len(x) > 3 else 0.0}
                         for x in lv]   # step-5 engine: leaves are (bank, shape, unique span), no per-use copies
    st['leaf_unique_gb'] = sum(x['unique_gb'] for x in st['leaf_pieces']); st['leaf_per_use_gb'] = sum(x['per_use_copy_gb'] for x in st['leaf_pieces'])
    res['steps'].append(st)
    print(f"step {k}: peak {st['peak_step_gb']:.2f} GB, after text fwd {st.get('alloc_after_text_fwd', 0):.2f}, saved {st['saved_for_backward_gb']:.2f} | "
          f"leaf copies {st['leaf_per_use_gb']:.2f} GB for {st['leaf_unique_gb']:.3f} GB of unique cores | {st['saved_by_op']}", flush=True)

caps = [c for e, s in zip(cval['captions_einsum'], cval['captions_symbols']) for c in zip(e, s)][:2560]
torch.cuda.synchronize(); base = torch.cuda.memory_allocated() / GB; torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    tm.eval(); tm(caps); tm.train()
torch.cuda.synchronize()
res['val_nograd_2560'] = {'alloc_before_gb': base, 'peak_gb': torch.cuda.max_memory_allocated() / GB}
print('val no-grad 2560 captions:', res['val_nograd_2560'])
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'x') as f: json.dump(res, f, indent=1)
print('wrote', out)
