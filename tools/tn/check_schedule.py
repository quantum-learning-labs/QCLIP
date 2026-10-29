"""Step 3: the vectorised levelwise planner (TNModel._schedule) must give exactly the step-2 planner's schedule (the
_levelwise planning code of 9caeb47, run verbatim): same index array, leaves, rounds and final spec, on many batches.
  PYTHONPATH=. python tools/tn/check_schedule.py <config> [n_batches] [batch_size]   (CPU only)"""
import sys, subprocess, textwrap, random, time, yaml
import numpy as np
from collections import defaultdict
from modules.utils.factory import build_experiment
from modules.utils.general import load_pkl
import modules.models.text.einsum_classical as ec

src = subprocess.check_output(['git', 'show', '9caeb47:modules/models/text/einsum_classical.py'], text=True).split('\n')
a = src.index('    def _levelwise(self, groups):'); b = next(i for i in range(a, len(src)) if 'idx = torch.from_numpy(np.concatenate(chunks))' in src[i])
body = textwrap.dedent('\n'.join(src[a + 1:b]))
ns = dict(ec.__dict__)
exec('def old_schedule(self, groups):\n' + textwrap.indent(body, '    ') + '\n    return np.concatenate(chunks), leaves, rounds, final\n', ns)
old_schedule = ns['old_schedule']

cfg = yaml.safe_load(open(sys.argv[1])); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 200; bs = int(sys.argv[3]) if len(sys.argv) > 3 else 512
ansatz, _, model, _ = build_experiment(cfg, 'cpu')
df = ansatz.compile_dataset(load_pkl(cfg['dataset']['train']['text_path']).iloc[:30000].copy())
model.from_symbols(df['captions_symbols'])
sents = [(e, s) for es, ss in zip(df['captions_einsum'], df['captions_symbols']) for e, s in zip(es, ss)]
model._plans, model._struct_ids, model._structs = {}, {}, []
rng = random.Random(0); t_old = t_new = 0.0
def same(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
    if isinstance(x, (list, tuple)):
        return type(x) == type(y) and len(x) == len(y) and all(same(p, q) for p, q in zip(x, y))
    return x == y
for k in range(nb):
    batch = rng.sample(sents, bs)
    groups = defaultdict(list)
    for i, (e, t) in enumerate(batch):
        sid, rows = model._plan(e, t); groups[sid].append((i, rows))
    t0 = time.perf_counter(); o = old_schedule(model, groups); t1 = time.perf_counter(); n = model._schedule(groups); t2 = time.perf_counter()
    t_old += t1 - t0; t_new += t2 - t1
    for name, x, y in zip(('idx', 'leaves', 'rounds', 'final'), o, n):
        if not same(x, y):
            raise SystemExit(f'batch {k}: {name} differs')
print(f'check_schedule {sys.argv[1]}: {nb} batches of {bs} identical | planning old {1e3 * t_old / nb:.1f} ms, new {1e3 * t_new / nb:.1f} ms per batch')
