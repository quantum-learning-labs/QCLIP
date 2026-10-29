"""Parameter count of the TN text model on a config's train+val vocabulary, for several dims (compile only, no training).
  PYTHONPATH=. python tools/tn/param_count.py configs/tn/ref_clip.yaml [d,out,bond,max_order[,word_dim]]...
  word_dim > 0 = decoupled model (step 3: shared word vectors + shared complex-type tensors)"""
import sys, yaml, math
from collections import Counter
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS
from modules.utils.general import load_pkl

cfg = yaml.safe_load(open(sys.argv[1]))
dfs = {s: load_pkl(cfg['dataset'][s]['text_path']) for s in ('train', 'val')}
COMBOS = [tuple(int(x) for x in c.split(',')) for c in sys.argv[2:]] or [(128, 512, 32, 2)]   # d,out,bond,max_order[,word_dim]
for d, out, bond, mo, *wd in COMBOS:
    w = wd[0] if wd else 0
    comp = TNCompiler({'n': d, 's': d, 'p': d, 'out': out, 'sem': w}, MPS(bond_dim=bond, max_order=mo), decouple=w > 0)
    shapes, uses = {}, Counter()
    for s, df in dfs.items():
        c = comp.compile_dataset(df.copy())
        for row in c['captions_symbols']:
            for sent in row:
                for t in sent:
                    shapes[t['name']] = tuple(t['shape']); uses[t['name']] += 1
    by_order = Counter()
    for k, sh in shapes.items(): by_order[len(sh)] += math.prod(sh)
    total = sum(by_order.values())
    root = sum(math.prod(sh) for k, sh in shapes.items() if out in sh)
    words = sum(math.prod(sh) for k, sh in shapes.items() if k.startswith('w:'))
    print(f"n=s=p={d} out={out} bond={bond} max_order={mo} word_dim={w}: word vectors {words/1e6:.1f}M, {len(shapes):,} tensors, {total/1e9:.2f} B params "
          f"(by order {dict((o, f'{v/1e9:.2f}B') for o, v in sorted(by_order.items()))}; tensors with the out leg {root/1e9:.2f}B); "
          f"fp32 params+grad+Adam = {16*total/2**30:.0f} GB; tensors used once: {sum(1 for v in uses.values() if v == 1):,}", flush=True)
