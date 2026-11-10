# How many decisions share one circuit shape? A contraction path (classical) or a transpiled layout (QPU) is found once per shape
# (gate sequence without angles) and reused for every decision with that shape, so the one-off search/compile energy is paid once per
# shape. Compiles every decision of the std test set and counts distinct shapes of the text circuit.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/shape_reuse.py -cp ckpt -o out.csv
import argparse, csv, hashlib
from collections import Counter
from pathlib import Path
from g_count import frames

ap = argparse.ArgumentParser()
ap.add_argument('-cp', '--checkpoint', required=True)
ap.add_argument('-o', '--out', required=True)
a = ap.parse_args()
config, qlimit, fr = frames(Path(a.checkpoint), 0)
key = lambda qc: hashlib.md5(repr([(ci.operation.name, tuple(qc.find_bit(q).index for q in ci.qubits)) for ci in qc.data]).encode()).hexdigest()
rows = []
for i, ((qt, qip, oq), (qt2, qin, _)) in fr:
    rows.append((i, qt.num_qubits, key(qt), key(qip), key(qin)))
C = Counter(r[2] for r in rows); N = len(rows); U = len(C)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['decision', 'nq', 'text_shape', 'img_pos_shape', 'img_neg_shape', 'decisions_with_shape'])
    for r in rows: w.writerow(list(r) + [C[r[2]]])
sizes = sorted(C.values(), reverse=True)
print(f'decisions {N}, distinct text-circuit shapes {U}, decisions per shape {N / U:.2f}; largest shape {sizes[0]}, top-10 shapes cover {sum(sizes[:10]) / N:.1%}, '
      f'singleton shapes {sum(1 for s in sizes if s == 1)} ({sum(1 for s in sizes if s == 1) / N:.1%} of decisions)', flush=True)
print('image circuit shapes:', len(set(r[3] for r in rows) | set(r[4] for r in rows)), flush=True)
W = Counter(r[1] for r in rows)
for nq, n in sorted(W.items()): print(f'  width {nq}: {n} decisions, {len(set(r[2] for r in rows if r[1] == nq))} shapes', flush=True)
print('done', flush=True)
