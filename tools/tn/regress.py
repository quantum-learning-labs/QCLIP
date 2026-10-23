"""Step-2 regression check: does an optimised run reproduce the reference?

  python tools/tn/regress.py losses REF NEW [--tol 1e-5]   REF/NEW: steps.csv or profile_train.py JSON (first common steps)
  python tools/tn/regress.py bench  REF NEW [--tol 1e-5] [--metric_tol 1e-3]   REF/NEW: scripts/benchmark.py JSONs
      per-item scores (details: ARO/SVO pos/neg scores) must agree to --tol; metrics (timing keys ignored) to
      --metric_tol: accuracies/recalls are counts, so a rounding-level score change can flip a near-tie (1/n)
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse, csv, json, sys

TIMING = ('_eval_s', '_rows_per_s', '_peak_mem_gb')

def load_losses(path):
    if path.endswith('.json'):
        return [float(x) for x in json.load(open(path))['losses']]
    return [float(r['loss']) for r in csv.DictReader(open(path))]

def cmp_losses(a, b, tol):
    n = min(len(a), len(b))
    d = [abs(x - y) for x, y in zip(a[:n], b[:n])]
    first = lambda t: next((i for i, x in enumerate(d) if x > t), None)
    exact = sum(x == 0.0 for x in d)
    worst = max(range(n), key=lambda i: d[i]) if n else None
    print(f"steps compared {n} (ref {len(a)}, new {len(b)}) | bit-identical {exact}/{n} | max |dloss| {max(d) if d else 0:.3e}"
          f" at step {worst} | first step >1e-6: {first(1e-6)} | first >1e-5: {first(1e-5)} | first >1e-3: {first(1e-3)}")
    for k in (0, n // 4, n // 2, 3 * n // 4, n - 1):
        if 0 <= k < n: print(f"  step {k}: ref {a[k]!r} new {b[k]!r}")
    return max(d, default=0.0) <= tol

def cmp_bench(a, b, tol, metric_tol):
    ja, jb = json.load(open(a)), json.load(open(b))
    ra, rb = ja['results'], jb['results']
    worst_s, n_s = 0.0, 0
    for split in sorted(set(ja.get('details', {})) & set(jb.get('details', {}))):
        for k in ('pos_scores', 'neg_scores'):
            xa, xb = ja['details'][split].get(k, []), jb['details'][split].get(k, [])
            if len(xa) != len(xb): print(f"  {split}.{k}: length {len(xa)} vs {len(xb)}"); return False
            d = max((abs(float(x) - float(y)) for x, y in zip(xa, xb)), default=0.0)
            worst_s = max(worst_s, d); n_s += len(xa)
    print(f"per-item scores compared {n_s} | max |diff| {worst_s:.3e}")
    keys = sorted(k for k in set(ra) & set(rb) if not k.endswith(TIMING))
    only = sorted(k for k in set(ra) ^ set(rb) if not k.endswith(TIMING))
    worst = 0.0
    for k in keys:
        dv = abs(float(ra[k]) - float(rb[k])); worst = max(worst, dv)
        if dv > 0: print(f"  {k:<45} ref {float(ra[k]):.6f} new {float(rb[k]):.6f} diff {dv:.2e}")
    print(f"metrics compared {len(keys)} | identical {sum(float(ra[k]) == float(rb[k]) for k in keys)} | max |diff| {worst:.3e}"
          + (f" | in one file only: {only}" if only else ""))
    return worst <= metric_tol and worst_s <= tol

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['losses', 'bench'])
    ap.add_argument('ref'); ap.add_argument('new')
    ap.add_argument('--tol', type=float, default=1e-5)
    ap.add_argument('--metric_tol', type=float, default=1e-3)
    args = ap.parse_args()
    ok = (cmp_losses(load_losses(args.ref), load_losses(args.new), args.tol) if args.kind == 'losses'
          else cmp_bench(args.ref, args.new, args.tol, args.metric_tol))
    print("PASS" if ok else "FAIL", f"(tol {args.tol}" + (f", metric_tol {args.metric_tol})" if args.kind == "bench" else ")"))
    sys.exit(0 if ok else 1)
