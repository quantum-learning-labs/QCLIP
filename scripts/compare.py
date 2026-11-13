import argparse, glob, json
import numpy as np

# python scripts/compare.py "results/a/*/benchmark_best.json" "results/b/*/benchmark_best.json" [--metric std_hard_neg_acc]
# Two models over the same splits: runs are paired by (split id, training seed), and the comparison refuses sets of runs
# that do not cover the same splits and seeds (a difference in splits would be counted as a difference in models).
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Paired comparison of two sets of benchmark results")
    p.add_argument('a'); p.add_argument('b')
    p.add_argument('--metric', action='append', help="results key (repeatable); default every *_hard_neg_acc")
    args = p.parse_args()
    runs = []
    for pattern in (args.a, args.b):
        side = {}
        for f in sorted(glob.glob(pattern)):
            r = json.load(open(f))
            if not r.get('split'):
                raise SystemExit(f"{f}: no split record (trained before split records, or on per-split files)")
            key = (r['split']['id'], r['seed'])
            if key in side:
                raise SystemExit(f"{f}: split {key[0]} seed {key[1]} appears twice")
            side[key] = r['results']
        runs.append(side)
    if set(runs[0]) != set(runs[1]):
        raise SystemExit(f"the two sets cover different (split, seed) runs:\n only a: {sorted(set(runs[0]) - set(runs[1]))}\n"
                         f" only b: {sorted(set(runs[1]) - set(runs[0]))}")
    keys = sorted(runs[0])
    metrics = args.metric or sorted(k for k in runs[0][keys[0]] if k.endswith('hard_neg_acc'))
    print(f"{len(keys)} paired runs over {len({k[0] for k in keys})} splits")
    print(f"{'metric':28s} {'a':>15s} {'b':>15s} {'b - a (paired)':>18s}")
    for m in metrics:
        a, b = (np.array([side[k][m] for k in keys]) for side in runs)
        d = b - a
        print(f"{m:28s} {a.mean():.4f}±{a.std(ddof=1):.4f} {b.mean():.4f}±{b.std(ddof=1):.4f} {d.mean():+.4f}±{d.std(ddof=1):.4f}")
