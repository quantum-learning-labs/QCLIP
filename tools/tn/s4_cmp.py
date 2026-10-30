"""Compare steps.csv losses (and trainstats.csv if both have it) of TN runs: s4_cmp.py <run_dir_a> <run_dir_b> [epoch]"""
import csv, sys
def load(d, f, ep):
    try: rows = list(csv.DictReader(open(f"{d}/{f}")))
    except FileNotFoundError: return None
    return [r for r in rows if ep is None or r['epoch'] == ep]
a, b = sys.argv[1], sys.argv[2]; ep = sys.argv[3] if len(sys.argv) > 3 else None
for f, cols in (('steps.csv', ['loss']), ('trainstats.csv', ['g_word', 'g_type', 'g_text_other', 'g_image', 'pre_min', 'pre_med', 'pre_max'])):
    ra, rb = load(a, f, ep), load(b, f, ep)
    if ra is None or rb is None: print(f, 'missing in one run'); continue
    n = min(len(ra), len(rb)); same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(ra[:n], rb[:n]))
    dmax = max(abs(float(x[c]) - float(y[c])) for x, y in zip(ra[:n], rb[:n]) for c in cols) if n else float('nan')
    print(f"{f}: rows {len(ra)} vs {len(rb)}, identical {same}/{n}, max |diff| {dmax:.3g}")
