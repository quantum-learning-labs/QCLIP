"""Step-2 efficiency table: one row per change from results/tn/step2/{prof,bench,regress}_<tag>.*

  python tools/tn/step2_table.py base:Baseline c1:"Change 1" ... [-o results/tn/step2/table.md]
Speed-ups are against the first tag (and against the previous row).
"""
import argparse, json, re
from pathlib import Path

R = Path('results/tn/step2')
EVAL = ['coco5k', 'sugarcrepe_replace_object', 'winoground', 'aro_rel']

def load(tag):
    prof = json.load(open(R / f'prof_{tag}.json'))
    bench = json.load(open(R / f'bench_{tag}.json')) if (R / f'bench_{tag}.json').exists() else None
    reg = (R / f'regress_{tag}.txt').read_text() if (R / f'regress_{tag}.txt').exists() else ''
    return prof, bench, reg

def regress_summary(reg):
    m = re.search(r'bit-identical (\d+)/(\d+) \| max \|dloss\| ([\d.e+-]+)', reg)
    b = re.search(r'metrics compared (\d+) \| identical (\d+)', reg)
    s = re.search(r'per-item scores compared \d+ \| max \|diff\| ([\d.e+-]+)', reg)
    out = []
    if m: out.append(f"loss {m.group(1)}/{m.group(2)} identical, max {float(m.group(3)):.1e}")
    if b: out.append(f"metrics {b.group(2)}/{b.group(1)} identical")
    if s: out.append(f"scores max {float(s.group(1)):.1e}")
    return '; '.join(out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+', help='tag[:label]')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()
    rows, base, prev = [], None, None
    hdr = ("| change | train samples/s | vs baseline | vs previous | step ms: fwd text / backward / optimizer / data | "
           "startup s | peak GB | " + " | ".join(f"{e} rows/s" for e in EVAL) + " | regression |")
    rows.append(hdr); rows.append("|" + "---|" * (hdr.count('|') - 1))
    for item in args.tags:
        tag, _, label = item.partition(':')
        prof, bench, reg = load(tag)
        sps = prof['samples_per_s']
        base = base or sps; vs_prev = f"{sps / prev:.2f}x" if prev else "-"
        ph = prof['phase_s_per_step']
        startup = sum(prof['startup_s'].values())
        ev = [f"{bench['results'].get(f'{e}_rows_per_s', float('nan')):.0f}" if bench else "-" for e in EVAL]
        rows.append(f"| {label or tag} ({prof['git_commit'][:7]}) | {sps:.0f} | {sps / base:.2f}x | {vs_prev} | "
                    f"{ph['fwd_text']*1e3:.0f} / {ph['backward']*1e3:.0f} / {ph['optimizer']*1e3:.0f} / {ph['data']*1e3:.0f} | "
                    f"{startup:.0f} | {prof['peak_mem_gb']:.1f} | " + " | ".join(ev) + f" | {regress_summary(reg)} |")
        prev = sps
    text = "\n".join(rows)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
