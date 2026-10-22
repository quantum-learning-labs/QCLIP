"""Step tables for the TN retrieval work: mean ± std over seeds, chance, distance to chance, efficiency.

  python tools/tn/summarise.py <out_dir> <label>=<run_glob> [<label>=<run_glob> ...]
  e.g. python tools/tn/summarise.py results/tn/step1 clip='tn_s1_clip_s*' ttn='tn_s1_ttn_s*'

Reads checkpoints/mscoco/tn/<run>/{benchmark_last.json, run_meta.json}; writes <out_dir>/summary.{md,json} (new files only).
"""
import sys, json, glob, math
from pathlib import Path
from statistics import mean as _mean, stdev

CKPT = Path('checkpoints/mscoco/tn')

def coco_chance(k, n_items, n_pos):
    # P(at least one of n_pos matches in the top k of n_items) under a random ranking
    return 1.0 - math.prod((n_items - n_pos - i) / (n_items - i) for i in range(k))

# (block, metric, label, chance)  chance None -> computed (coco); see the step-1 report for definitions
ROWS = [('coco5k', f'{d}R{k}', f'MSCOCO 5k {d} R@{k}', None) for d in ('i2t', 't2i') for k in (1, 5, 10)] + [
    ('aro_rel', 'hard_neg_acc', 'ARO VG-Relation', 0.5), ('aro_att', 'hard_neg_acc', 'ARO VG-Attribution', 0.5),
    ('svo_std', 'hard_neg_acc', 'SVO-probes std (image choice)', 0.5),
    ('svo_swap_full', 'hard_neg_acc', 'SVO swap_full (text choice)', 0.5),
    ('svo_swap_animate', 'hard_neg_acc', 'SVO swap_animate (text choice)', 0.5)] + [
    (f'sugarcrepe_{s}', 'acc', f'SugarCREPE++ {s} (both pos > neg)', 1 / 3)
    for s in ('swap_object', 'swap_atribute', 'replace_object', 'replace_attribute', 'replace_relation')] + [
    ('winoground', 'txt_score', 'Winoground text', 0.25), ('winoground', 'img_score', 'Winoground image', 0.25),
    ('winoground', 'grp_score', 'Winoground group', 1 / 6)]

def load(glob_pat):
    runs = []
    for d in sorted(glob.glob(str(CKPT / glob_pat))):
        b, m = Path(d) / 'benchmark_last.json', Path(d) / 'run_meta.json'
        if b.exists() and m.exists():
            runs.append({'name': Path(d).name, 'res': json.load(open(b))['results'], 'meta': json.load(open(m))})
    return runs

def summarise(runs):
    out = {}
    for blk, met, label, chance in ROWS:
        vals = [r['res'].get(f'{blk}_{met}') for r in runs]
        vals = [v for v in vals if v is not None]
        if not vals: continue
        n = runs[0]['res'].get(f'{blk}_n')
        if chance is None:   # coco val2017: 5,000 images, 25,014 captions (5 per image; i2t has 5 positives)
            k = int(met.split('R')[-1])
            chance = coco_chance(k, 25014, 5) if met.startswith('i2t') else k / n
        mean, std = float(_mean(vals)), (stdev(vals) if len(vals) > 1 else 0.0)
        se = math.sqrt(max(mean * (1 - mean), 1e-12) / n) if n else float('nan')
        out[label] = {'mean': mean, 'std': std, 'seeds': len(vals), 'n': n, 'chance': chance, 'binom_se': se,
                      'above_chance': mean - chance, 'z_vs_chance': (mean - chance) / se if se else float('nan'),
                      'oov_sentence_frac': runs[0]['res'].get(f'{blk}_oov_sentence_frac'), 'values': vals}
    eff = {}
    m = [r['meta'] for r in runs]
    eff['params_text'] = m[0]['params_text']; eff['params_text_by_order'] = m[0]['params_text_by_order']
    eff['params_image_trainable'] = m[0]['params_image_trainable']
    eff['train_samples_per_s'] = float(_mean([e['train_samples_per_s'] for x in m for e in x['epochs']]))
    eff['epoch_train_min'] = float(_mean([e['train_s'] for x in m for e in x['epochs']])) / 60
    eff['val_eval_s'] = float(_mean([e['val_eval_s'] for x in m for e in x['epochs']]))
    eff['peak_train_mem_gb'] = max(e['peak_mem_gb'] or 0 for x in m for e in x['epochs'])
    eff['bench_rows_per_s'] = {k[:-len('_rows_per_s')]: float(_mean([r['res'][k] for r in runs]))
                               for k in runs[0]['res'] if k.endswith('_rows_per_s')}
    eff['gpus'] = sorted({x['gpu'] for x in m}); eff['runs'] = [r['name'] for r in runs]
    return out, eff

if __name__ == '__main__':
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    groups = dict(a.split('=', 1) for a in sys.argv[2:])
    data = {lab: summarise(load(g)) for lab, g in groups.items()}
    labs = list(groups)
    lines = ['| benchmark | chance | n | ' + ' | '.join(f'{l} mean ± std (seeds)' for l in labs) + ' | ' +
             ' | '.join(f'{l} above chance (z)' for l in labs) + ' | OOV sent. |',
             '|---|---|---|' + '---|' * (2 * len(labs)) + '---|']
    for _, _, label, _ in ROWS:
        if label not in data[labs[0]][0]: continue
        r0 = data[labs[0]][0][label]
        cells = [f"{data[l][0][label]['mean']:.4f} ± {data[l][0][label]['std']:.4f} ({data[l][0][label]['seeds']})" if label in data[l][0] else '-' for l in labs]
        gaps = [f"{data[l][0][label]['above_chance']:+.4f} ({data[l][0][label]['z_vs_chance']:.1f})" if label in data[l][0] else '-' for l in labs]
        oov = f"{r0['oov_sentence_frac']:.3f}" if r0['oov_sentence_frac'] is not None else '-'
        lines.append(f"| {label} | {r0['chance']:.4f} | {r0['n']} | " + ' | '.join(cells) + ' | ' + ' | '.join(gaps) + f" | {oov} |")
    lines += ['', '| efficiency | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    for k in ('params_text', 'params_image_trainable', 'train_samples_per_s', 'epoch_train_min', 'val_eval_s', 'peak_train_mem_gb', 'gpus'):
        lines.append(f'| {k} | ' + ' | '.join(f"{data[l][1][k]:,.2f}" if isinstance(data[l][1][k], float) else
                                               (f"{data[l][1][k]:,}" if isinstance(data[l][1][k], int) else str(data[l][1][k])) for l in labs) + ' |')
    md = '\n'.join(lines) + '\n'
    with open(out_dir / 'summary.md', 'x') as f: f.write(md)
    with open(out_dir / 'summary.json', 'x') as f: json.dump(data, f, indent=1)
    print(md)
