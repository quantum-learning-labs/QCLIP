# Summarises the v1 results (results/v1/<tag>/k<k>.json and zero-shot <model>_k<k>.json) into results/v1/summary.json
# and a markdown table (results/v1/summary.md): test accuracy with a tie counted 1/2, mean +- sample SD over split seeds 0-4.
#   python scripts/summarise_v1.py results/v1
import json, os, statistics, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'results/v1'
SETS = {'svo-probes': ['std', 'swap_full', 'swap95'], 'aro': ['att', 'rel']}
# (row label, dataset, results dir, file prefix, score head, text params)
ROWS = [
    ('Zero-shot CLIP ViT-B/32', 'svo-probes', 'zeroshot_svo', 'clip_', 'cos', '-'),
    ('Zero-shot OpenCLIP ViT-B/32 (LAION-2B)', 'svo-probes', 'zeroshot_svo', 'openclip_', 'cos', '-'),
    ('Zero-shot BLIP (ITM)', 'svo-probes', 'zeroshot_svo', 'blip_', 'itm', '-'),
    ('Zero-shot BLIP (ITC)', 'svo-probes', 'zeroshot_svo', 'blip_', 'itc', '-'),
    ('QBoW (5 layers)', 'svo-probes', 'svo_qbow', '', None, '24,435'),
    ('MicroCLIP-16 (matched)', 'svo-probes', 'svo_microclip_cpu', '', None, '19,504'),
    ('CCG tensor network (bond 2)', 'svo-probes', 'svo_tn', '', None, '364,360'),
    ('CCG-VQC, post-selection (3 qubits/type, 5 layers)', 'svo-probes', 'svo_ps_n3L5', '', None, '19,485'),
    ('Zero-shot CLIP ViT-B/32', 'aro', 'zeroshot_aro', 'clip_', 'cos', '-'),
    ('Zero-shot OpenCLIP ViT-B/32 (LAION-2B)', 'aro', 'zeroshot_aro', 'openclip_', 'cos', '-'),
    ('Zero-shot BLIP (ITM)', 'aro', 'zeroshot_aro', 'blip_', 'itm', '-'),
    ('Zero-shot BLIP (ITC)', 'aro', 'zeroshot_aro', 'blip_', 'itc', '-'),
    ('QBoW (3 layers)', 'aro', 'aro_qbow', '', None, '141,912'),
    ('MicroCLIP-24 (matched)', 'aro', 'aro_microclip', '', None, '59,134'),
    ('CCG tensor network (bond 2)', 'aro', 'aro_tn', '', None, '288,720'),
    ('CCG-VQC, post-selection (3 qubits/type, 3 layers)', 'aro', 'aro_ps_n3L3', '', None, '59,130'),
]

def find(d, key):
    # the metric may sit at the top level or one level down (e.g. under 'metrics')
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return None

summary, md = [], []
for ds in SETS:
    md += [f'### {ds}', '', '| model | text params | ' + ' | '.join(SETS[ds]) + ' |', '|---|---|' + '---|' * len(SETS[ds])]
    for label, rds, tag, prefix, head, params in ROWS:
        if rds != ds:
            continue
        cells, entry = [], {'model': label, 'dataset': ds, 'dir': tag, 'text_params': params, 'scores': {}}
        for ts in SETS[ds]:
            key = f'{ts}_{head}_hard_neg_acc_tie_half' if head else f'{ts}_hard_neg_acc_tie_half'
            vals = []
            for k in range(5):
                p = os.path.join(ROOT, tag, f'{prefix}k{k}.json')
                if os.path.exists(p):
                    v = find(json.load(open(p)), key)
                    if v is not None:
                        vals.append(float(v))
            if len(vals) == 5:
                m, s = statistics.mean(vals), statistics.stdev(vals)
                cells.append(f'{m:.3f} ± {s:.3f}')
                entry['scores'][ts] = {'mean': m, 'sd': s, 'per_split': vals}
            else:
                cells.append(f'n={len(vals)}')
        summary.append(entry)
        md.append(f'| {label} | {params} | ' + ' | '.join(cells) + ' |')
    md.append('')
json.dump(summary, open(os.path.join(ROOT, 'summary.json'), 'w'), indent=1)
open(os.path.join(ROOT, 'summary.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
