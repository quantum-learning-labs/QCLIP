"""E2a + E2b data (step 5, user-approved 2026-11-11). No model: reads the parsed diagrams (word, wires, atomic leg types).
E2b inventory: every word occurrence in the COCO training captions has a type signature (its leg types joined by '@'); counts of
signatures by order, the token share of rare signatures, and per function word the spread of its occurrences over signatures
(how fragmented one word's meaning is across grammatical roles).
E2a confound check: joins audit_<run>.json per-item correctness with item features (length of the reference caption, training
count of the rarest signature in the item) and reports accuracy by length bin x rare/common, plus the length-stratified
(Mantel-Haenszel weighted) rare-minus-common difference.
  python tools/tn/s5_e2_types.py <out.json> <audit_run.json>...   (run from qclip/, CPU only)"""
import sys, json, collections, re
from pathlib import Path
import numpy as np, pandas as pd

out_path = Path(sys.argv[1]); audits = sys.argv[2:]
if out_path.exists(): raise FileExistsError(out_path)
sig = lambda types: '@'.join(types)

tr = pd.read_pickle('data/mscoco/processed/tn/train.pkl')
dcol = [c for c in tr.columns if c.endswith('_diagram')][0]
tok, word_sig, n_caps = collections.Counter(), collections.defaultdict(collections.Counter), 0
for caps in tr[dcol]:
    for d in (caps or []):
        n_caps += 1
        for w, _, t in d:
            s = sig(t); tok[s] += 1; word_sig[w.lower()][s] += 1
total = sum(tok.values())
order = collections.Counter()
for s, c in tok.items(): order[len(s.split('@'))] += c
res = {'train_captions': n_caps, 'train_word_tokens': total, 'n_signatures': len(tok),
       'tokens_by_order': {k: v / total for k, v in sorted(order.items())},
       'signatures_by_order': dict(sorted(collections.Counter(len(s.split('@')) for s in tok).items()))}
for th in (100, 1000, 10000):
    rare = [s for s, c in tok.items() if c <= th]
    res[f'signatures_le_{th}'] = {'n': len(rare), 'token_share': sum(tok[s] for s in rare) / total}
res['top_signatures'] = [{'sig': s, 'tokens': c, 'share': c / total,
                          'top_words': [w for w, _ in collections.Counter({w: word_sig[w][s] for w in word_sig if s in word_sig[w]}).most_common(8)]}
                         for s, c in tok.most_common(60)]
FW = ['on', 'in', 'with', 'of', 'at', 'by', 'near', 'next', 'behind', 'under', 'above', 'over', 'for', 'from', 'to', 'into',
      'through', 'and', 'while', 'that', 'is', 'are', 'has', 'holding', 'sitting', 'standing', 'riding', 'hanging', 'white', 'red']
res['word_fragmentation'] = {w: {'tokens': sum(word_sig[w].values()), 'n_signatures': len(word_sig[w]),
                                 'top': [(s, c / sum(word_sig[w].values())) for s, c in word_sig[w].most_common(6)],
                                 'share_in_sigs_le_1000': sum(c for s, c in word_sig[w].items() if tok[s] <= 1000) / max(1, sum(word_sig[w].values()))}
                             for w in FW if w in word_sig}

# E2a: features per benchmark item (same row order as the audit: compile_text keeps the pickle order)
SPLIT_PKL = {'sugarcrepe_' + k: f'data/sugarcrepe/processed/{k}.pkl' for k in
             ('swap_object', 'swap_atribute', 'replace_object', 'replace_attribute', 'replace_relation')}
SPLIT_PKL.update({'winoground': 'data/winoground/processed/test.pkl', 'aro_rel': 'data/aro/processed/test_rel.pkl',
                  'aro_att': 'data/aro/processed/test_att.pkl'})
def lbin(n): return '<=8' if n <= 8 else '9-11' if n <= 11 else '12-14' if n <= 14 else '>=15'
feats = {}
for split, p in SPLIT_PKL.items():
    df = pd.read_pickle(p); dc = [c for c in df.columns if c.endswith('_diagram')]
    L, R, O = [], [], []
    for _, r in df.iterrows():
        ds = [r[c] for c in dc if isinstance(r[c], list)]
        L.append(len(ds[0])); R.append(min(tok[sig(t)] for d in ds for _, _, t in d)); O.append(max(len(t) for d in ds for _, _, t in d))
    feats[split] = (np.array(L), np.array(R), np.array(O))
res['e2a'] = {}
for a in audits:
    A = json.load(open(a)); run = Path(a).stem.replace('audit_', '')
    res['e2a'][run] = {}
    for split, (L, R, O) in feats.items():
        if split not in A: continue
        key = 'acc' if 'acc' in A[split]['per_item'] else 'txt'
        c = np.array(A[split]['per_item'][key], dtype=float)
        if len(c) != len(L): res['e2a'][run][split] = f'length mismatch {len(c)} vs {len(L)}'; continue
        rare = R <= 1000
        cells, num, den = {}, 0.0, 0.0
        for b in ('<=8', '9-11', '12-14', '>=15'):
            m = np.array([lbin(x) == b for x in L])
            nr, nc = (m & rare).sum(), (m & ~rare).sum()
            cells[b] = {'n_rare': int(nr), 'acc_rare': float(c[m & rare].mean()) if nr else None,
                        'n_common': int(nc), 'acc_common': float(c[m & ~rare].mean()) if nc else None}
            if nr and nc:
                w = nr * nc / (nr + nc); num += w * (c[m & rare].mean() - c[m & ~rare].mean()); den += w
        res['e2a'][run][split] = {'n': len(c), 'rare_frac': float(rare.mean()), 'acc_rare': float(c[rare].mean()) if rare.any() else None,
                                  'acc_common': float(c[~rare].mean()), 'raw_diff': float(c[rare].mean() - c[~rare].mean()) if rare.any() else None,
                                  'length_stratified_diff': num / den if den else None, 'mean_len_rare': float(L[rare].mean()) if rare.any() else None,
                                  'mean_len_common': float(L[~rare].mean()), 'by_length': cells}
json.dump(res, open(out_path, 'x'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('top_signatures', 'word_fragmentation', 'e2a')}, indent=1))
for s in res['top_signatures'][:25]: print(f"  {s['sig']:28s} {s['share']:.4f} {s['top_words'][:6]}")
for w, v in res['word_fragmentation'].items(): print(f"  {w:9s} tok {v['tokens']:7d} sigs {v['n_signatures']:4d} rare-share {v['share_in_sigs_le_1000']:.3f} top {[(s, round(x, 2)) for s, x in v['top'][:3]]}")
for run, d in res['e2a'].items():
    for split, v in d.items():
        if isinstance(v, dict) and v['acc_rare'] is not None:
            print(f"  {run:18s} {split:28s} rare {v['rare_frac']:.2f} raw {v['raw_diff']:+.3f} len-strat {v['length_stratified_diff'] if v['length_stratified_diff'] is None else round(v['length_stratified_diff'], 3)} len r/c {v['mean_len_rare']:.1f}/{v['mean_len_common']:.1f}")
