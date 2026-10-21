# Build a large subject/object-swap set from an SVO-probes split: for every row whose diagram has exactly two
# noun boxes (type ['n']), swap the two noun words. Grammar and vocabulary are unchanged, so no new symbols.
# Drops rows whose swapped sentence equals the original and verbs that are (roughly) symmetric, where the swap
# is still true of the image. Writes a NEW pickle in the swap.pkl layout; never overwrites.
# usage: python scripts/make_swap.py <in.pkl> <out.pkl> [--keep_all]  (--keep_all: every row kept, swap_valid marks real swaps;
#        used for training with swapped-caption negatives so the training set itself is unchanged)
import sys, os, copy, re
import pandas as pd
SYMMETRIC = {'meet', 'hug', 'kiss', 'marry', 'touch', 'fight', 'play', 'talk', 'dance', 'race', 'wrestle', 'greet',
             'shake', 'face', 'resemble', 'join', 'accompany', 'chat', 'argue', 'embrace', 'cuddle', 'date'}
keep_all = '--keep_all' in sys.argv; argv = [a for a in sys.argv if a != '--keep_all']
src, out = argv[1], argv[2]
if os.path.exists(out): sys.exit(f'{out} exists; refusing to overwrite')
df = pd.read_pickle(src)
rows, stats = [], dict(total=len(df), not_two_nouns=0, same_noun=0, symmetric=0)
for _, r in df.iterrows():
    d = r['corrected_sentence_diagram']
    nouns = [i for i, b in enumerate(d) if list(b[2]) == ['n']]
    reason = 'not_two_nouns' if len(nouns) != 2 else 'same_noun' if d[nouns[0]][0].lower() == d[nouns[1]][0].lower() \
        else 'symmetric' if str(r['verb']).lower() in SYMMETRIC else None
    if reason:
        stats[reason] += 1
        if keep_all:
            row = r.to_dict(); row['swapped_sentence'] = r['corrected_sentence']; row['swapped_sentence_diagram'] = d; row['swap_valid'] = False
            rows.append(row)
        continue
    a, b = nouns
    sw = [list(x) for x in d]; sw[a][0], sw[b][0] = d[b][0], d[a][0]
    sw = [type(x)(y) if isinstance(x, tuple) else y for x, y in zip(d, sw)]   # keep the box container type
    sent = r['corrected_sentence']; wa, wb = d[a][0], d[b][0]
    swapped = re.sub(rf'\b{re.escape(wa)}\b', '\0', sent, count=1)
    swapped = re.sub(rf'\b{re.escape(wb)}\b', wa, swapped, count=1).replace('\0', wb)
    row = r.to_dict(); row['swapped_sentence'] = swapped; row['swapped_sentence_diagram'] = sw; row['swap_valid'] = True
    rows.append(row)
res = pd.DataFrame(rows)
res = res[[c for c in res.columns if not c.endswith('_diagram')] + ['corrected_sentence_diagram', 'swapped_sentence_diagram']]
res.to_pickle(out)
print(stats, '-> kept', len(res)); print(res[['corrected_sentence', 'swapped_sentence']].head(8).to_string())
