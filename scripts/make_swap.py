# Build a large subject/object-swap set from an SVO-probes split: for every row whose diagram has exactly two
# noun boxes (type ['n']), swap the two noun words. Grammar and vocabulary are unchanged, so no new symbols.
# Drops rows whose swapped sentence equals the original and verbs that are (roughly) symmetric, where the swap
# is still true of the image. Writes a NEW pickle in the swap.pkl layout; never overwrites.
# usage: python scripts/make_swap.py <in.pkl> <out.pkl>
import sys, os, copy, re
import pandas as pd
SYMMETRIC = {'meet', 'hug', 'kiss', 'marry', 'touch', 'fight', 'play', 'talk', 'dance', 'race', 'wrestle', 'greet',
             'shake', 'face', 'resemble', 'join', 'accompany', 'chat', 'argue', 'embrace', 'cuddle', 'date'}
src, out = sys.argv[1], sys.argv[2]
if os.path.exists(out): sys.exit(f'{out} exists; refusing to overwrite')
df = pd.read_pickle(src)
rows, stats = [], dict(total=len(df), not_two_nouns=0, same_noun=0, symmetric=0)
for _, r in df.iterrows():
    d = r['corrected_sentence_diagram']
    nouns = [i for i, b in enumerate(d) if list(b[2]) == ['n']]
    if len(nouns) != 2: stats['not_two_nouns'] += 1; continue
    a, b = nouns
    if d[a][0].lower() == d[b][0].lower(): stats['same_noun'] += 1; continue
    if str(r['verb']).lower() in SYMMETRIC: stats['symmetric'] += 1; continue
    sw = [list(x) for x in d]; sw[a][0], sw[b][0] = d[b][0], d[a][0]
    sw = [type(x)(y) if isinstance(x, tuple) else y for x, y in zip(d, sw)]   # keep the box container type
    sent = r['corrected_sentence']; wa, wb = d[a][0], d[b][0]
    swapped = re.sub(rf'\b{re.escape(wa)}\b', '\0', sent, count=1)
    swapped = re.sub(rf'\b{re.escape(wb)}\b', wa, swapped, count=1).replace('\0', wb)
    row = r.to_dict(); row['swapped_sentence'] = swapped; row['swapped_sentence_diagram'] = sw
    rows.append(row)
res = pd.DataFrame(rows)
res = res[[c for c in res.columns if not c.endswith('_diagram')] + ['corrected_sentence_diagram', 'swapped_sentence_diagram']]
res.to_pickle(out)
print(stats, '-> kept', len(res)); print(res[['corrected_sentence', 'swapped_sentence']].head(8).to_string())
