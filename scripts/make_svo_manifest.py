# One-off provenance script: builds modules/data_pipeline/svo_probes_assets/{split_manifest.csv.gz,swap.pkl.gz} from the
# original raw pickles (data/svo-probes/raw/{train,val,test,swap}.pkl) and DeepMind's svo_probes.csv, then checks that
# svo_probes.build_splits reproduces the pickles exactly. Never overwrites: refuses if the outputs exist.
# usage: python scripts/make_svo_manifest.py <raw_dir> <svo_probes.csv>
import sys, shutil
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.data_pipeline import svo_probes as sp

raw, csvf = Path(sys.argv[1]), sys.argv[2]
out = sp.ASSETS; out.mkdir(exist_ok=True)
assert not (out / "split_manifest.csv.gz").exists(), "manifest exists; refusing to overwrite"
csv = pd.read_csv(csvf)
key = ['sentence', 'pos_triplet', 'neg_triplet', 'pos_url', 'neg_url', 'pos_image_id', 'neg_image_id', 'subj_neg', 'verb_neg', 'obj_neg']
c = csv.assign(dm_row=range(len(csv))); c['occ'] = c.groupby(key).cumcount()
rows = []
for s in sp.SPLITS:
    d = pd.read_pickle(raw / f"{s}.pkl"); d = d.assign(split=s, index=d.index); d['occ'] = d.groupby(key).cumcount()
    m = d.merge(c[key + ['occ', 'dm_row']], on=key + ['occ'], how='left'); assert len(m) == len(d) and m.dm_row.notna().all(), s
    rows.append(m[['split', 'index', 'dm_row', 'corrected_sentence']])
man = pd.concat(rows); man['dm_row'] = man.dm_row.astype(int)
man.to_csv(out / "split_manifest.csv.gz", index=False)
pd.read_pickle(raw / "swap.pkl").to_pickle(out / "swap.pkl.gz")
new = sp.build_splits(csv)
for s in list(sp.SPLITS) + ['swap']:
    pd.testing.assert_frame_equal(new[s], pd.read_pickle(raw / f"{s}.pkl"), check_exact=True); print(s, len(new[s]), "identical to the original pickle")
