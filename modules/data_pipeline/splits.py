import hashlib
import numpy as np
import pandas as pd

# Train/val/test splits of a dataset's pool (data/<name>/processed/all.pkl), computed when the data is loaded from
# config['dataset']['split']:
#   {by: <column>, fractions: [train, val, test], seed: N}: whole groups (rows sharing the value of `by`, e.g. an image)
#       go to one split, chosen by a hash of (seed, group). A group's split depends only on its own key, so it does not
#       change with row order, with rows the parser dropped, or with rows added to the pool.
#   legacy: the pool's legacy_split column (SVO-Probes: the split of all results before 2026-11-13).
SPLITS = ('train', 'val', 'test')

def group_split(groups, fractions, seed):
    bounds = np.cumsum(fractions) / np.sum(fractions)
    def label(key):
        u = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15
        return SPLITS[min(int(np.searchsorted(bounds, u, side='right')), len(SPLITS) - 1)]
    keys = pd.Series(np.asarray(groups))
    return keys.map({k: label(k) for k in keys.unique()}).to_numpy()

def select(pool, spec, split, subset=None):
    # rows of the pool in `split` (train, val, test, or all = the whole pool); `subset` keeps rows whose subset column
    # matches (ARO: att or rel)
    if split != 'all':
        labels = pool['legacy_split'].to_numpy() if spec == 'legacy' else group_split(pool[spec['by']], spec['fractions'], spec['seed'])
        pool = pool[labels == split]
    if subset is not None:
        pool = pool[pool['subset'] == subset]
    return pool.reset_index(drop=True)
