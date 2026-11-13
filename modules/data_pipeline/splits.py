import hashlib
import numpy as np
import pandas as pd
from modules.symbolic.root import apply_root

# Train/val/test splits of a dataset's pool (data/<name>/processed/all.pkl), computed when the data is loaded from
# config['dataset']['split']:
#   {by: <column>, fractions: [train, val, test], seed: N}: whole groups (rows sharing the value of `by`, e.g. an image)
#       go to one split, chosen by a hash of (seed, group). Then the vocabulary is closed: a val or test group with a
#       word (word, wire types: one parameter tensor of the text model) that no train row has moves to train, so every
#       val/test word is trained. A group's split depends only on the pool, never on row order.
#   legacy: the pool's legacy_split column (SVO-Probes: the split of all results before 2026-11-13; not closed).
# split_record() fingerprints the assignment: compare models only over the same set of split ids.
SPLITS = ('train', 'val', 'test')

def _unit(seed, key):
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15

def group_split(groups, fractions, seed):
    bounds = np.cumsum(fractions) / np.sum(fractions)
    keys = pd.Series(np.asarray(groups))
    label = {k: SPLITS[min(int(np.searchsorted(bounds, _unit(seed, k), side='right')), len(SPLITS) - 1)] for k in keys.unique()}
    return keys.map(label).to_numpy()

def row_vocab(pool, root='word'):
    # the words of each row: (word, wire types) of every box of every parsed caption column, after text.root (a verb
    # that is the root has other wires than inside a sentence, and in head mode the head is a word too)
    cols = [c for c in pool.columns if c.endswith('_diagram')]
    if not cols:
        raise ValueError("closing the vocabulary needs the parsed pool (processed/all.pkl, *_diagram columns)")
    return [{(box[0], tuple(box[2])) for diagram in row for box in apply_root(diagram, root)} for row in zip(*(pool[c] for c in cols))]

def close_vocabulary(pool, groups, labels, seed, root='word'):
    # move val/test groups holding a word no train row has to train, in a seeded order (a moved group only adds train
    # words, so one pass leaves every val/test word in train)
    labels = labels.copy()
    vocab = row_vocab(pool, root)
    train_vocab = set().union(*(v for v, l in zip(vocab, labels) if l == 'train'))
    rows = pd.Series(range(len(pool))).groupby(np.asarray(groups)).apply(list)
    moved = 0
    for key in sorted(rows.index, key=lambda k: _unit(f"close{seed}", k)):
        idx = rows[key]
        if labels[idx[0]] == 'train':
            continue
        words = set().union(*(vocab[i] for i in idx))
        if not words <= train_vocab:
            labels[idx] = 'train'
            train_vocab |= words
            moved += 1
    return labels, moved

def assign(pool, spec, root='word'):
    # split label of every row, and a record of the split: spec, sizes, groups moved for the vocabulary and a fingerprint
    # (sha256 of the sorted (group, split) pairs; legacy: of the row order and labels)
    if spec == 'legacy':
        labels, moved = pool['legacy_split'].to_numpy(), 0
        keys = pd.Series(range(len(pool))).astype(str)
    else:
        groups = pool[spec['by']]
        labels = group_split(groups, spec['fractions'], spec['seed'])
        labels, moved = close_vocabulary(pool, groups, labels, spec['seed'], root)
        keys = groups.astype(str)
    pairs = sorted(set(zip(keys, labels)))
    fingerprint = hashlib.sha256('\n'.join(f"{k}\t{l}" for k, l in pairs).encode()).hexdigest()[:16]
    record = {'spec': spec, 'id': fingerprint, 'rows': {s: int((labels == s).sum()) for s in SPLITS}, 'groups_moved_for_vocab': moved}
    return labels, record

def select(pool, labels, split, subset=None):
    # rows of the pool in `split` (train, val, test, or all = the whole pool); `subset` keeps rows whose subset column
    # matches (ARO: att or rel)
    if split != 'all':
        pool = pool[labels == split]
    if subset is not None:
        pool = pool[pool['subset'] == subset]
    return pool.reset_index(drop=True)
