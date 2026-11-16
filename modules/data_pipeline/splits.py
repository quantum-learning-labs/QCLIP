import hashlib
import numpy as np
import pandas as pd
from modules.symbolic.root import apply_root, COPY_BOX, ROOT_MODES

# Train/val/test splits of a dataset's pool (data/<name>/processed/all.pkl), computed when the data is loaded from
# config['dataset']['split']:
#   {by: <column>, fractions: [train, val, test], seed: N}: whole groups (rows sharing the value of `by`, e.g. an image)
#       go to one split, chosen by a hash of (seed, group). Then the vocabulary is closed: a val or test group with a
#       word (word, wire types: one parameter tensor of the text model) that no train row has moves to train, so every
#       val/test word is trained. A group's split depends only on the pool, never on row order.
#   test_always: <set> (optional, e.g. swap95): the groups of that fixed evaluation set's rows always go to test and are
#       never moved by the vocabulary closure (their words must then be in train, or assign raises), so the set is
#       identical on every split seed and none of its images is trained on.
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

def row_vocab(pool):
    # the words of each row, per text.root mode: (mode, word, wire types) of every box of every parsed caption column
    # (a root word has other wires in word mode than in frobenius mode, and in head mode the head is a word too; the
    # frobenius copy box has no parameters, so it is not vocabulary). Closing on every mode at once makes the split the
    # same for every mode, so runs that differ in text.root are paired on the same splits and none meets an unseen word.
    cols = [c for c in pool.columns if c.endswith('_diagram')]
    if not cols:
        raise ValueError("closing the vocabulary needs the parsed pool (processed/all.pkl, *_diagram columns)")
    return [{(mode, box[0], tuple(box[2])) for diagram in row for mode in ROOT_MODES for box in _rooted(diagram, mode)
             if box[0] != COPY_BOX} for row in zip(*(pool[c] for c in cols))]

def _rooted(diagram, mode):
    # a mode with no rule for this root type (frobenius beyond n and s) raises when it is used; no words here
    try:
        return apply_root(diagram, mode)
    except ValueError:
        return []

FIXED_TEST_SETS = {}   # name -> function(pool) -> the set's rows (registered by datasets.py)

def close_vocabulary(pool, groups, labels, seed, pinned=frozenset()):
    # move val/test groups holding a word no train row has to train, in a seeded order (a moved group only adds train
    # words, so one pass leaves every val/test word in train). Pinned groups (test_always) stay in test.
    labels = labels.copy()
    vocab = row_vocab(pool)
    train_vocab = set().union(*(v for v, l in zip(vocab, labels) if l == 'train'))
    rows = pd.Series(range(len(pool))).groupby(np.asarray(groups)).apply(list)
    moved = 0
    for key in sorted(rows.index, key=lambda k: _unit(f"close{seed}", k)):
        idx = rows[key]
        if labels[idx[0]] == 'train' or key in pinned:
            continue
        words = set().union(*(vocab[i] for i in idx))
        if not words <= train_vocab:
            labels[idx] = 'train'
            train_vocab |= words
            moved += 1
    return labels, moved

def assign(pool, spec):
    # split label of every row, and a record of the split: spec, sizes, groups moved for the vocabulary and a fingerprint
    # (sha256 of the sorted (group, split) pairs; legacy: of the row order and labels)
    if spec == 'legacy':
        labels, moved = pool['legacy_split'].to_numpy(), 0
        keys = pd.Series(range(len(pool))).astype(str)
    else:
        groups = pool[spec['by']]
        labels = group_split(groups, spec['fractions'], spec['seed'])
        pinned = frozenset()
        if spec.get('test_always'):
            import modules.data_pipeline.datasets   # noqa: F401 (registers the fixed test sets)
            pinned = frozenset(FIXED_TEST_SETS[spec['test_always']](pool)[spec['by']])
            labels[groups.isin(pinned).to_numpy()] = 'test'
        labels, moved = close_vocabulary(pool, groups, labels, spec['seed'], pinned)
        if pinned:
            vocab = row_vocab(pool)
            train_vocab = set().union(*(v for v, l in zip(vocab, labels) if l == 'train'))
            missing = set().union(*(v for v, g in zip(vocab, groups) if g in pinned)) - train_vocab
            if missing:
                raise ValueError(f"test_always {spec['test_always']}: words not in train {sorted(missing)[:10]}")
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

def sample_rows(df, n, seed, key_cols=('corrected_sentence', 'pos_image_id', 'neg_image_id')):
    # n rows of df chosen by seed: the n smallest hashes of (seed, row key), so the choice depends only on the rows (not
    # their order) and is the same for every model on the same split. Identical rows are told apart by their occurrence.
    if len(df) <= n:
        return df.reset_index(drop=True)
    cols = [c for c in key_cols if c in df.columns]
    if not cols:
        raise ValueError(f"sample needs one of the columns {key_cols}")
    keys = df[cols].astype(str).agg('|'.join, axis=1)
    keys = keys + '#' + keys.groupby(keys).cumcount().astype(str)
    order = np.argsort([_unit(f"sample{seed}", k) for k in keys], kind='stable')
    return df.iloc[np.sort(order[:n])].reset_index(drop=True)
