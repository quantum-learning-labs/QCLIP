from collections import Counter

ROOT_MODES = ('head')

def _open_root(tn):
    word_arr, idx_arr, type_arr = zip(*tn)
    count_dict = Counter(sum(idx_arr, []))
    open_idx = [key for key, val in count_dict.items() if val == 1]
    root = next(i for i, word_idx in enumerate(idx_arr) if set(open_idx).issubset(word_idx))
    open_types = [type_arr[root][idx_arr[root].index(k)] for k in open_idx]
    return word_arr, idx_arr, type_arr, count_dict, open_idx, root, open_types

def root_head(tn):
    word_arr, idx_arr, type_arr, count_dict, open_idx, root, open_types = _open_root(tn)
    # the root's open wires get fresh indices, joined to the head; the head's output takes index 0 (the output wire)
    fresh = dict(zip(open_idx, range(max(count_dict) + 1, max(count_dict) + 1 + len(open_idx))))
    idx_arr[root][:] = [fresh.get(k, k) for k in idx_arr[root]]
    return list(zip(word_arr, idx_arr, type_arr)) + [('ROOT', list(fresh.values()) + [0], open_types + ['out'])]

def unify_codomain(tn, mode):
    tn = [(w, list(i), list(t)) for w, i, t in tn]
    if mode == 'head':
        return root_head(tn)
    raise ValueError(f"text.root {mode!r}: choose one of {ROOT_MODES}")