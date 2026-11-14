from collections import Counter

# text.root: how a sentence's open wires become the output wire (index 0), applied when the data is compiled
# on top of the parsed networks, which scripts/retrieve.py stores as parsed (unchanged by any method)
#   word:      the root word's open wires are replaced by 'out', so a word has another signature (and other parameters) as root
#   head:      the root word keeps its own type; a box ROOT_HEAD maps the open wires to 'out'. Its symbol depends only on the
#              open wires' types, so one head is shared by every sentence with that root type (s, n, ...)
#   frobenius: the output is the sentence space S = N^(x)m (needs s = embedding_qubits and s a multiple of n). No word is
#              retyped: an s root is read out as it is; an n root (a noun-phrase caption) is mapped into S by the Frobenius
#              copy N -> N^(x)m (COPY_BOX, parameter-free: |i> -> |i...i>, the diagonal of S). Other root types raise.
#              Unified sentence space via Frobenius copying: Kartsaklis, Sadrzadeh & Pulman, COLING 2012.
ROOT_MODES = ('word', 'head', 'frobenius')
ROOT_HEAD = 'ROOT'
COPY_BOX = 'COPY'   # structural box: compiled to fixed gates, no parameters

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
    return list(zip(word_arr, idx_arr, type_arr)) + [(ROOT_HEAD, list(fresh.values()) + [0], open_types + ['out'])]

def root_frobenius(tn):
    word_arr, idx_arr, type_arr, count_dict, open_idx, root, open_types = _open_root(tn)
    if open_types == ['s']:
        assert open_idx == [0], open_idx
        return list(zip(word_arr, idx_arr, type_arr))
    if open_types == ['n']:
        fresh = max(count_dict) + 1
        idx_arr[root][:] = [fresh if k == open_idx[0] else k for k in idx_arr[root]]
        return list(zip(word_arr, idx_arr, type_arr)) + [(COPY_BOX, [fresh, 0], ['n', 's'])]
    raise ValueError(f"text.root frobenius: no map into S for a root with open wires {open_types}")

def apply_root(tn, mode):
    # copies: unify_codomain edits the diagram's lists in place
    tn = [(w, list(i), list(t)) for w, i, t in tn]
    if mode == 'word':
        return unify_codomain(tn)
    if mode == 'head':
        return root_head(tn)
    if mode == 'frobenius':
        return root_frobenius(tn)
    raise ValueError(f"text.root {mode!r}: choose one of {ROOT_MODES}")

def unify_codomain(tn):
    word_arr, idx_arr, type_arr = zip(*tn)

    # Get output indices of tensor network
    flat_idx_arr = sum(idx_arr, [])
    count_dict = Counter(flat_idx_arr)
    output_idx = [key for key, val in count_dict.items() if val == 1]

    # Find indices of word containing output indices (root word)
    for i, word_idx in enumerate(idx_arr):
        if set(output_idx).issubset(word_idx):
            break 
    
    # Compute location of output indices inside root word indices
    start_idx = word_idx.index(output_idx[0])
    end_idx = start_idx + len(output_idx) 
    
    # Replace root word output indices and type with single unique output index and type
    idx_arr[i][start_idx:end_idx] = [0]
    type_arr[i][start_idx:end_idx] = ['out']
    return list(zip(word_arr, idx_arr, type_arr))
