from collections import Counter

# text.root: how a sentence's open wires become the output wire 'out' (index 0), applied when the data is compiled
#   word: the root word's open wires are replaced by 'out', so a word has another signature (and other parameters) as root
#   head: the root word keeps its own type; a box ROOT_HEAD maps the open wires to 'out'. Its symbol depends only on the
#         open wires' types, so one head is shared by every sentence with that root type (s, n, ...)
ROOT_HEAD = 'ROOT'

def root_head(tn):
    word_arr, idx_arr, type_arr = zip(*tn)
    if 0 in sum(idx_arr, []):
        raise ValueError("diagram already has an output wire (parsed before 2026-11-13 with the root merged): re-run "
                         "scripts/retrieve.py to use text.root: head")
    count_dict = Counter(sum(idx_arr, []))
    open_idx = [key for key, val in count_dict.items() if val == 1]
    root = next(i for i, word_idx in enumerate(idx_arr) if set(open_idx).issubset(word_idx))
    open_types = [type_arr[root][idx_arr[root].index(k)] for k in open_idx]
    return list(tn) + [(ROOT_HEAD, open_idx + [0], open_types + ['out'])]

def apply_root(tn, mode):
    # copies: unify_codomain edits the diagram's lists in place (it is idempotent on diagrams already unified)
    tn = [(w, list(i), list(t)) for w, i, t in tn]
    if mode == 'word':
        return unify_codomain(tn)
    if mode == 'head':
        return root_head(tn)
    raise ValueError(f"text.root {mode!r}: choose word or head")

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
