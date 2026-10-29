from __future__ import annotations
import pandas as pd
import opt_einsum as oe
from itertools import count
import os, multiprocessing
from functools import partial
from tqdm import tqdm
    
class TNCompiler: 
    def __init__(self, obmap: dict[str, int], decomp_fn, decouple: bool = False, orient: bool = False):
        # decouple: one shared vector per word + one shared tensor per complex type (compile_decoupled_tn)
        # orient: each symbol also gets 'out' = position of its leg towards the sentence output (for text.init: norm)
        self.obmap = obmap
        self.word_dim = obmap.get('sem', 16)
        self.decomposition = decomp_fn
        self.decouple = decouple
        self.orient = orient
        self.id = type(self).__name__ + '_' + str(obmap['n']) + '_' + str(obmap['s']) + '_' + str(obmap['p']) + '_' + str(obmap['out'])
        if decouple:
            self.id += f'_dec{self.word_dim}'

    def compile_tn(self, tn):
        if self.decouple:
            return self.compile_decoupled_tn(tn)
        all_indices = [idx for _, idx_arr, _ in tn for idx in idx_arr]
        max_idx = max(all_indices) if all_indices else 0
        index_counter = count(start=max_idx + 1)

        processed_tensors = []

        for word, idx_arr, type_arr in tn:
            if len(idx_arr) != len(type_arr):
                raise ValueError(f"Mismatch between indices and types for word '{word}' in sentence '{' '.join([w for w, _, _ in tn])}': "
                                 f"indices={idx_arr}, types={type_arr}")
            dims = [self.obmap.get(t, 1) for t in type_arr]
            symbol = word + '_' + '@'.join(type_arr)
            if self.decomposition is None:
                processed_tensors.append((symbol, idx_arr, tuple(dims)))
            else:
                cores = self.decomposition.decompose(symbol, idx_arr, dims, index_counter)
                processed_tensors.extend(cores)

        input_subs = []
        symbols = []

        for symbol, idx_arr, shape in processed_tensors:
            subs = "".join(oe.parser.get_symbol(i) for i in idx_arr)
            input_subs.append(subs)
            symbols.append({'name': symbol, 'shape': list(shape)})
        if self.orient:
            for sym, out in zip(symbols, out_legs([idx for _, idx, _ in processed_tensors])):
                sym['out'] = out

        output_subs = oe.parser.get_symbol(0)
        einsum_str = f"{','.join(input_subs)}->{output_subs}"
        return einsum_str, symbols

    def compile_decoupled_tn(self, tn):
        # word -> vector 'w:<word>' (word_dim, never decomposed), shared by all its occurrences whatever the type;
        # complex type -> tensor 't:<types>' shared by all words of that type, with one extra leg of dim word_dim
        # in the middle of its legs (shortest MPS path from the word to every grammatical leg). Only the type tensor
        # goes through the decomposition.
        all_indices = [idx for _, idx_arr, _ in tn for idx in idx_arr]
        max_idx = max(all_indices) if all_indices else 0
        index_counter = count(start=max_idx + 1)

        processed_tensors = []

        for word, idx_arr, type_arr in tn:
            if len(idx_arr) != len(type_arr):
                raise ValueError(f"Mismatch between indices and types for word '{word}' in sentence '{' '.join([w for w, _, _ in tn])}': "
                                 f"indices={idx_arr}, types={type_arr}")
            wire = next(index_counter)
            processed_tensors.append(('w:' + word, [wire], (self.word_dim,)))

            mid = len(idx_arr) // 2
            type_idx = list(idx_arr[:mid]) + [wire] + list(idx_arr[mid:])
            dims = [self.obmap.get(t, 1) for t in type_arr]
            type_dims = dims[:mid] + [self.word_dim] + dims[mid:]
            symbol = 't:' + '@'.join(type_arr)
            if self.decomposition is None:
                processed_tensors.append((symbol, type_idx, tuple(type_dims)))
            else:
                cores = self.decomposition.decompose(symbol, type_idx, type_dims, index_counter)
                processed_tensors.extend(cores)

        input_subs = []
        symbols = []

        for symbol, idx_arr, shape in processed_tensors:
            subs = "".join(oe.parser.get_symbol(i) for i in idx_arr)
            input_subs.append(subs)
            symbols.append({'name': symbol, 'shape': list(shape)})
        if self.orient:
            for sym, out in zip(symbols, out_legs([idx for _, idx, _ in processed_tensors])):
                sym['out'] = out

        output_subs = oe.parser.get_symbol(0)
        einsum_str = f"{','.join(input_subs)}->{output_subs}"
        return einsum_str, symbols

    def compile_dataset(self, tn_df):
        # blueprint_df = pd.DataFrame(index=tn_df.index)
        tn_cols = [col for col in tn_df.columns if col.endswith('_diagram')]
        for col in tn_cols:
            base_label = col.replace('_diagram', '')
            einsum_col = f"{base_label}_einsum"
            symbols_col = f"{base_label}_symbols"

            first_val = tn_df[col].iloc[0]
            is_nested_list = (isinstance(first_val, list) 
                              and len(first_val) > 0 
                              and isinstance(first_val[0], list)
                             )
            einsum_arr, symbols_arr = [], []

            rows = tn_df[col].tolist()
            workers = int(os.environ.get('TN_PATH_WORKERS', 0)) or len(os.sched_getaffinity(0))
            if len(rows) >= 20000 and workers > 1:   # big sets: rows compiled in a fork pool (same results, same order)
                global _POOL_COMPILER
                _POOL_COMPILER = self
                with multiprocessing.get_context('fork').Pool(workers) as pool:
                    out = pool.map(partial(_compile_row, nested=is_nested_list), rows, chunksize=2000)
            else:
                out = [_compile_row(r, is_nested_list, self) for r in tqdm(rows)]
            for e, syms in out:
                einsum_arr.append(e)
                symbols_arr.append(syms)
            tn_df[einsum_col] = einsum_arr
            tn_df[symbols_col] = symbols_arr
        return tn_df
    

_POOL_COMPILER = None

def _compile_row(row_val, nested, comp=None):
    # one dataframe cell: a caption's diagram, or (nested, decided per column) a list of them (several captions per image)
    comp = comp or _POOL_COMPILER
    if nested:
        pairs = [comp.compile_tn(tn) for tn in row_val]
        return [e for e, _ in pairs], [syms for _, syms in pairs]
    return comp.compile_tn(row_val)


def out_legs(idx_lists):
    # leg of each tensor towards the output wire (index 0), by a walk from the output over the (tree) network;
    # None for tensors without legs or not reached (a cycle or a disconnected part)
    by_idx = {}
    for t, idx in enumerate(idx_lists):
        for leg, i in enumerate(idx):
            by_idx.setdefault(i, []).append((t, leg))
    out = [None] * len(idx_lists)
    frontier = [(0, None)]   # (index, tensor it came from)
    while frontier:
        i, src = frontier.pop()
        for t, leg in by_idx.get(i, []):
            if t == src or out[t] is not None:
                continue
            out[t] = leg
            frontier += [(j, t) for k, j in enumerate(idx_lists[t]) if k != leg]
    return out
