from __future__ import annotations
import pandas as pd
import opt_einsum as oe
from itertools import count
from tqdm import tqdm
from collections import Counter
    
class TNCompiler: 
    def __init__(self, obmap: dict[str, int], decomp_fn, decouple: bool = False, orient: bool = False, lift=None,
                 lift_rootmap: str = None, own_pairs: str = None):
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
        # lift (text.lift, decoupled only, default off; E2): Frobenius lifting of higher-order roles from one core per word
        # (modules/compilation/classical/lift.py). 'frobenius' = all rules, or a list of rule names ['R1', ...]
        self.lift = None
        if lift:
            from modules.compilation.classical.lift import RULES, load_rootmap
            if not decouple:
                raise ValueError('text.lift needs text.decouple: true')
            self.lift = tuple(RULES) if lift == 'frobenius' else tuple(lift)
            assert set(self.lift) <= set(RULES) and 'R1' in self.lift, self.lift   # the other rules assume R1 at the root
            self.rootmap = load_rootmap(lift_rootmap or 'modules/compilation/classical/e2_rootmap.json')
            self.lift_stats = Counter()
            self.id += '_lift' + ''.join(r[1:] for r in self.lift)
        # own_pairs (text.own_pairs, decoupled only, default off; E1): JSON from tools/tn/s5_e1_pairs.py. Each listed
        # (relation word, type symbol) occurrence gets its own tensor 'u:<word>|<symbol>' (no word leg, own argument factors:
        # never shared via share_args); every other occurrence keeps the shared word-vector + type-tensor path
        self.own = None
        if own_pairs:
            import json, copy, hashlib
            if not decouple:
                raise ValueError('text.own_pairs needs text.decouple: true')
            if not self.lift:   # R1 maps every root to an atomic type: own tensors never carry the 'out' leg
                raise ValueError('text.own_pairs needs text.lift (R1 at least)')
            raw = open(own_pairs, 'rb').read()
            self.own = frozenset(json.loads(raw)['pairs'])
            self.own_decomp = copy.copy(decomp_fn)
            if self.own_decomp is not None and hasattr(self.own_decomp, 'share_args'):
                self.own_decomp.share_args = None
            self.id += '_own' + hashlib.md5(raw).hexdigest()[:8]

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

        uf = None
        if self.lift:
            from modules.compilation.classical.lift import lift
            items, extras, uf, stats = lift(tn, self.lift, self.rootmap, index_counter)
            self.lift_stats.update(stats); self.lift_stats['words'] += len(tn); self.lift_stats['captions'] += 1
        else:
            items, extras = [(w, i, t, None) for w, i, t in tn], []
        for word, idx_arr, type_arr, type_symbol in items:
            if len(idx_arr) != len(type_arr):
                raise ValueError(f"Mismatch between indices and types for word '{word}' in sentence '{' '.join([w for w, _, _ in tn])}': "
                                 f"indices={idx_arr}, types={type_arr}")
            symbol = type_symbol or 't:' + '@'.join(type_arr)
            if self.own and f'{word}|{symbol}' in self.own:   # E1: the word's own tensor for this role
                dims = [self.obmap.get(t, 1) for t in type_arr]
                own_sym = f'u:{word}|{symbol}'
                if self.own_decomp is None:
                    processed_tensors.append((own_sym, list(idx_arr), tuple(dims)))
                else:
                    processed_tensors.extend(self.own_decomp.decompose(own_sym, list(idx_arr), dims, index_counter))
                continue
            wire = next(index_counter)
            processed_tensors.append(('w:' + word, [wire], (self.word_dim,)))

            mid = len(idx_arr) // 2
            type_idx = list(idx_arr[:mid]) + [wire] + list(idx_arr[mid:])
            dims = [self.obmap.get(t, 1) for t in type_arr]
            type_dims = dims[:mid] + [self.word_dim] + dims[mid:]
            if self.decomposition is None:
                processed_tensors.append((symbol, type_idx, tuple(type_dims)))
            else:
                cores = self.decomposition.decompose(symbol, type_idx, type_dims, index_counter)
                processed_tensors.extend(cores)
        for symbol, idx_arr, type_arr in extras:   # lift: codomain and s<->n maps (no word leg, never decomposed)
            processed_tensors.append((symbol, list(idx_arr), tuple(self.obmap.get(t, 1) for t in type_arr)))
        if uf is not None:   # spiders: every leg of one spider gets the same index (a hyperedge of the einsum)
            processed_tensors = [(s, [uf.find(i) for i in idx], sh) for s, idx, sh in processed_tensors]
            # one output path: with R1 no type tensor keeps an 'out' leg (the codomain maps m:X->out are the only route out)
            assert not any(s[0] in 'tu' and 'out' in s.split(':')[-1].split('_core')[0].split('@') for s, _, _ in processed_tensors), \
                [s for s, _, _ in processed_tensors]

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

            for row_val in tqdm(tn_df[col]):
                if is_nested_list:
                    row_einsums, row_symbols = [], []
                    for tn in row_val:
                        e_str, syms = self.compile_tn(tn)
                        row_einsums.append(e_str)
                        row_symbols.append(syms)
                    einsum_arr.append(row_einsums)
                    symbols_arr.append(row_symbols)
                else:
                    e_str, syms = self.compile_tn(row_val)
                    einsum_arr.append(e_str)
                    symbols_arr.append(syms)
            tn_df[einsum_col] = einsum_arr
            tn_df[symbols_col] = symbols_arr
        return tn_df
    

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
