from itertools import count
from collections import Counter
from abc import ABC, abstractmethod
from tqdm import tqdm
from modules.compilation.einsum import interleaved2einsum, sort_tn
from modules.symbolic.root import COPY_BOX

class BaseAnsatz(ABC):
    # Tensor network -> circuit compilation shared by every ansatz (wires, currying, discard, einsum strings).
    # A subclass defines only ansatz(): the gates of one word box. Register it in ANSATZE to select it with text.ansatz.
    def __init__(self, obmap=set(), layers=1):
        self.obmap = obmap
        self.layers = layers
        self.char_idx = count(0)

    def __call__(self, tn, curry=False, discard=False, spider=False):
        # spider() is placeholder code and not implemented properly yet
        if spider:
            raise NotImplementedError("spider ansatz is not implemented yet; set text.spider: false")
        if curry:
            return self.tn2ansatz_curried(tn,  discard=discard)
        else:
            return self.tn2ansatz(tn, discard=discard)

    def tns2ansatze(self, tn_arr, curry=False, discard=False, spider=False):
        return [self(tn, curry=curry, discard=discard, spider=spider) for tn in tn_arr]

    def reset_char(self):
        self.char_idx = count(0)

    def get_char(self):
        i = next(self.char_idx)
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return chars[i] if i < len(chars) else chr(192 + i - len(chars))

    def ccg_map(self, tn):
        ccg_map = {}
        for _, idx_arr, type_arr in tn:
            for idx, typ in zip(idx_arr, type_arr):
                if idx not in ccg_map:
                    ccg_map[idx] = [self.get_char() for _ in range(self.obmap.get(typ, 1))]
        return ccg_map

    def box(self, word, current_wires, base_symbol, curried):
        # structural boxes (text.root) have fixed gates shared by every ansatz; word boxes get the ansatz
        if word == COPY_BOX:
            return self.copy_gates(current_wires, curried)
        return self.ansatz(current_wires, base_symbol)

    def _cx(self, wires, c, t, new_indices, new_tensors):
        c_out, t_out = self.get_char(), self.get_char()
        new_indices.append(wires[c] + wires[t] + c_out + t_out)
        new_tensors.append({'name': None, 'op_type': 'CX'})
        wires[c], wires[t] = c_out, t_out

    def copy_gates(self, current_wires, curried):
        # Frobenius copy N -> N^(x)m, |i> -> |i>^(x)m, on wires [n input qubits] + [s = m*n output qubits, copy k qubit j at k*n+j].
        # curried (input wires carry the state): CX fan-out to every copy, then uncompute the input from copy 0, which leaves
        #   the input wires in |0> exactly (their post-selection succeeds with probability 1; under discard they trace out).
        # uncurried (the box is a state joined by a cup): sum_i |i>|i>^(x)m, i.e. H on the input then the fan-out.
        n, s = self.obmap['n'], self.obmap['s']
        if s % n:
            raise ValueError(f"text.root frobenius needs s a multiple of n (s={s}, n={n})")
        if len(current_wires) == n:   # s == n, curried: a pass-through box, and the copy N -> N is the identity
            return list(current_wires), [], []
        wires, new_indices, new_tensors = list(current_wires), [], []
        if not curried:
            for j in range(n):
                nxt = self.get_char()
                new_indices.append(wires[j] + nxt)
                new_tensors.append({'name': None, 'op_type': 'H'})
                wires[j] = nxt
        for j in range(n):
            for k in range(s // n):
                self._cx(wires, j, n + k * n + j, new_indices, new_tensors)
            if curried:
                self._cx(wires, n + j, j, new_indices, new_tensors)
        return wires, new_indices, new_tensors

    @abstractmethod
    def ansatz(self, current_wires, base_symbol):
        # Append the gates of one word box acting on current_wires (list of index chars, updated in place).
        # Parameterised gates are named f"{base_symbol}..."; return (current_wires, new_indices, new_tensors).
        pass


    def gen_einsum_expr(self, input_indices, ccg_map):
        all_idx = [i for sub in input_indices for i in sub]
        idx_counts = Counter(all_idx)
        output_indices = [idx for idx, cnt in idx_counts.items() if cnt == 1]

        if 0 in ccg_map:
            root_indices = ccg_map[0]
        elif '0' in ccg_map:
            root_indices = ccg_map['0']
        elif len(ccg_map) > 0:
            root_indices = list(ccg_map.values())[0]
        else:
            root_indices = []

        # root_indices = ccg_map.get(0, ccg_map.get('0', []))
        out_list = [i for i in root_indices if i in output_indices] + \
                   [i for i in output_indices if i not in root_indices]
        
        return interleaved2einsum(input_indices, out_list)
    
    def tn2ansatz(self, tn, discard=False):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []

        for word, idx_arr, type_arr in tn:
            N = sum(self.obmap.get(typ, 1) for typ in type_arr)
            current_wires = [self.get_char() for _ in range(N)]

            for w in current_wires:
                input_indices.append([w])
                # tensor_arr.append((None, '0'))
                tensor_arr.append({'name': None, 'op_type': '0'})

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.box(word, current_wires, base_symbol, curried=False)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            i = 0
            for idx, typ in zip(idx_arr, type_arr):
                n = self.obmap.get(typ, 1)
                if idx not in ccg_map:
                    ccg_map[idx] = current_wires[i:i+n]
                else:
                    if not discard:
                        target_wires = ccg_map[idx]                    
                        replace_map = dict(zip(current_wires[i:i+n], target_wires))
                        input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]
                i += n
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr
    
    def tn2ansatz_curried(self, tn, discard=False):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []
        tn_sorted = sort_tn(tn)

        for word, idx_arr, type_arr in tn_sorted:
            current_wires = []

            input_wires, input_types = [], []
            output_wires, output_types = [], []
            for idx, atomic_type in zip(idx_arr, type_arr):
                if idx in ccg_map:
                    input_wires.append(idx)
                    input_types.append(atomic_type)
                else:
                    output_wires.append(idx)
                    output_types.append(atomic_type)
            # pass-through only if each output wire can reuse an input wire of the same width
            # (an intransitive verb maps n (1 qubit) -> out (embedding_qubits), so it must not)
            same_widths = all(self.obmap.get(t_in, 1) == self.obmap.get(t_out, 1)
                              for t_in, t_out in zip(input_types, output_types))
            passthrough = len(input_wires) == len(output_wires) and same_widths
            if passthrough:
                for in_idx, out_idx in zip(input_wires, output_wires):
                    current_wires.extend(ccg_map[in_idx])
                    ccg_map[out_idx] = ccg_map[in_idx]
                    del ccg_map[in_idx]
            else:
                for idx, atomic_type in zip(input_wires, input_types):
                    current_wires.extend(ccg_map[idx])
                for idx, atomic_type in zip(output_wires, output_types):
                    n = self.obmap.get(atomic_type, 1)
                    new_wires = [self.get_char() for _ in range(n)]
                    current_wires.extend(new_wires)
                    ccg_map[idx] = new_wires
                    input_indices.extend([[w] for w in new_wires])
                    tensor_arr.extend([{'name': None, 'op_type': '0'}] * n)

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.box(word, current_wires, base_symbol, curried=True)
            input_indices.extend([[idx for idx in ten] for ten in new_indices])
            tensor_arr.extend(new_tensors)
            
            if passthrough:   # the outputs carry the box's wires on; the inputs are used up
                i = 0
                for idx, typ in zip(output_wires, output_types):
                    n = self.obmap.get(typ, 1)
                    ccg_map[idx] = current_wires[i:i+n]
                    i += n
                continue
            i = 0
            for idx, typ in zip(idx_arr, type_arr):
                n = self.obmap.get(typ, 1)
                ccg_map[idx] = current_wires[i:i+n]
                i += n

        if not discard:
            for virtual_idx, physical_wires in ccg_map.items():
                if virtual_idx != 0:
                    input_indices.extend([[w] for w in physical_wires])
                    # tensor_arr.extend([(None, '0_dag')] * len(physical_wires))
                    tensor_arr.extend([{'name': None, 'op_type': '0_dag'}] * len(physical_wires))
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr
    
    def spider(self, tn):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []

        idx = 0
        N = self.obmap.get('out', 1)
        for word, _, type_arr in tn:
            current_wires = [self.get_char() for _ in range(N)]

            for w in current_wires:
                input_indices.append([w])
                tensor_arr.append({'name': None, 'op_type': '0'})

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            ccg_map[idx] = current_wires
            idx += 1

        target_wires = ccg_map[0]                    
        replace_map = {}
        for i in range(idx):
            for j, output_wires in enumerate(ccg_map[i]):
                replace_map[output_wires] = target_wires[j]
        input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]
        for i in range(idx):
            ccg_map[i] = target_wires
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        lhs, _ = einsum_expr.split('->')
        einsum_expr = f"{lhs}->{''.join(target_wires)}"
        return einsum_expr, tensor_arr

    def compile_dataset(self, df, curry=False, discard=False, spider=False):
        #blueprint_df = pd.DataFrame(index=df.index)
        cols = [col for col in df.columns if col.endswith('_diagram')]
        for col in cols:
            base_label = col.replace('_diagram', '')
            einsum_col = f"{base_label}_einsum"
            symbols_col = f"{base_label}_symbols"

            first_val = df[col].iloc[0]
            is_nested_list = (isinstance(first_val, list) 
                              and len(first_val) > 0 
                              and isinstance(first_val[0], list)
                             )
            einsum_arr, symbols_arr = [], []

            for row_val in tqdm(df[col]):
                if is_nested_list:
                    row_einsums, row_symbols = [], []
                    for tn in row_val:
                        e_str, syms = self(tn, curry=curry, discard=discard, spider=spider)
                        row_einsums.append(e_str)
                        row_symbols.append(syms)
                    einsum_arr.append(row_einsums)
                    symbols_arr.append(row_symbols)
                else:
                    e_str, syms = self(row_val, curry=curry, discard=discard, spider=spider)
                    einsum_arr.append(e_str)
                    symbols_arr.append(syms)
            df[einsum_col] = einsum_arr
            df[symbols_col] = symbols_arr
        return df


class CustomV5Ansatz(BaseAnsatz):
    # H on each wire, then per layer Rz-Ry-Rz on each wire and a ring of CX
    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for i in range(N):
            nxt = self.get_char()
            new_indices.append(current_wires[i] + nxt)
            new_tensors.append({'name': None, 'op_type': 'H'})
            current_wires[i] = nxt

        for l in range(self.layers):
            op_idx = 0
            for i in range(N):
                for g in ['Rz', 'Ry', 'Rz']:
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': g})
                    current_wires[i] = nxt
                    op_idx += 1
            if N > 1:
                for i in range(N):
                    c_idx, t_idx = i, (i + 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    new_tensors.append({'name': None, 'op_type': 'CX'})
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
        return current_wires, new_indices, new_tensors


ANSATZE = {'custom_v5': CustomV5Ansatz}

def build_ansatz(name, obmap, layers=1):
    if name not in ANSATZE:
        raise ValueError(f"unknown ansatz {name!r}; registered: {sorted(ANSATZE)}")
    return ANSATZE[name](obmap=obmap, layers=layers)
