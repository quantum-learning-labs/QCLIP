from collections import Counter, defaultdict
import torch
import numpy as np
from tqdm import tqdm
from opt_einsum import contract_path

def einsum2interleaved(expr):
    if isinstance(expr, str):
        if '->' not in expr:
            raise ValueError("Invalid einsum string format. Missing '->' operator.")
            
        lhs, rhs = expr.split('->')
        input_indices = [list(tok.strip()) for tok in lhs.split(',') if tok.strip()]
        out_list = list(rhs.strip())
        
        return [input_indices, out_list]
    
def interleaved2einsum(input_indices, out_list):
    lhs = ",".join("".join(tensor) for tensor in input_indices)
    rhs = "".join(out_list)
    
    return f"{lhs}->{rhs}"

def sort_tn(tn):
    remaining = list(tn)
    sorted_tn = []

    counts = Counter(sum([tensor[1] for tensor in tn], []))
    boundary_indices = {k for k, v in counts.items() if v == 1}

    for tensor in list(remaining):
        if any(idx in boundary_indices for idx in tensor[1]):
            sorted_tn.append(tensor)
            remaining.remove(tensor)
    
    current_indices = set()
    for tensor in sorted_tn:
        current_indices.update(tensor[1])

    while remaining:
        for tensor in list(remaining):
            if any(idx in current_indices for idx in tensor[1]):
                sorted_tn.append(tensor)
                remaining.remove(tensor)
                current_indices.update(tensor[1])
                break

    return list(reversed(sorted_tn))

def tn_metadata(data_arr):
    max_nq = max_gates = max_width = max_cdepth = max_2q_gates = 0
    avg_nq = avg_gates = avg_width = avg_cdepth = avg_2q_gates = 0
    N = len(data_arr)
    path_cache = {}
    for einsum_expr, tarr in tqdm(data_arr):
        nq, ngates, cdepth, width, twoq_gates = analyse_einsum(einsum_expr, tarr, cache=path_cache)
        max_nq = max(max_nq, nq)
        max_gates = max(max_gates, ngates)
        max_cdepth = max(max_cdepth, cdepth)
        max_width = max(max_width, width)
        max_2q_gates = max(max_2q_gates, twoq_gates)
        avg_nq += nq
        avg_gates += ngates
        avg_cdepth += cdepth
        avg_width += width
        avg_2q_gates += twoq_gates
    avg_width /= N
    avg_nq /= N
    avg_cdepth /= N
    avg_gates /= N
    avg_2q_gates /= N
    return {'max': (max_nq, max_gates, max_cdepth, max_width, max_2q_gates), 
            'avg': (int(round(avg_nq)), int(round(avg_gates)), int(round(avg_cdepth)), int(round(avg_width)), int(round(avg_2q_gates)))}

def analyse_einsum(einsum_expr, tarr, cache={}):
    op_types = tuple(op['op_type'] for op in tarr)
    cache_key = (einsum_expr, op_types)
    if cache_key in cache:
        return cache[cache_key]

    qubit_depths = defaultdict(int)
    shapes = []
    nq = 0
    twoq_gates = 0

    input_subs = einsum_expr.split('->')[0].split(',')
            
    for subscript, gate in zip(input_subs, tarr):
        op_type = gate['op_type']

        data_shape = torch.Size([2] * len(subscript))
        shapes.append(data_shape)

        if op_type == '0': nq += 1
        elif len(subscript) == 4: twoq_gates += 1

        if op_type not in ['0']:
            current_gate_max = 0
            for char in subscript:
                current_gate_max = max(current_gate_max, qubit_depths[char])
            new_depth = current_gate_max + 1
            for char in subscript:
                qubit_depths[char] = new_depth

    cdepth = max(qubit_depths.values()) if qubit_depths else 0
    path_info = contract_path(einsum_expr, *shapes, shapes=True)
    max_width = int(np.log2(float(path_info[1].largest_intermediate)))
    ngates = len(tarr) - nq
    cache[cache_key] = (nq, ngates, cdepth, max_width, twoq_gates)
    return nq, ngates, cdepth, max_width, twoq_gates
