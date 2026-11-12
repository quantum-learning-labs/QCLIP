from collections import Counter

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
