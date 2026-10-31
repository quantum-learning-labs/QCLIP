class MPS: 
    def __init__(self, bond_dim: int, max_order: int, root_bond: int = None):
        self.bond_dim = bond_dim 
        self.max_order = max_order
        self.root_bond = root_bond   # text.root_bond_dim: bond of the tensors that carry the sentence output leg (type 'out')

    def decompose(self, symbol: str, idx_arr: list[int], dims: list[int], idx_counter):
        order = len(idx_arr)
        if order <= self.max_order:
            return [(symbol, idx_arr, dims)]
        
        bond = self.root_bond if self.root_bond and 'out' in symbol.split(':')[-1].split('@') else self.bond_dim
        cores = []
        vbonds = [next(idx_counter) for _ in range(order - 1)]

        for k in range(order):
            if k == 0:
                core_idx = [idx_arr[0], vbonds[0]]
                core_dims = [dims[0], bond]
            elif k == order - 1:
                core_idx = [vbonds[-1], idx_arr[-1]]
                core_dims = [bond, dims[-1]]
            else:
                core_idx = [vbonds[k-1], idx_arr[k], vbonds[k]]
                core_dims = [bond, dims[k], bond]
            cores.append((f"{symbol}_core{k}", core_idx, core_dims))
        return cores