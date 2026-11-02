class MPS: 
    def __init__(self, bond_dim: int, max_order: int, root_bond: int = None, out_bond: int = None, share_out: bool = False):
        self.bond_dim = bond_dim 
        self.max_order = max_order
        self.root_bond = root_bond   # text.root_bond_dim: bond of the tensors that carry the sentence output leg (type 'out')
        # step 4c (output span; default off). out_bond (text.out_bond_dim): only the two bonds next to the core that holds the
        # output leg get this dim (span per root type <= out_bond^2 instead of bond^2), the rest keep bond_dim. share_out
        # (text.share_out): that core is one tensor shared by all root types ('o:out_mid' (L, out, R); 'o:out_end0' / 'o:out_end1'
        # when the output leg is first / last), so every caption is written into one common output space
        self.out_bond = out_bond
        self.share_out = share_out

    def decompose(self, symbol: str, idx_arr: list[int], dims: list[int], idx_counter):
        order = len(idx_arr)
        if order <= self.max_order:
            return [(symbol, idx_arr, dims)]
        
        is_root = 'out' in symbol.split(':')[-1].split('@')
        bond = self.root_bond if self.root_bond and is_root else self.bond_dim
        bonds = [bond] * (order - 1)
        out_pos = None
        if is_root and (self.out_bond or self.share_out):
            out_pos = symbol.split(':')[-1].split('@').index('out')
            mid = (order - 1) // 2   # the compiler puts the word leg at position mid of the type's legs
            out_pos += out_pos >= mid
            if self.out_bond:
                for j in (out_pos - 1, out_pos):
                    if 0 <= j < order - 1:
                        bonds[j] = self.out_bond
        cores = []
        vbonds = [next(idx_counter) for _ in range(order - 1)]

        for k in range(order):
            if k == 0:
                core_idx = [idx_arr[0], vbonds[0]]
                core_dims = [dims[0], bonds[0]]
            elif k == order - 1:
                core_idx = [vbonds[-1], idx_arr[-1]]
                core_dims = [bonds[-1], dims[-1]]
            else:
                core_idx = [vbonds[k-1], idx_arr[k], vbonds[k]]
                core_dims = [bonds[k-1], dims[k], bonds[k]]
            name = f"{symbol}_core{k}"
            if self.share_out and k == out_pos:
                name = 'o:out_end0' if k == 0 else 'o:out_end1' if k == order - 1 else 'o:out_mid'
            cores.append((name, core_idx, core_dims))
        return cores
