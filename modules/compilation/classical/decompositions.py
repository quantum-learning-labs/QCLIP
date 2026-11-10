class MPS: 
    def __init__(self, bond_dim: int, max_order: int, root_bond: int = None, out_bond: int = None, share_out: bool = False, share_args: str = None):
        self.bond_dim = bond_dim 
        self.max_order = max_order
        self.root_bond = root_bond   # text.root_bond_dim: bond of the tensors that carry the sentence output leg (type 'out')
        # step 4c (output span; default off). out_bond (text.out_bond_dim): only the two bonds next to the core that holds the
        # output leg get this dim (span per root type <= out_bond^2 instead of bond^2), the rest keep bond_dim. share_out
        # (text.share_out): that core is one tensor shared by all root types ('o:out_mid' (L, out, R); 'o:out_end0' / 'o:out_end1'
        # when the output leg is first / last), so every caption is written into one common output space
        self.out_bond = out_bond
        self.share_out = share_out
        # step 5 (default off). share_args (text.share_args): the argument cores (every core but the word leg's and the output
        # leg's) are shared across complex types, keyed by their atomic leg type, side of the word leg and core shape, plus the
        # distance from the word leg for 'leg_dist' (S1); 'leg' (S2) drops the distance. Name 'a:<leg>.<L|R><dist>.<shape>';
        # the word-leg core stays per complex type (it carries the type's identity)
        assert share_args in (None, 'leg_dist', 'leg'), share_args
        self.share_args = share_args

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
            if self.share_args:
                legs = symbol.split(':')[-1].split('@')
                mid = (order - 1) // 2
                if k != mid:
                    leg = legs[k if k < mid else k - 1]
                    if leg != 'out':
                        dist = abs(k - mid) if self.share_args == 'leg_dist' else ''
                        name = f"a:{leg}.{'L' if k < mid else 'R'}{dist}.{'x'.join(map(str, core_dims))}"
            if self.share_out and k == out_pos:
                name = 'o:out_end0' if k == 0 else 'o:out_end1' if k == order - 1 else 'o:out_mid'
            cores.append((name, core_idx, core_dims))
        return cores


def _arg_name(symbol, k, mid, shape, share_args):
    # name of the factor/core at leg position k (word leg at mid) under text.share_args; None = keep the per-type name
    if not share_args or k == mid:
        return None
    leg = symbol.split(':')[-1].split('@')[k if k < mid else k - 1]
    if leg == 'out':
        return None
    dist = abs(k - mid) if share_args == 'leg_dist' else ''
    return f"a:{leg}.{'L' if k < mid else 'R'}{dist}.{'x'.join(map(str, shape))}"


class TR(MPS):
    # step 5 (text.decomp: tr): tensor ring, the MPS chain closed by one more bond between the last and the first core, so every
    # core is (bond, leg, bond). Options as MPS (out_bond widens the two bonds of the output core; share_out names it
    # 'o:out_tr'; share_args as MPS). Order <= max_order stays dense.
    def decompose(self, symbol: str, idx_arr: list[int], dims: list[int], idx_counter):
        order = len(idx_arr)
        if order <= self.max_order:
            return [(symbol, idx_arr, dims)]
        is_root = 'out' in symbol.split(':')[-1].split('@')
        bond = self.root_bond if self.root_bond and is_root else self.bond_dim
        bonds = [bond] * order                          # bond j joins core j and core j+1 (mod order)
        mid = (order - 1) // 2
        out_pos = None
        if is_root:
            out_pos = symbol.split(':')[-1].split('@').index('out')
            out_pos += out_pos >= mid
            if self.out_bond:
                bonds[out_pos - 1] = bonds[out_pos] = self.out_bond
        vb = [next(idx_counter) for _ in range(order)]
        cores = []
        for k in range(order):
            shape = [bonds[k - 1], dims[k], bonds[k]]
            name = _arg_name(symbol, k, mid, shape, self.share_args) or f"{symbol}_core{k}"
            if self.share_out and k == out_pos:
                name = 'o:out_tr'
            cores.append((name, [vb[k - 1], idx_arr[k], vb[k]], shape))
        return cores


class CP:
    # step 5 (text.decomp: cp): canonical polyadic, T = sum_r prod_k A_k[i_k, r]: one factor matrix (leg dim, rank) per leg, all
    # sharing one rank index (a hyperedge); the output span of a root type is <= rank. share_out: the output factor is one
    # (out, rank) matrix 'o:cp_out' for every root type; share_args as MPS (factor shape in the key). The word-leg factor stays
    # per type (the type's identity). Order <= max_order stays dense.
    def __init__(self, rank: int, max_order: int, share_out: bool = False, share_args: str = None):
        assert share_args in (None, 'leg_dist', 'leg'), share_args
        self.rank, self.max_order, self.share_out, self.share_args = rank, max_order, share_out, share_args

    def decompose(self, symbol: str, idx_arr: list[int], dims: list[int], idx_counter):
        order = len(idx_arr)
        if order <= self.max_order:
            return [(symbol, idx_arr, dims)]
        legs = symbol.split(':')[-1].split('@')
        mid = (order - 1) // 2
        r = next(idx_counter)
        out = []
        for k in range(order):
            shape = [dims[k], self.rank]
            name = _arg_name(symbol, k, mid, shape, self.share_args) or f"{symbol}_core{k}"
            if self.share_out and k != mid and legs[k if k < mid else k - 1] == 'out':
                name = 'o:cp_out'
            out.append((name, [idx_arr[k], r], shape))
        return out
