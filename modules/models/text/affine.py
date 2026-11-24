"""Step 5e (text.affine): affine-multilinear type tensors by homogeneous coordinates.

Every grammatical and word wire gets one extra slot h (the last index) that carries a constant 1, and every CP rank leg one
extra component r* (its last index). A tensor's leg towards the sentence output (TNCompiler orient) is its OUTPUT leg; the
others are its arguments. Pinned entries (by masks applied to the parameter rows at every forward) keep the slot exact:

  word vectors        v[h] = 1
  CP factor, argument column r* = e_h (it reads the argument's 1); row h, r < R = bias c_r (trainable)
  CP factor, output   column r* = e_h, row h = e_r*, so the output's slot is prod of the arguments' slots = 1, and the
                      trainable components r < R never write into it
  dense tensor, out k slice i_k = h equals prod_{j != k} e_h(i_j); entries with an argument index at h are biases

Contracting [x; 1] instead of x gives prod_k (a_k.x_k + c_k) per rank component: the full multilinear term plus every
lower-order (additive) term down to a constant, the TN analogue of an RNTN's W x and a Tree-LSTM's additive path, with no
nonlinearity. The 1 is projective: scale_free's per-intermediate rescaling scales the whole vector, slot included, so the
represented function is unchanged. A tensor used with different orientations reads the same parameter row through a
different mask: the compiler names output-view uses '<symbol>^o' (CP factor) or '<symbol>^<k>' (dense, output leg k).
Legs of dimension out_dim (the 512-d sentence leg) carry no slot."""
import torch

SEP = '^'


def split_alias(name):
    if SEP in name and not name.startswith('w:'):
        base, view = name.rsplit(SEP, 1)
        return base, view
    return name, None


def is_cp_factor(name):
    return name.startswith('a:') or name.startswith('o:') or '_core' in name


def leg_kinds(name, shape, out_dim):
    # 'h' = wire with a homogeneous slot, 'o' = sentence leg (no slot), 'r' = CP rank leg (extra component r*)
    if name.startswith('w:'):
        return ['h']
    if is_cp_factor(name) and len(shape) == 2:
        return ['o' if shape[0] == out_dim else 'h', 'r']
    return ['o' if d == out_dim else 'h' for d in shape]


def alias_for(name, shape, out, out_dim):
    # compiler: the name of one use of a tensor given its output leg `out` (None = not reached from the output)
    if name.startswith('w:') or out is None:
        return name
    kinds = leg_kinds(name, shape, out_dim)
    if kinds[out] == 'r':   # CP factor read as an argument: the base name is the argument view
        return name
    if is_cp_factor(name) and len(shape) == 2:
        return name + SEP + 'o'
    return name + SEP + str(out)


def row_mask(name, shape, view, out_dim):
    # (M, C): the row used is row * M + C
    M, C = torch.ones(shape), torch.zeros(shape)
    kinds = leg_kinds(name, shape, out_dim)
    if name.startswith('w:'):
        M[-1] = 0; C[-1] = 1
        return M, C
    if is_cp_factor(name) and len(shape) == 2:
        M[:, -1] = 0
        if kinds[0] == 'h':
            C[-1, -1] = 1   # column r* = e_h
            if view == 'o':
                M[-1, :] = 0   # row h = e_r* (C[-1, -1] = 1 already)
        return M, C
    if view is None or view == 'o':
        return M, C
    k = int(view)
    if kinds[k] != 'h' or any(kinds[j] != 'h' for j in range(len(shape)) if j != k):
        return M, C
    idx = [slice(None)] * len(shape); idx[k] = -1
    M[tuple(idx)] = 0
    unit = [slice(None)] * len(shape); unit = tuple(-1 for _ in shape)
    C[unit] = 1
    return M, C


@torch.no_grad()
def zero_bias(row, name, shape, out_dim):
    # init: every entry with an index at a slot (bias or pinned) set to 0, so the model starts as the plain multilinear one
    kinds = leg_kinds(name, shape, out_dim)
    if name.startswith('w:'):
        return
    for k, kind in enumerate(kinds):
        if kind in ('h', 'r'):
            idx = [slice(None)] * len(shape); idx[k] = -1
            row[tuple(idx)] = 0
