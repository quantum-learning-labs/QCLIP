from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch, math
import torch.nn as nn
from opt_einsum import contract_expression, contract_path
from opt_einsum.contract import ContractExpression
from opt_einsum import parser as oe_parser
from opt_einsum.parser import convert_to_valid_einsum_chars
from tqdm import tqdm
import os, zlib, multiprocessing
import numpy as np

class TNModel(nn.Module):
    real_scores = True   # real embeddings: image-text scores are plain dot products (no |.| readout)

    def __init__(self, out_dim: int = 512, engine: str = 'grouped'):
        super().__init__()
        # 'grouped': one batched contraction per sentence structure (step-1 reference engine). 'levelwise': the same
        # pairwise steps (opt_einsum path of each structure), executed step by step for all sentences at once, one
        # batched einsum per distinct (equation, operand shapes): far fewer kernel launches, same maths up to float
        # rounding of differently batched kernels
        if engine not in ('grouped', 'levelwise'):
            raise ValueError(f"text.tn_engine must be 'grouped' or 'levelwise', got {engine!r}")
        self.engine = engine
        self.symbols: List[str] = []
        self.shapes: List[Tuple[int, ...]] = []
        # word/core tensors packed in one bank per distinct shape (parameters bank0, bank1, ...): a step gathers the rows
        # it uses with one index_select per bank. sym_loc: symbol -> (bank, row). Checkpoints keep one 'params.<symbol>'
        # entry per tensor, as before packing (see _save_to_state_dict).
        self.sym_loc: Dict[str, Tuple[int, int]] = {}
        self.bank_shapes: List[Tuple[int, ...]] = []
        self._active = []   # (bank, unique rows, leaf) gathered by training forwards since the last optimiser step
        self.path_cache = {}
        self.level_cache = {}   # (batched einsum, shapes) -> levelwise plan (precompute_paths)
        self.out_dim: int = out_dim
        self.norm_head = nn.LayerNorm(out_dim)
    
    @property
    def pcount(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def grad_norm(self):
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    

    def from_symbols(self, tn_arr):
        symbols_hash = set(self.symbols)

        for tensor_arr in tqdm(tn_arr):
            flat_tensors = self._flatten_symbols(tensor_arr)
            for symbol_info in flat_tensors:
                symbol = symbol_info['name']
                if symbol not in symbols_hash:
                    symbols_hash.add(symbol)
                    self.symbols.append(symbol)
                    self.shapes.append(tuple(symbol_info['shape']))

        self.init_params()

    def from_state_dict(self, state_dict):
        # vocabulary (names, shapes, order) from a checkpoint's own tensors: evaluation needs no train/val recompile
        for key, val in state_dict.items():
            if key.startswith('params.'):
                self.symbols.append(key[len('params.'):]); self.shapes.append(tuple(val.shape))
        self._build_banks(init=False)

    def _flatten_symbols(self, val):
        if isinstance(val, dict) and 'name' in val:
            return [val]
        elif isinstance(val, list):
            res = []
            for item in val:
                res.extend(self._flatten_symbols(item))
            return res
        return []

    @staticmethod
    def _init_tensor(shape, generator=None, out=None):
        # out: fill this tensor (a bank row) in place; the draws are the same as for a fresh tensor of that shape
        param = nn.Parameter(torch.empty(shape)) if out is None else out
        order = len(shape)
        if order == 0:
            nn.init.constant_(param, 1.0)
        elif order == 1:
            nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(shape[0]), generator=generator)
        else:
            fan_out = shape[0]
            fan_in = math.prod(shape[1:])
            std = math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(param, mean=0.0, std=std, generator=generator)
        return param

    def init_params(self):
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through ')
        self._build_banks(init=True)

    def _bank(self, b):
        return getattr(self, f'bank{b}')

    def _build_banks(self, init):
        # rows in symbol order; init=True draws each row from the global RNG in symbol order, i.e. the same values as
        # one parameter per symbol; init=False leaves them empty (filled by load_state_dict)
        bank_of, counts = {}, []
        for name, shape in zip(self.symbols, self.shapes):
            b = bank_of.get(shape)
            if b is None:
                b = bank_of[shape] = len(self.bank_shapes)
                self.bank_shapes.append(shape); counts.append(0)
            self.sym_loc[name] = (b, counts[b]); counts[b] += 1
        banks = [torch.empty(n, *shape) for n, shape in zip(counts, self.bank_shapes)]
        if init:
            with torch.no_grad():
                for name, shape in zip(self.symbols, self.shapes):
                    b, r = self.sym_loc[name]
                    self._init_tensor(shape, out=banks[b][r])
        for b, t in enumerate(banks):
            self.register_parameter(f'bank{b}', nn.Parameter(t))

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # one entry per symbol (views into the banks; torch.save stores each bank once): same keys as before packing
        for name in self.symbols:
            b, r = self.sym_loc[name]
            bank = self._bank(b)
            destination[prefix + 'params.' + name] = (bank if keep_vars else bank.detach())[r]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        with torch.no_grad():
            for name in self.symbols:
                key = prefix + 'params.' + name
                if key not in state_dict:
                    missing_keys.append(key); continue
                b, r = self.sym_loc[name]
                row = self._bank(b)[r]
                if tuple(state_dict[key].shape) != tuple(row.shape):
                    error_msgs.append(f"{key}: shape {tuple(state_dict[key].shape)} vs {tuple(row.shape)}"); continue
                row.copy_(state_dict[key])
        n = len(prefix)
        for key in state_dict:
            if key.startswith(prefix + 'params.') and key[n + len('params.'):] not in self.sym_loc:
                unexpected_keys.append(key)

    def extend_symbols(self, tn_arr, seed: int = 0):
        # eval-time out-of-vocabulary word@type tensors, initialised (untrained) from a generator seeded by (seed, name):
        # the same tensor gets the same values whichever test sets ran before. OOV rates are relative to the trained
        # vocabulary (the symbols present when this is first called).
        if not hasattr(self, 'trained_symbols'):
            self.trained_symbols = frozenset(self.symbols)
        known = set(self.symbols)
        n_sent = n_oov_sent = n_tensors = n_oov_tensors = 0
        new = {}
        for tensor_arr in tn_arr:
            flat = self._flatten_symbols(tensor_arr)
            unseen = [s for s in flat if s['name'] not in self.trained_symbols]
            n_sent += 1; n_oov_sent += bool(unseen)
            n_tensors += len(flat); n_oov_tensors += len(unseen)
            for s in unseen:
                if s['name'] not in known:
                    new.setdefault(s['name'], tuple(s['shape']))
        if new:
            device = self._bank(0).device
            rows = defaultdict(list)   # bank -> new rows (appended; existing rows keep their place)
            for name in sorted(new):
                shape = new[name]
                self.symbols.append(name); self.shapes.append(shape)
                if shape not in self.bank_shapes:
                    self.bank_shapes.append(shape)
                    self.register_parameter(f'bank{len(self.bank_shapes) - 1}', nn.Parameter(torch.empty(0, *shape, device=device)))
                b = self.bank_shapes.index(shape)
                gen = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
                self.sym_loc[name] = (b, self._bank(b).shape[0] + len(rows[b]))
                rows[b].append(self._init_tensor(shape, gen).data)
            for b, ts in rows.items():
                self.register_parameter(f'bank{b}', nn.Parameter(torch.cat([self._bank(b).data, torch.stack(ts).to(device)])))
        return {"oov_sentence_frac": n_oov_sent / max(n_sent, 1), "oov_tensor_frac": n_oov_tensors / max(n_tensors, 1),
                "oov_new_symbols": len(new)}

    def param_breakdown(self):
        # parameter count by tensor order (vectors, matrices, order-3 cores, ...)
        out = defaultdict(int)
        for b, shape in enumerate(self.bank_shapes):
            out[f"order{len(shape)}"] += self._bank(b).numel()
        return dict(out)

    def pop_row_grads(self):
        # for RowAdam: per bank, the rows the training forwards since the last step gathered and their gradients
        # (several forwards in one step, e.g. swapped-caption negatives, are merged)
        act, self._active = self._active, []
        by_bank = defaultdict(list)
        for b, rows, leaf in act:
            if leaf.grad is not None:
                by_bank[b].append((rows, leaf.grad))
        out = []
        for b, items in by_bank.items():
            if len(items) == 1:
                rows, grad = items[0]
            else:
                rows, inv = torch.unique(torch.cat([r for r, _ in items]), return_inverse=True)
                grad = torch.zeros(len(rows), *self.bank_shapes[b], device=rows.device).index_add_(0, inv, torch.cat([g for _, g in items]))
            out.append((self._bank(b), rows, grad))
        self._last_row_grads = out
        return out

    @property
    def last_row_grad_sq(self):
        # squared gradient norm by tensor order of the last optimiser step (grad_norm_by_type, once per epoch)
        sq = defaultdict(float)
        for bank, _, grad in getattr(self, '_last_row_grads', []):
            sq[f"order{bank.dim() - 1}"] += float(grad.pow(2).sum())
        return dict(sq)

    def compile_batch(self, batch: List[tuple[str, List[str]]]) -> torch.Tensor:
        groups = defaultdict(list)
        for einsum_expr, tensor_arr in batch:
            shapes, _ = zip(*[(tuple(sym_dict['shape']), sym_dict['name']) for sym_dict in tensor_arr])
            groups[(einsum_expr, shapes)].append(tensor_arr)
        
        for (einsum_expr, _), tensor_arr in groups.items():
            minibatch_size = len(tensor_arr)
            tensor_columns = zip(*tensor_arr)
            shapes = []
            for col in tensor_columns:
                shapes.append(torch.Size([minibatch_size, *col[0]['shape']]))

            input_subscripts, output_subscript = einsum_expr.split('->')
            batch_symbol = next(oe_parser.gen_unused_symbols(einsum_expr, 1))
            batched_inputs = [batch_symbol + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->{batch_symbol}{output_subscript}"

            cache_key = (batched_str, tuple(shapes))
            if cache_key not in self.path_cache: 
                self.path_cache[cache_key] = contract_expression(batched_str, *shapes)

    def _plan(self, einsum_expr, tensor_arr):
        # per-sentence plan, built once per sentence object (keyed by id, the entry keeps the object alive so ids are
        # never reused): structure id (einsum + shapes) and the bank row of each tensor
        hit = self._plans.get(id(tensor_arr))
        if hit is not None:
            return hit[1], hit[2]
        key = (einsum_expr, tuple(tuple(d['shape']) for d in tensor_arr))
        sid = self._struct_ids.get(key)
        if sid is None:
            sid = self._struct_ids[key] = len(self._structs)
            banks = tuple(self.sym_loc[d['name']][0] for d in tensor_arr)   # a slot's bank is fixed by its shape
            self._structs.append([_batched(einsum_expr), key[1], None, banks])
        rows = tuple(self.sym_loc[d['name']][1] for d in tensor_arr)
        self._plans[id(tensor_arr)] = (tensor_arr, sid, rows)
        return sid, rows

    def forward(self, batch: List[tuple[str, List[str]]]) -> torch.Tensor:
        if not hasattr(self, '_plans'):
            self._plans, self._struct_ids, self._structs = {}, {}, []
        groups = defaultdict(list)   # structure id -> [(row, bank rows)], in order of first appearance (as before)
        for i, (einsum_expr, tensor_arr) in enumerate(batch):
            sid, rows = self._plan(einsum_expr, tensor_arr)
            groups[sid].append((i, rows))

        if self.engine == 'levelwise':
            output = self._levelwise(groups)
            output = self.norm_head(output)
            return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)

        # one gather per bank for the whole batch: requests[b] = the row lists of every (group, slot) on bank b
        requests, refs = defaultdict(list), []
        for sid, items in groups.items():
            mat = np.array([r for _, r in items], dtype=np.int64).reshape(len(items), -1)
            ref = []
            for j, b in enumerate(self._structs[sid][3]):
                ref.append((b, len(requests[b]))); requests[b].append(mat[:, j])
            refs.append(ref)
        device = self._bank(0).device
        pieces = {}
        for b, lists in requests.items():
            idx = np.concatenate(lists)
            if torch.is_grad_enabled():
                # training: gather the unique rows into a small leaf; RowAdam updates exactly these rows from its grad
                uniq, inv = np.unique(idx, return_inverse=True)
                uniq = torch.from_numpy(uniq).to(device)
                leaf = self._bank(b).detach().index_select(0, uniq).requires_grad_()
                self._active.append((b, uniq, leaf))
                src, idx = leaf, inv
            else:
                src = self._bank(b)
            pieces[b] = src.index_select(0, torch.from_numpy(idx.reshape(-1)).to(device)).split([len(l) for l in lists])

        output = self._grouped(groups, refs, pieces)
        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)

    def _expression(self, struct, tensor_shapes):
        # contraction expression of a structure; path cached without the group size: the batch index is on every
        # tensor and the output, so it scales every pairwise cost equally and the path doesn't depend on it
        if struct[2] is None:
            cache_key = (struct[0], struct[1])
            hit = self.path_cache.get(cache_key)
            if not isinstance(hit, ContractExpression):   # missing, or a path from precompute_paths
                hit = self.path_cache[cache_key] = contract_expression(struct[0], *tensor_shapes, optimize=hit or 'auto')
            struct[2] = hit
        return struct[2]

    def precompute_paths(self, einsum_arr, symbols_arr, workers=None):
        # contraction paths of every structure, searched up front by a fork pool (the same deterministic opt_einsum
        # search as on first use, so the same paths): ~89k structures in MSCOCO train cost ~0.4-1 ms each serially
        todo = {}
        for e, syms in zip(einsum_arr, symbols_arr):
            key = (_batched(e), tuple(tuple(d['shape']) for d in syms))
            if key not in self.path_cache:
                todo[key] = None
        if not todo:
            return 0
        workers = workers or len(os.sched_getaffinity(0))
        with multiprocessing.get_context('fork').Pool(workers) as pool:
            found = pool.map(_search_path, list(todo), chunksize=512)
        for key, (path, plan) in zip(todo, found):
            self.path_cache[key] = path; self.level_cache[key] = plan
        return len(todo)

    def _grouped(self, groups, refs, pieces):
        rows, outs = [], []
        for (sid, items), ref in zip(groups.items(), refs):
            tensors = [pieces[b][k] for b, k in ref]
            outs.append(self._expression(self._structs[sid], [t.shape for t in tensors])(*tensors))
            rows.extend(i for i, _ in items)
        # one gather puts the group outputs back in batch order (each row is written once, so values are exact)
        inverse = torch.empty(len(rows), dtype=torch.long)
        inverse[torch.tensor(rows)] = torch.arange(len(rows))
        return torch.cat(outs, dim=0).index_select(0, inverse.to(outs[0].device))

    def _steps(self, sid, n):
        # the structure's pairwise steps as replayed by _levelwise: [(key, operand refs, output shape)], key =
        # (equation relabelled by first appearance, operand shapes without the batch index), so steps that contract
        # the same axis positions share a key; refs are ('leaf', slot) or ('tmp', earlier step). Same contraction list
        # as the grouped engine's expression. levels[d] = the steps at dependency depth d+1 (independent of each other).
        struct = self._structs[sid]
        if len(struct) > 4:
            return struct[4]
        plan = self.level_cache.get((struct[0], struct[1]))
        if plan is None:
            plan = _level_plan(struct[0], struct[1], self._expression(struct, [(n, *shape) for shape in struct[1]]).contraction_list)
        struct.append(plan)
        return struct[4]

    def _levelwise(self, groups):
        # Every tensor of the batch is a row of a pool piece: per bank, one piece with a row per use of a word tensor
        # (expanded once from the bank's unique rows), then one piece per (round, output shape) of results. Each row is
        # read exactly once. Round t runs the steps at dependency depth t+1 of every sentence (they only need earlier
        # rounds); steps with the same key run as one batched einsum,
        # and all its operands of one shape come from one gather (_Take). The schedule depends only on counts, so it
        # is planned on the CPU first and all its indices move to the GPU in one copy.
        device = self._bank(0).device
        plans = [(sid, np.array([r for _, r in items], dtype=np.int64).reshape(len(items), -1), self._steps(sid, len(items)))
                 for sid, items in groups.items()]
        chunks, pos = [], [0]
        def put(a):   # index array -> its span in the single transfer
            chunks.append(a); pos[0] += len(a); return (pos[0] - len(a), pos[0])

        # pieces[shape] = sizes of the pool pieces of that shape; a row is (piece, local row)
        piece_sizes = defaultdict(list)
        leaf_ref, leaves = {}, []
        by_bank = defaultdict(list)
        for g, (sid, mat, _) in enumerate(plans):
            for j, b in enumerate(self._structs[sid][3]):
                by_bank[b].append((g, j, mat[:, j]))
        for b, items in by_bank.items():
            uses = np.concatenate([r for _, _, r in items])
            uniq, inv = np.unique(uses, return_inverse=True)
            shape = self.bank_shapes[b]
            pid = len(piece_sizes[shape]); piece_sizes[shape].append(len(uses))
            leaves.append((b, shape, put(uniq), put(inv.reshape(-1))))
            o = 0
            for g, j, r in items:
                leaf_ref[(g, j)] = (pid, np.arange(o, o + len(r))); o += len(r)

        tmp = [[None] * len(st[0]) for _, _, st in plans]
        def ref_of(g, ref):
            return leaf_ref[(g, ref[1])] if ref[0] == 'leaf' else tmp[g][ref[1]]

        def take(shape, refs):
            # one gather of the rows refs = [(piece, local rows)] from the pieces of `shape` they touch
            used = sorted({pid for pid, _ in refs})
            off, o = {}, 0
            for pid in used:
                off[pid] = o; o += piece_sizes[shape][pid]
            return (shape, used, put(np.concatenate([off[pid] + rows for pid, rows in refs])))

        rounds = []
        for t in range(max(len(st[2]) for _, _, st in plans)):
            buckets = defaultdict(list)   # key -> [(group, step, operand refs, out shape)]
            for g, (_, _, (st, _, levels)) in enumerate(plans):
                if t < len(levels):
                    for k in levels[t]:
                        buckets[st[k][0]].append((g, k, st[k][1], st[k][2]))
            # operands grouped by shape: requests[shape] = [(bucket, position, (piece, rows) per member)]
            requests = defaultdict(list)
            for k, ((eq, shapes), members) in enumerate(buckets.items()):
                for p, shape in enumerate(shapes):
                    requests[shape].append((k, p, [ref_of(g, refs[p]) for g, _, refs, _ in members]))
            gathers = []
            for shape, reqs in requests.items():
                flat = [r for _, _, rs in reqs for r in rs]
                sizes = [sum(len(rows) for _, rows in rs) for _, _, rs in reqs]
                gathers.append((take(shape, flat), [(k, p) for k, p, _ in reqs], sizes))
            # results: one piece per output shape, buckets in order
            out_pieces, ops = {}, []
            for (eq, shapes), members in buckets.items():
                out_shape = members[0][3]
                if out_shape not in out_pieces:
                    out_pieces[out_shape] = [len(piece_sizes[out_shape]), 0]
                    piece_sizes[out_shape].append(0)
                pid, o = out_pieces[out_shape]
                for g, k, _, _ in members:
                    n = len(plans[g][1]); tmp[g][k] = (pid, np.arange(o, o + n)); o += n
                out_pieces[out_shape][1] = o
                piece_sizes[out_shape][pid] = o
                ops.append((eq, len(shapes), out_shape))
            rounds.append((gathers, ops, list(out_pieces)))
        # final outputs straight into batch order
        order = np.concatenate([[i for i, _ in items] for items in groups.values()])
        finals = [ref_of(g, st[1]) for g, (_, _, st) in enumerate(plans)]   # st = (steps, final ref, levels)
        inverse = np.empty(len(order), dtype=np.int64); inverse[order] = np.arange(len(order))
        per_row = [(pid, rows[k:k + 1]) for pid, rows in finals for k in range(len(rows))]
        final = take((self.out_dim,), [per_row[i] for i in inverse])

        idx = torch.from_numpy(np.concatenate(chunks))
        if device.type == 'cuda':
            idx = idx.pin_memory().to(device, non_blocking=True)
        sl = lambda span: idx[span[0]:span[1]]

        pieces = defaultdict(list)
        def run_take(spec):
            shape, used, span = spec
            return _Take.apply(sl(span), *[pieces[shape][pid] for pid in used])
        for b, shape, uspan, ispan in leaves:
            if torch.is_grad_enabled():   # training: RowAdam updates exactly these rows from the leaf's gradient
                leaf = self._bank(b).detach().index_select(0, sl(uspan)).requires_grad_()
                self._active.append((b, sl(uspan), leaf))
            else:
                leaf = self._bank(b).index_select(0, sl(uspan))
            pieces[shape].append(leaf.index_select(0, sl(ispan)))   # one row per use
        for gathers, ops, out_shapes in rounds:
            operands = [[None] * n for _, n, _ in ops]
            for spec, slots, sizes in gathers:
                for (k, p), part in zip(slots, run_take(spec).split(sizes)):
                    operands[k][p] = part
            results = defaultdict(list)
            for (eq, _, out_shape), ins in zip(ops, operands):
                results[out_shape].append(torch.einsum(eq, ins))
            for shape in out_shapes:   # planned order: one new piece per output shape
                rs = results[shape]
                pieces[shape].append(torch.cat(rs) if len(rs) > 1 else rs[0])
        return run_take(final)


def _level_plan(batched_str, struct_shapes, contraction_list):
    # see TNModel._steps: (steps, final ref, steps by dependency depth) of one structure
    inputs = batched_str.split('->')[0].split(',')
    dim = {c: d for sub, shape in zip(inputs, struct_shapes) for c, d in zip(sub[1:], shape)}   # sub[0] = batch index
    ops, steps = [('leaf', j) for j in range(len(inputs))], []
    for inds, _, einsum_str, _, blas_flag in contraction_list:
        if blas_flag and 'EINSUM' not in blas_flag:
            raise NotImplementedError(f"levelwise engine: tensordot step in {batched_str}")
        operands = [ops.pop(x) for x in inds]
        ins, out = einsum_str.split('->')
        shapes = tuple(tuple(dim[c] for c in sub[1:]) for sub in ins.split(','))
        steps.append(((_positional(einsum_str), shapes), operands, tuple(dim[c] for c in out[1:])))
        ops.append(('tmp', len(steps) - 1))
    assert len(ops) == 1
    depth = []
    for _, operands, _ in steps:
        depth.append(1 + max((depth[r[1]] for r in operands if r[0] == 'tmp'), default=0))
    levels = [[k for k, d in enumerate(depth) if d == level] for level in range(1, max(depth, default=0) + 1)]
    return steps, ops[0], levels

def _positional(einsum_str):
    # relabel indices by first appearance: equations that contract the same axis positions become identical
    labels = {}
    return ''.join(labels.setdefault(c, oe_parser.get_symbol(len(labels))) if c not in ',->' else c for c in einsum_str)

def _batched(einsum_expr):
    # the batch index (first unused symbol) in front of every input and of the output
    input_subscripts, output_subscript = einsum_expr.split('->')
    batch_symbol = next(oe_parser.gen_unused_symbols(einsum_expr, 1))
    return f"{','.join(batch_symbol + s for s in input_subscripts.split(','))}->{batch_symbol}{output_subscript}"

def _search_path(key):
    # path and levelwise plan of a batched structure; batch size 64 stands in for the group size (the path doesn't
    # depend on it)
    batched_str, shapes = key
    path, info = contract_path(batched_str, *[(64, *shape) for shape in shapes], shapes=True)
    return path, _level_plan(batched_str, shapes, info.contraction_list)

class _Take(torch.autograd.Function):
    # rows idx of the concatenated pieces. Every pool row is read once, so idx has no repeats and the backward is one
    # index_copy into zeros (no accumulation; deterministic) instead of index_select's index_add
    @staticmethod
    def forward(ctx, idx, *pieces):
        ctx.save_for_backward(idx)
        ctx.sizes = [p.shape[0] for p in pieces]
        src = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        return src.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad):
        idx, = ctx.saved_tensors
        full = grad.new_zeros((sum(ctx.sizes), *grad.shape[1:])).index_copy_(0, idx, grad)
        return (None, *full.split(ctx.sizes))
