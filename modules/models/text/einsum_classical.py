from collections import defaultdict, Counter
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

    def __init__(self, out_dim: int = 512, engine: str = 'grouped', init: str = 'glorot', scale_free: bool = False, bow: bool = False,
                 head: str = 'affine', hybrid: bool = False, word_base: float = None, centre: bool = False,
                 mix_fixed: float = None, eval_chunk: int = None):
        super().__init__()
        # scale_free (text.scale_free, levelwise only): divide every intermediate of the contraction by its (detached) max |entry|,
        # per sentence. The network is multilinear, so this only rescales each sentence vector by a positive number, which
        # the LayerNorm head removes: same outputs and gradients (up to the LayerNorm eps), no fp32 overflow/underflow
        # however large the products of the tensors' gains grow (step 4: they reach inf and the run diverges)
        if scale_free and engine != 'levelwise':
            raise ValueError("text.scale_free needs text.tn_engine: levelwise")
        self.scale_free = scale_free
        # bow (text.arch: bow, step-4b lexical baseline): the caption's word vectors ('w:' symbols of the decoupled compile)
        # averaged, then a linear map word_dim -> out_dim (bow_proj) and the same head; the type tensors are never used
        self.bow, self.bow_proj = bow, None
        # step 4c (composition that keeps recall; levelwise only). hybrid (text.compose: hybrid): sentence = unit(TN) +
        # mix * unit(bow_proj(mean of the word vectors)), mix learned (init 1): a lexical path beside the TN.
        # word_base (text.word_base: alpha): word vector = base + d_w, base one shared learned vector, the bank rows hold
        # d_w (init scaled by alpha): around base the multilinear TN is to first order additive in the words, each word
        # still entering through its own leg. centre (text.centre, needs word_base): subtract the same network with every
        # word set to base (the caption's 'empty' structure term), so captions are compared by their word contributions
        if (hybrid or word_base is not None or centre) and engine != 'levelwise':
            raise ValueError("text.compose: hybrid / text.word_base / text.centre need text.tn_engine: levelwise")
        if centre and word_base is None:
            raise ValueError("text.centre needs text.word_base")
        self.hybrid, self.word_scale, self.centre = hybrid, word_base, centre
        # mix_fixed (text.mix_fixed, hybrid only): mix held at this value, not learned (step 4c: a learned mix grows 1 -> 5.9
        # and the lexical path drowns the TN)
        if mix_fixed is not None and not hybrid:
            raise ValueError("text.mix_fixed needs text.compose: hybrid")
        self.mix_fixed = mix_fixed
        # eval_chunk (text.eval_chunk, default off): no-grad forwards (validation/benchmarks encode ~2.5k captions at once) run
        # in chunks of at most this many captions; rows are independent, so only peak memory changes (wide bonds OOM'd 24 GB)
        self.eval_chunk = eval_chunk
        self.extra = None   # nn.ParameterDict of the step-4c parameters ('word_base', 'mix'), built with the banks
        self.capture = None   # a list: every forward appends its pre-LayerNorm outputs (analysis tools)
        # init 'glorot': step-1 scheme (shape only). 'norm': std 1/sqrt(dim of the leg towards the sentence output) (the
        # compiler's 'out' per symbol, majority over its uses), so each tensor maps unit-norm inputs to a unit-norm
        # output in expectation and sentence vectors start at norm ~1 (glorot: ~1e-6, decoupled ~1e-11: below the
        # LayerNorm eps, gradients vanish)
        if init not in ('glorot', 'norm'):
            raise ValueError(f"text.init must be 'glorot' or 'norm', got {init!r}")
        self.init = init
        self.out_votes = defaultdict(Counter)   # symbol -> Counter of its output leg over its uses (init 'norm')
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
        # head (text.head): 'affine' = LayerNorm with gain and bias (default); 'no_bias' = gain only; 'plain' = no parameters
        # (step 4b: the shared gain/bias move first and collapse the embeddings)
        if head not in ('affine', 'no_bias', 'plain'):
            raise ValueError(f"text.head must be 'affine', 'no_bias' or 'plain', got {head!r}")
        self.norm_head = nn.LayerNorm(out_dim) if head == 'affine' else nn.LayerNorm(out_dim, elementwise_affine=head == 'no_bias', bias=False)
        self.stats = None   # tn_stats: a list; training forwards append (min, median, max, #non-finite) of the pre-LayerNorm norms
    
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
                if self.init == 'norm' and symbol_info.get('out') is not None:
                    self.out_votes[symbol][symbol_info['out']] += 1
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
    def _init_tensor(shape, generator=None, out=None, std=None):
        # out: fill this tensor (a bank row) in place; the draws are the same as for a fresh tensor of that shape
        # std: init 'norm' (see __init__); None = the glorot scheme
        param = nn.Parameter(torch.empty(shape)) if out is None else out
        order = len(shape)
        if std is not None and order > 0:
            nn.init.normal_(param, mean=0.0, std=std, generator=generator)
        elif order == 0:
            nn.init.constant_(param, 1.0)
        elif order == 1:
            nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(shape[0]), generator=generator)
        else:
            fan_out = shape[0]
            fan_in = math.prod(shape[1:])
            std = math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(param, mean=0.0, std=std, generator=generator)
        return param

    def _norm_std(self, name, shape):
        # init 'norm': 1/sqrt(dim of the output leg); vectors are always outputs (1/sqrt(d), as glorot); a symbol never
        # seen with an orientation (e.g. a disconnected tensor) uses its first leg
        if self.init != 'norm' or len(shape) == 0:
            return None
        votes = self.out_votes.get(name)
        leg = votes.most_common(1)[0][0] if votes else 0
        return 1.0 / math.sqrt(shape[leg])

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
                    self._init_tensor(shape, out=banks[b][r], std=self._norm_std(name, shape))
        for b, t in enumerate(banks):
            self.register_parameter(f'bank{b}', nn.Parameter(t))
        if (self.bow or self.hybrid) and self.bow_proj is None:
            self.bow_proj = nn.Linear(next(s[0] for s in self.bank_shapes if len(s) == 1), self.out_dim, bias=False)
        if self.word_scale is not None:
            wbanks = {self.sym_loc[n][0] for n in self.symbols if n.startswith('w:')}
            if len(wbanks) != 1 or any(not n.startswith('w:') for n in self.symbols if self.sym_loc[n][0] in wbanks):
                raise ValueError("text.word_base needs the decoupled compile: all word vectors ('w:') in one bank of their own")
            self._word_bank = wbanks.pop()
        if (self.word_scale is not None or self.hybrid) and self.extra is None:
            extra = {}
            if self.word_scale is not None:
                dim = self.bank_shapes[self._word_bank][0]
                if init:
                    with torch.no_grad():
                        self._bank(self._word_bank).mul_(self.word_scale)
                extra['word_base'] = nn.Parameter(torch.randn(dim) / math.sqrt(dim) if init else torch.empty(dim))
            if self.hybrid:
                extra['mix'] = (nn.Parameter(torch.ones(())) if self.mix_fixed is None
                                else nn.Parameter(torch.tensor(float(self.mix_fixed)), requires_grad=False))
            self.extra = nn.ParameterDict(extra)

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
                t = self._init_tensor(shape, gen, std=self._norm_std(name, shape)).data
                rows[b].append(t * self.word_scale if self.word_scale is not None and name.startswith('w:') else t)
            for b, ts in rows.items():
                self.register_parameter(f'bank{b}', nn.Parameter(torch.cat([self._bank(b).data, torch.stack(ts).to(device)])))
        return {"oov_sentence_frac": n_oov_sent / max(n_sent, 1), "oov_tensor_frac": n_oov_tensors / max(n_tensors, 1),
                "oov_new_symbols": len(new)}

    def _record(self, output):
        if self.capture is not None:
            self.capture.append(output.detach())
        if self.stats is not None and torch.is_grad_enabled():
            n = output.detach().norm(dim=1)
            self.stats.append(torch.stack([n.min(), n.median(), n.max(), (~torch.isfinite(n)).sum().to(n.dtype)]))

    @torch.no_grad()
    def param_stats(self):
        # tn_stats: Frobenius norms of the word vectors (order-1 banks) and of the type tensors / MPS cores (the other banks):
        # quantiles, and quantiles of each tensor's norm over its norm at the first call (the init, epoch -1)
        norms = {'word': [self._bank(b).norm(dim=1) for b, s in enumerate(self.bank_shapes) if len(s) == 1],
                 'type': [self._bank(b).flatten(1).norm(dim=1) for b, s in enumerate(self.bank_shapes) if len(s) > 1]}
        norms = {k: torch.cat(v).float() for k, v in norms.items() if v}
        if not hasattr(self, '_norm0'):
            self._norm0 = {k: v.clone() for k, v in norms.items()}
        out, qs = {}, torch.tensor([0.0, 0.5, 1.0])
        for k, v in norms.items():
            for name, x in ((k, v), (k + '_rel', v / self._norm0[k].clamp_min(1e-30))):
                for q, val in zip(('min', 'med', 'max'), torch.quantile(x.cpu(), qs).tolist()):
                    out[f'{name}_{q}'] = val
        out['head_w'] = float(self.norm_head.weight.norm()) if self.norm_head.weight is not None else float('nan')
        if self.extra is not None:
            for k, v in self.extra.items():
                out[k] = float(v.norm())
        return out

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
        if self.eval_chunk and not torch.is_grad_enabled() and len(batch) > self.eval_chunk:
            return torch.cat([self.forward(batch[j:j + self.eval_chunk]) for j in range(0, len(batch), self.eval_chunk)])
        if self.bow:
            return self._bow_forward(batch)
        if not hasattr(self, '_plans'):
            self._plans, self._struct_ids, self._structs = {}, {}, []
        groups = defaultdict(list)   # structure id -> [(row, bank rows)], in order of first appearance (as before)
        for i, (einsum_expr, tensor_arr) in enumerate(batch):
            sid, rows = self._plan(einsum_expr, tensor_arr)
            groups[sid].append((i, rows))

        if self.engine == 'levelwise':
            output = self._centred(groups) if self.centre else self._levelwise(groups)
            if self.hybrid:
                output = (torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)
                          + self.extra['mix'] * torch.nn.functional.normalize(self._bow_vec(batch), p=2, dim=-1, eps=1e-12))
            self._record(output)
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
        self._record(output)
        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)

    def _bow_forward(self, batch):
        output = self._bow_vec(batch)
        self._record(output)
        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)

    def _bow_vec(self, batch):
        # bow_proj of the mean word vector of each caption (pre-head)
        rows, seg, b = [], [], None
        for i, (_, tensor_arr) in enumerate(batch):
            for d in tensor_arr:
                if d['name'].startswith('w:'):
                    b, r = self.sym_loc[d['name']]; rows.append(r); seg.append(i)
        device = self._bank(0).device
        if b is None:
            output = torch.zeros(len(batch), self.out_dim, device=device)
        else:
            uniq, inv = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
            uniq = torch.from_numpy(uniq).to(device)
            if torch.is_grad_enabled():   # training: RowAdam updates exactly these word rows
                leaf = self._bank(b).detach().index_select(0, uniq).requires_grad_()
                self._active.append((b, uniq, leaf))
            else:
                leaf = self._bank(b).index_select(0, uniq)
            vecs = leaf.index_select(0, torch.from_numpy(inv).to(device))
            if self.word_scale is not None:
                vecs = vecs + self.extra['word_base']
            seg = torch.tensor(seg, device=device)
            summed = torch.zeros(len(batch), vecs.shape[1], device=device).index_add_(0, seg, vecs)
            output = self.bow_proj(summed / torch.bincount(seg, minlength=len(batch)).clamp_min(1).unsqueeze(1))
        return output

    def _centred(self, groups):
        # text.centre: TN(caption) - TN(same caption, every word = base), computed as two levelwise passes. The structure
        # term depends on the caption's type tensors, not just its structure: one base pass per distinct (structure,
        # type-tensor rows) of the batch. Under scale_free both passes track their per-sentence log scale, so the
        # difference is exact up to one positive factor per sentence; then unit norm (the difference can be small next
        # to the LayerNorm eps)
        n = sum(len(items) for items in groups.values())
        wb, keyed = self._word_bank, {}
        base_groups, to_base = defaultdict(list), np.empty(n, dtype=np.int64)
        for sid, items in groups.items():
            types = [j for j, b in enumerate(self._structs[sid][3]) if b != wb]
            for i, rows in items:
                key = (sid, tuple(rows[j] for j in types))
                k = keyed.get(key)
                if k is None:
                    k = keyed[key] = len(keyed)
                    base_groups[sid].append((k, rows))
                to_base[i] = k
        # one levelwise run for both passes (base rows after the real ones): same steps, half the kernel launches
        out, scale = self._levelwise(groups, track=True, base_groups=base_groups)
        real, ls, base, bls = out[:n], scale[:n], out[n:], scale[n:]
        idx = torch.from_numpy(to_base).to(real.device)
        bls = bls.index_select(0, idx); top = torch.maximum(ls, bls)   # both factors <= 1: no overflow
        diff = real * (ls - top).exp().unsqueeze(1) - base.index_select(0, idx) * (bls - top).exp().unsqueeze(1)
        return torch.nn.functional.normalize(diff, p=2, dim=-1, eps=1e-30)

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
        if len(todo) < 4000:   # forking the (large) training process costs ~1 s: small sets are searched in-process
            found = [_search_path(key) for key in todo]
        else:
            # TN_PATH_WORKERS caps the pool: on a cluster node the affinity is the whole machine and each fork counts
            # against the job's h_vmem (a 64G job died silently here)
            workers = workers or int(os.environ.get('TN_PATH_WORKERS', 0)) or len(os.sched_getaffinity(0))
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

    def _schedule(self, groups, base_groups=None):
        # The levelwise schedule of a batch, planned on the CPU (it depends only on counts): one int64 index array (all
        # gathers, moved to the GPU in one copy) and specs of spans into it. Pool pieces per shape: one per bank holding only
        # the bank's UNIQUE rows of the batch (step 5: no per-use copy; a use is a virtual row mapped to its unique row), then
        # one piece per (round, output shape) of results. Round t runs the steps at dependency depth t+1 of every sentence;
        # steps with the same key run as one batched einsum, and all its operands of one shape come from one gather (_Take;
        # with repeats when it reads leaf rows). An operand slot whose members all read the same single leaf row (e.g. the
        # shared output core, a dense type with one row, the centring base vector) is passed unbatched instead (broadcast).
        # A ref is (piece, first virtual row, count). Each piece is dropped after its last read (forward memory).
        # base_groups (text.centre): extra groups whose word slots are all word_base (pseudo bank 'base', one row); their
        # outputs follow the real ones (row len(batch) + k for base key k)
        entries = [(sid, items, False) for sid, items in groups.items()]
        if base_groups:
            entries += [(sid, items, True) for sid, items in base_groups.items()]
        plans = [(sid, np.array([r for _, r in items], dtype=np.int64).reshape(len(items), -1), self._steps(sid, len(items)))
                 for sid, items, _ in entries]
        chunks, pos = [], [0]
        def put(a):   # index array -> its span in the single transfer
            chunks.append(a); pos[0] += len(a); return (pos[0] - len(a), pos[0])

        piece_sizes = defaultdict(list)   # real rows per piece
        vmap = defaultdict(list)          # leaf pieces: virtual (per-use) row -> unique row; None for result pieces
        leaf_ref, leaves = {}, []
        by_bank = defaultdict(list)
        wb = self._word_bank if self.word_scale is not None else None
        for g, (sid, mat, _) in enumerate(plans):
            base = entries[g][2]
            for j, b in enumerate(self._structs[sid][3]):
                by_bank['base' if base and b == wb else b].append((g, j, mat[:, j]))
        for b, items in by_bank.items():
            uses = np.concatenate([r for _, _, r in items])
            if b == 'base':
                uniq, inv = np.zeros(1, dtype=np.int64), np.zeros(len(uses), dtype=np.int64)
            else:
                uniq, inv = np.unique(uses, return_inverse=True)
            shape = self.bank_shapes[wb if b == 'base' else b]
            pid = len(piece_sizes[shape]); piece_sizes[shape].append(len(uniq)); vmap[shape].append(inv.reshape(-1))
            leaves.append((b, shape, put(uniq)))
            o = 0
            for g, j, r in items:
                leaf_ref[(g, j)] = (pid, o, len(r)); o += len(r)

        tmp = [[None] * len(st[0]) for _, _, st in plans]
        last_use = {}   # (shape, pid) -> index of the last round that reads it (len(rounds) = the final gather)
        def ref_of(g, ref):
            return leaf_ref[(g, ref[1])] if ref[0] == 'leaf' else tmp[g][ref[1]]

        def gather_index(shape, refs, t):
            # index into the concatenation of the used pieces for virtual row ranges refs; repeats: reads leaf rows
            used = sorted({pid for pid, _, _ in refs})
            vstart = np.array([s for _, s, _ in refs], dtype=np.int64); cnt = np.array([n for _, _, n in refs], dtype=np.int64)
            for pid in used:
                last_use[(shape, pid)] = t
            if all(vmap[shape][pid] is None for pid in used):   # result pieces only: virtual = real rows (as before)
                off, o = {}, 0
                for pid in used:
                    off[pid] = o; o += piece_sizes[shape][pid]
                return used, _ranges(np.array([off[pid] for pid, _, _ in refs], dtype=np.int64) + vstart, cnt), False
            table, voff, roff, vo = [], {}, 0, 0
            for pid in used:
                m = vmap[shape][pid]
                table.append(roff + (m if m is not None else np.arange(piece_sizes[shape][pid], dtype=np.int64)))
                voff[pid] = vo; vo += len(m) if m is not None else piece_sizes[shape][pid]; roff += piece_sizes[shape][pid]
            table = np.concatenate(table)
            return used, table[_ranges(np.array([voff[pid] for pid, _, _ in refs], dtype=np.int64) + vstart, cnt)], True

        def single_leaf_row(shape, refs):
            # the one unique leaf row every ref reads, or None
            pids = {pid for pid, _, _ in refs}
            if len(pids) != 1:
                return None
            pid = next(iter(pids)); m = vmap[shape][pid]
            if m is None:
                return None
            rows = np.concatenate([m[s:s + n] for _, s, n in refs])
            return (pid, int(rows[0])) if (rows == rows[0]).all() else None

        rounds = []
        for t in range(max(len(st[2]) for _, _, st in plans)):
            buckets = defaultdict(list)   # key -> [(group, step, operand refs, out shape)]
            for g, (_, _, (st, _, levels)) in enumerate(plans):
                if t < len(levels):
                    for k in levels[t]:
                        buckets[st[k][0]].append((g, k, st[k][1], st[k][2]))
            requests = defaultdict(list)   # shape -> [(bucket, position, refs per member)]
            ops = []
            for k, ((eq, shapes), members) in enumerate(buckets.items()):
                slot_refs = [[ref_of(g, refs[p]) for g, _, refs, _ in members] for p in range(len(shapes))]
                single = [single_leaf_row(shape, rs) for shape, rs in zip(shapes, slot_refs)]
                if all(s is not None for s in single):   # keep one batched operand so the result has the batch index
                    single[max(range(len(shapes)), key=lambda p: len(shapes[p]))] = None
                ins, out = eq.split('->'); ins = ins.split(',')
                bcast = []
                for p, (shape, rs) in enumerate(zip(shapes, slot_refs)):
                    if single[p] is not None:
                        pid, row = single[p]; bcast.append((p, shape, pid, row)); ins[p] = ins[p][1:]
                        last_use[(shape, pid)] = t
                    else:
                        requests[shape].append((k, p, rs))
                ops.append((','.join(ins) + '->' + out, len(shapes), members[0][3], bcast))
            gathers = []
            for shape, reqs in requests.items():
                flat = [r for _, _, rs in reqs for r in rs]
                sizes = [sum(n for _, _, n in rs) for _, _, rs in reqs]
                used, ix, rep = gather_index(shape, flat, t)
                gathers.append(((shape, used, put(ix), rep), [(k, p) for k, p, _ in reqs], sizes))
            # results: one piece per output shape, buckets in order
            out_pieces = {}
            for (eq, shapes), members in buckets.items():
                out_shape = members[0][3]
                if out_shape not in out_pieces:
                    out_pieces[out_shape] = [len(piece_sizes[out_shape]), 0]
                    piece_sizes[out_shape].append(0); vmap[out_shape].append(None)
                pid, o = out_pieces[out_shape]
                for g, k, _, _ in members:
                    n = len(plans[g][1]); tmp[g][k] = (pid, o, n); o += n
                out_pieces[out_shape][1] = o
                piece_sizes[out_shape][pid] = o
            rounds.append([gathers, ops, list(out_pieces), None])
        # final outputs straight into batch order: the groups' output rows in plan order, then permuted to batch order
        nreal = sum(len(items) for items in groups.values())
        order = np.concatenate([[i + (nreal if base else 0) for i, _ in items] for _, items, base in entries])
        finals = [ref_of(g, st[1]) for g, (_, _, st) in enumerate(plans)]   # st = (steps, final ref, levels)
        inverse = np.empty(len(order), dtype=np.int64); inverse[order] = np.arange(len(order))
        shape = (self.out_dim,)
        used, rows, rep = gather_index(shape, finals, len(rounds))
        final = (shape, used, put(rows[inverse]), rep)
        for t, r in enumerate(rounds):   # pieces whose last read is in round t: dropped after its gathers
            r[3] = [key for key, lt in last_use.items() if lt == t]
        return np.concatenate(chunks), leaves, rounds, final

    def _levelwise(self, groups, schedule=None, track=False, base_only=False, base_groups=None):
        # runs the batch's schedule (_schedule; `schedule` = one planned ahead of time). word_base: word rows are base +
        # bank row; base_only: every word is base (text.centre's structure term). track: also return each sentence's log
        # scale (sum of the scale_free divisors along its contraction; 0 without scale_free): true output = out * exp(.)
        device = self._bank(0).device
        idx, leaves, rounds, final = schedule if schedule is not None else self._schedule(groups, base_groups)
        idx = torch.from_numpy(idx)
        if device.type == 'cuda':
            idx = idx.pin_memory().to(device, non_blocking=True)
        sl = lambda span: idx[span[0]:span[1]]

        pieces, scales = defaultdict(list), defaultdict(list)
        def run_take(spec):
            shape, used, span, rep = spec
            return _Take.apply(sl(span), rep, *[pieces[shape][pid] for pid in used])
        def take_scale(spec):
            shape, used, span, _ = spec
            src = [scales[shape][pid] for pid in used]
            return (torch.cat(src) if len(src) > 1 else src[0]).index_select(0, sl(span))
        wb = self._word_bank if self.word_scale is not None else None
        for b, shape, uspan in leaves:
            if b == 'base':
                piece = self.extra['word_base'].unsqueeze(0)
            elif base_only and b == wb:
                piece = self.extra['word_base'].expand(uspan[1] - uspan[0], -1)
            else:
                if torch.is_grad_enabled():   # training: RowAdam updates exactly these rows from the leaf's gradient
                    leaf = self._bank(b).detach().index_select(0, sl(uspan)).requires_grad_()
                    self._active.append((b, sl(uspan), leaf))
                else:
                    leaf = self._bank(b).index_select(0, sl(uspan))
                piece = leaf + self.extra['word_base'] if b == wb else leaf   # unique rows only
            pieces[shape].append(piece)
            if track:
                scales[shape].append(piece.new_zeros(piece.shape[0]))
        for gathers, ops, out_shapes, drop in rounds:
            operands = [[None] * n for _, n, _, _ in ops]
            op_scales = [[None] * n for _, n, _, _ in ops] if track else None
            for k, (_, _, _, bcast) in enumerate(ops):
                for p, shape, pid, row in bcast:   # unbatched operand, no scale (leaf rows have log scale 0)
                    operands[k][p] = pieces[shape][pid][row]
            for spec, slots, sizes in gathers:
                for (k, p), part in zip(slots, run_take(spec).split(sizes)):
                    operands[k][p] = part
                if track:
                    for (k, p), part in zip(slots, take_scale(spec).split(sizes)):
                        op_scales[k][p] = part
            for shape, pid in drop:   # last read done: free the piece (autograd keeps what it needs)
                pieces[shape][pid] = None
                if track:
                    scales[shape][pid] = None
            results, res_scales = defaultdict(list), defaultdict(list)
            for j, ((eq, _, out_shape, _), ins) in enumerate(zip(ops, operands)):
                r = torch.einsum(eq, ins)
                if self.scale_free:
                    m = r.detach().abs().flatten(1).amax(dim=1).clamp_min(1e-30)
                    r = r / m.view(-1, *([1] * (r.dim() - 1)))
                results[out_shape].append(r)
                if track:
                    s = sum(x for x in op_scales[j] if x is not None)
                    res_scales[out_shape].append(s + m.log() if self.scale_free else s)
            for shape in out_shapes:   # planned order: one new piece per output shape
                rs = results[shape]
                pieces[shape].append(torch.cat(rs) if len(rs) > 1 else rs[0])
                if track:
                    ss = res_scales[shape]
                    scales[shape].append(torch.cat(ss) if len(ss) > 1 else ss[0])
        if track:
            return run_take(final), take_scale(final)
        return run_take(final)

def _ranges(starts, counts):
    # concatenation of arange(s, s + n) for each (s, n), in one vectorised step
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    excl = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.int64) + np.repeat(starts - excl, counts)


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
    # rows idx of the concatenated pieces. Without repeats (result pieces: every row is read once) the backward is one
    # index_copy into zeros (no accumulation; deterministic); with repeats (leaf pieces hold unique rows, read once per use)
    # it is an index_add
    @staticmethod
    def forward(ctx, idx, repeats, *pieces):
        ctx.save_for_backward(idx)
        ctx.sizes, ctx.repeats = [p.shape[0] for p in pieces], repeats
        src = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        return src.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad):
        idx, = ctx.saved_tensors
        full = grad.new_zeros((sum(ctx.sizes), *grad.shape[1:]))
        full = full.index_add_(0, idx, grad) if ctx.repeats else full.index_copy_(0, idx, grad)
        return (None, None, *full.split(ctx.sizes))
