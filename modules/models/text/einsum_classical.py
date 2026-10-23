from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch, math
import torch.nn as nn
from opt_einsum import contract_expression
from opt_einsum import parser as oe_parser
from tqdm import tqdm
import os, zlib

class TNModel(nn.Module):
    real_scores = True   # real embeddings: image-text scores are plain dot products (no |.| readout)

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.symbols: List[str] = []
        self.shapes: List[Tuple[int, ...]] = []
        self.params = nn.ParameterDict()
        # self.params = nn.ParameterList([])
        # self.sym2param: Dict[str, nn.Parameter] = nn.ParameterDict({})
        self.path_cache = {}
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
                name = key[len('params.'):]
                self.symbols.append(name); self.shapes.append(tuple(val.shape))
                self.params[name] = nn.Parameter(val)

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
    def _init_tensor(shape, generator=None):
        param = nn.Parameter(torch.empty(shape))
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
        for symbol, shape in zip(self.symbols, self.shapes):
            self.params[symbol] = self._init_tensor(shape)

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
            device = next(self.params.parameters()).device
            for name in sorted(new):
                self.symbols.append(name); self.shapes.append(new[name])
                gen = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
                self.params[name] = nn.Parameter(self._init_tensor(new[name], gen).data.to(device))
        return {"oov_sentence_frac": n_oov_sent / max(n_sent, 1), "oov_tensor_frac": n_oov_tensors / max(n_tensors, 1),
                "oov_new_symbols": len(new)}

    def param_breakdown(self):
        # parameter count by tensor order (vectors, matrices, order-3 cores, ...)
        out = defaultdict(int)
        for p in self.params.values():
            out[f"order{p.dim()}"] += p.numel()
        return dict(out)

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
        # never reused): structure id (einsum + shapes) and the symbol names in tensor order
        hit = self._plans.get(id(tensor_arr))
        if hit is not None:
            return hit[1], hit[2]
        key = (einsum_expr, tuple(tuple(d['shape']) for d in tensor_arr))
        sid = self._struct_ids.get(key)
        if sid is None:
            sid = self._struct_ids[key] = len(self._structs)
            input_subscripts, output_subscript = einsum_expr.split('->')
            batch_symbol = next(oe_parser.gen_unused_symbols(einsum_expr, 1))
            batched_inputs = [batch_symbol + s for s in input_subscripts.split(',')]
            self._structs.append([f"{','.join(batched_inputs)}->{batch_symbol}{output_subscript}", key[1], None])
        names = tuple(d['name'] for d in tensor_arr)
        self._plans[id(tensor_arr)] = (tensor_arr, sid, names)
        return sid, names

    def forward(self, batch: List[tuple[str, List[str]]]) -> torch.Tensor:
        if not hasattr(self, '_plans'):
            self._plans, self._struct_ids, self._structs = {}, {}, []
        groups = defaultdict(list)   # structure id -> [(row, names)], in order of first appearance (as before)
        for i, (einsum_expr, tensor_arr) in enumerate(batch):
            sid, names = self._plan(einsum_expr, tensor_arr)
            groups[sid].append((i, names))

        rows, outs = [], []
        for sid, items in groups.items():
            indices, names_arr = zip(*items)
            tensors = [torch.stack([self.params[name] for name in col], dim=0) for col in zip(*names_arr)]
            struct = self._structs[sid]
            if struct[2] is None:
                # path cached without the group size: the batch index is on every tensor and the output, so it scales
                # every pairwise cost equally and the contraction path doesn't depend on it
                cache_key = (struct[0], struct[1])
                if cache_key not in self.path_cache:
                    self.path_cache[cache_key] = contract_expression(struct[0], *[t.shape for t in tensors])
                struct[2] = self.path_cache[cache_key]
            outs.append(struct[2](*tensors))
            rows.extend(indices)
        # one gather puts the group outputs back in batch order (each row is written once, so values are exact)
        inverse = torch.empty(len(rows), dtype=torch.long)
        inverse[torch.tensor(rows)] = torch.arange(len(rows))
        output = torch.cat(outs, dim=0).index_select(0, inverse.to(outs[0].device))

        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)
