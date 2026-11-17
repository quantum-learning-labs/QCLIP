from collections import defaultdict
from typing import List, Tuple
import torch, math
import torch.nn as nn
from opt_einsum import contract_expression
from opt_einsum import parser as oe_parser
from tqdm import tqdm

class TNModel(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.symbols: List[str] = []
        self.shapes: List[Tuple[int, ...]] = []
        self.params = nn.ParameterDict()
        self.path_cache = {}
        self.out_dim: int = out_dim
    
    @property
    def pcount(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

    def _flatten_symbols(self, val):
        if isinstance(val, dict) and 'name' in val:
            return [val]
        elif isinstance(val, list):
            res = []
            for item in val:
                res.extend(self._flatten_symbols(item))
            return res
        return []

    def init_params(self):
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through ')
        
        for symbol, shape in zip(self.symbols, self.shapes):
            self.params[symbol] = self._init_tensor(shape)

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

    def forward(self, batch: List[tuple[str, List[str]]]) -> torch.Tensor:
        groups = defaultdict(list)
        for i, (einsum_expr, tensor_arr) in enumerate(batch):
            shapes, _ = zip(*[(tuple(sym_dict['shape']), sym_dict['name']) for sym_dict in tensor_arr])
            groups[(einsum_expr, shapes)].append((i, tensor_arr))

        output = torch.zeros(len(batch), self.out_dim, device=next(self.params.parameters()).device)
        for (einsum_expr, _), items in groups.items():
            indices, symbols_arr =  zip(*items)
            symbol_columns = zip(*symbols_arr)
            shapes, tensors = [], []
            for col in symbol_columns:
                tensor_batch = torch.stack([self.params[sym_dict['name']] for sym_dict in col], dim=0)
                shapes.append(tensor_batch.shape)
                tensors.append(tensor_batch)
            
            input_subscripts, output_subscript = einsum_expr.split('->')
            batch_symbol = next(oe_parser.gen_unused_symbols(einsum_expr, 1))
            batched_inputs = [batch_symbol + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->{batch_symbol}{output_subscript}"

            cache_key = (batched_str, tuple(shapes))
            if cache_key not in self.path_cache:
                self.path_cache[cache_key] = contract_expression(batched_str, *shapes)
            output[list(indices)] = self.path_cache[cache_key](*tensors)
        
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)