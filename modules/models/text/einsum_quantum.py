from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from opt_einsum import contract_expression
from collections import defaultdict
from modules.compilation.quantum.gates import *
from modules.utils.general import get_device

class VQCModel(nn.Module):
    def __init__(self, out_q: int = 9, discard=False, precision = torch.complex64, renorm=False):
        super().__init__()
        self.symbols: List[str] = []
        self.params = nn.ParameterList([])
        self.path_cache = {}
        self.plan_cache = {}
        self.sym2param: Dict[str, nn.Parameter] = {}
        self.out_q: int = out_q
        self.precision: torch.dtype = precision
        self.discard = discard
        self.renorm = renorm  # unit-trace density matrices (hybrid circuits post-select some wires)

        h_gate = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=precision) / (2.0**0.5)
        cx_gate = torch.block_diag(
            torch.eye(2, dtype=precision), 
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=precision)
        ).reshape(2,2,2,2)
        zero_gate = torch.tensor([1, 0], dtype=precision)

        self.register_buffer('gate_H', h_gate, persistent=False)
        self.register_buffer('gate_CX', cx_gate, persistent=False)
        self.register_buffer('gate_0', zero_gate, persistent=False)
        self.register_buffer('gate_0_dag', zero_gate, persistent=False)
        self.register_buffer('gate_bell', torch.eye(2, dtype=precision) / (2.0**0.5), persistent=False)   # (|00>+|11>)/sqrt2

    def _cswap_gate(self, m):
        # text.phi_block: on (m flag wires, s, a), identity if all flags are |0>, else SWAP(s, a). The matrix is a symmetric
        # permutation, so its tensor reads the same with inputs or outputs first.
        p0 = torch.zeros(2**m, 2**m, dtype=self.precision); p0[0, 0] = 1
        swap = torch.eye(4, dtype=self.precision)[[0, 2, 1, 3]]
        M = torch.kron(p0, torch.eye(4, dtype=self.precision)) + torch.kron(torch.eye(2**m, dtype=self.precision) - p0, swap)
        return M.reshape(*[2] * (2 * (m + 2))).to(self.gate_0.device)

    @property
    def grad_norm(self):
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def from_symbols(self, tn_arr, id_init=False):
        symbols_hash = set(self.symbols)

        for tensor_arr in tqdm(tn_arr):
            flat_tensors = self._flatten_symbols(tensor_arr)
            for symbol_info in flat_tensors:
                symbol = symbol_info['name']
                if (symbol is not None) and (symbol not in symbols_hash):
                    symbols_hash.add(symbol)
                    self.symbols.append(symbol)
        self.init_params(id_init=id_init)

    def _flatten_symbols(self, val):
        if isinstance(val, dict) and 'name' in val:
            return [val]
        elif isinstance(val, list):
            res = []
            for item in val:
                res.extend(self._flatten_symbols(item))
            return res
        return []

    def init_params(self, id_init=False) -> None:
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through ')

        num_symbols = len(self.symbols)
        param_values = torch.empty(num_symbols)

        if id_init:
            param_values.uniform_(-0.01, 0.01)
        else:
            param_values.uniform_(-torch.pi/2, torch.pi/2)
        self.params = nn.ParameterList([nn.Parameter(param_values[i:i+1]) for i in range(num_symbols)])
        self.sym2param = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.unk_param_index = len(self.params)
        self.params.append(nn.Parameter(torch.zeros(1), requires_grad=False))

    # def _get_params(self):
    #     return {sym: float(self.params[idx].detach().cpu().item()) for sym, idx in self.sym2param.items()}

    def _get_params(self):
        with torch.no_grad():
            raw_tensor = torch.cat([p for p in self.params])
            thetas = torch.tanh(raw_tensor) * 2 * torch.pi
            
            return {
                sym: float(thetas[idx].detach().cpu().item()) 
                for sym, idx in self.sym2param.items()
            }

    def compile_batch(self, batch_recipes):
        groups = defaultdict(list)
        for einsum_str, tensors in batch_recipes:
            groups[einsum_str].append(tensors)

        for einsum_str, tensor_lists in groups.items():
            minibatch_size = len(tensor_lists)
            gate_columns = zip(*tensor_lists)
            shapes = []

            for column in gate_columns:
                symbol, op_type = column[0]['name'], column[0]['op_type']
                if symbol is None:
                    if op_type == '0': shapes.append(torch.Size([minibatch_size, 2]))
                    elif op_type == 'H': shapes.append(torch.Size([minibatch_size, 2, 2]))
                    elif op_type == 'CX': shapes.append(torch.Size([minibatch_size, 2, 2, 2, 2]))
                    data = data.unsqueeze(0).expand(minibatch_size, *data.shape)
                    shapes.append(data.shape)
                else:
                    if op_type == 'Rz' or op_type == 'Rx' or op_type == 'Ry':
                        shapes.append(torch.Size([minibatch_size, 2, 2]))
                    elif op_type == 'CRz' or op_type == 'CRx' or op_type == 'CRy': 
                        shapes.append(torch.Size([minibatch_size, 2, 2, 2, 2]))

            input_subscripts, output_subscript = einsum_str.split('->')
            batched_inputs = ["$" + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->${output_subscript}"
            cache_key = (batched_str, tuple(shapes))
            if cache_key not in self.path_cache: 
                self.path_cache[cache_key] = contract_expression(batched_str, *shapes)

    GATE_FNS = {'Rz': Rz, 'Rx': Rx, 'Ry': Ry, 'CRz': CRz, 'CRx': CRx, 'CRy': CRy}

    def _plan(self, einsum_str, tensors, nominal_batch=64):
        # Per circuit structure: the batched contraction (path searched once, valid for any
        # minibatch size) and which operand slots are static gates / which op type each
        # parametrised slot is, so a forward builds each op type's gates in one call.
        if einsum_str not in self.plan_cache:
            input_subscripts, output_subscript = einsum_str.split('->')
            batched_str = f"{','.join('$' + s for s in input_subscripts.split(','))}->${output_subscript}"
            shapes, static, param_pos, by_type = [], [], [], defaultdict(list)
            for pos, gate in enumerate(tensors):
                if gate['name'] is None:
                    if gate['op_type'].startswith('cswap') and not hasattr(self, f"gate_{gate['op_type']}"):
                        self.register_buffer(f"gate_{gate['op_type']}", self._cswap_gate(int(gate['op_type'][5:])), persistent=False)
                    buf = getattr(self, f"gate_{gate['op_type']}")
                    static.append((pos, f"gate_{gate['op_type']}"))
                    shapes.append((nominal_batch, *buf.shape))
                else:
                    by_type[gate['op_type']].append(len(param_pos))
                    param_pos.append(pos)
                    n_wires = 2 if gate['op_type'].startswith('C') else 1
                    shapes.append((nominal_batch, *[2] * (2 * n_wires)))
            self.plan_cache[einsum_str] = {'expr': contract_expression(batched_str, *shapes), 'n_ops': len(tensors),
                                           'static': static, 'param_pos': param_pos, 'by_type': dict(by_type)}
        return self.plan_cache[einsum_str]

    def forward(self, batch_recipes):
        groups = defaultdict(list)
        for i, (einsum_str, tensors) in enumerate(batch_recipes):
            groups[einsum_str].append((i, tensors))

        thetas = torch.tanh(torch.cat([p for p in self.params])) * 2 * torch.pi
        dev = thetas.device

        batch_size = len(batch_recipes)
        if self.discard:
            results = torch.zeros(batch_size, 2**self.out_q, 2**self.out_q, dtype=self.precision, device=dev)
        else:
            results = torch.zeros(batch_size, *[2]*self.out_q, dtype=self.precision, device=dev)
        for einsum_str, items in groups.items():
            indices, tensor_lists = zip(*items)
            minibatch_size = len(indices)
            try:
                plan = self._plan(einsum_str, tensor_lists[0])
                operands = [None] * plan['n_ops']
                for pos, buf_name in plan['static']:
                    buf = getattr(self, buf_name)
                    operands[pos] = buf.expand(minibatch_size, *buf.shape)
                if plan['param_pos']:
                    param_idx = torch.tensor([[self.sym2param.get(tensors[p]['name'], self.unk_param_index) for p in plan['param_pos']]
                                              for tensors in tensor_lists], device=dev)
                    angles = thetas[param_idx].to(dtype=self.precision)  # [minibatch, n_param_gates]
                    for op_type, cols in plan['by_type'].items():
                        gate_batch = self.GATE_FNS[op_type](angles[:, cols])  # [minibatch, len(cols), 2, 2(, 2, 2)]
                        for j, c in enumerate(cols):
                            operands[plan['param_pos'][c]] = gate_batch[:, j]
                group_out = plan['expr'](*operands, backend='torch')

                if self.discard:
                    elems_per_item = group_out.numel() // minibatch_size
                    target_sent_dim = 2**self.out_q
                    if elems_per_item >= target_sent_dim and elems_per_item % target_sent_dim == 0:
                        sent_dim = target_sent_dim
                        traced_dim = elems_per_item // sent_dim
                    else:
                        sent_dim = elems_per_item
                        traced_dim = 1
                    group_out = group_out.reshape(minibatch_size, sent_dim, traced_dim)
                    rho = torch.bmm(group_out, group_out.conj().transpose(1, 2))
                    if self.renorm:
                        rho = rho / rho.diagonal(dim1=1, dim2=2).sum(-1).real.clamp(min=1e-12)[:, None, None]
                    results[list(indices)] = rho
                else:    
                    results[list(indices)] = group_out
            except Exception as e:
                if self.discard:
                    results[list(indices)] = torch.zeros(minibatch_size, 2**self.out_q, 2**self.out_q, dtype=self.precision, device=dev)
                else:
                    results[list(indices)] = torch.zeros(minibatch_size, *[2]*self.out_q, dtype=self.precision, device=dev)
                print(f"[Error] Failed to contract batch with error: {e}. Returning zeroed tensors for this batch.")
        return results