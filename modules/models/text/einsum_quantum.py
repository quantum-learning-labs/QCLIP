from tqdm import tqdm
from typing import Dict, List
import torch
import torch.nn as nn
from opt_einsum import contract_expression, contract_path, get_symbol
from torch.utils.checkpoint import checkpoint
from collections import defaultdict, Counter
import math
import numpy as np
from modules.compilation.quantum.gates import *

class VQCModel(nn.Module):
    def __init__(self, out_q: int = 9, discard=False, precision = torch.complex64, fuse='auto', fuse_mem_frac=0.25, ckpt_mb='auto', ckpt_mem_frac=0.35):
        super().__init__()
        # gate fusion (forward): 'auto' contracts each word box on its own first, once per distinct box in the batch, when its
        # tensor fits the memory budget (fuse_mem_frac of the GPU memory free at the first batch); 'off': gate by gate
        if fuse not in ('auto', 'off'):
            raise ValueError(f"text.fuse {fuse!r}: 'auto' or 'off'")
        self.fuse, self.fuse_mem_frac = fuse, fuse_mem_frac
        # memory: a contraction whose saved intermediates (for backward) would exceed the budget runs in batch chunks under
        # activation checkpointing (same result, its forward recomputed in backward); in eval, chunks bound the largest one.
        # Budget: ckpt_mb MiB, or 'auto' = ckpt_mem_frac of the GPU memory free at the first batch
        self.ckpt_mb, self.ckpt_mem_frac = ckpt_mb, ckpt_mem_frac
        self.ckpt_bytes = None if ckpt_mb == 'auto' else int(ckpt_mb) * 2**20
        self.max_fuse_wires = None
        self.struct_cache, self.fplan_cache = {}, {}
        self._reset_records()
        self.symbols: List[str] = []
        # word angles: one tensor (theta[i] belongs to symbols[i]) and the fixed angle of unseen words (theta_unk, 0)
        self.theta = nn.Parameter(torch.zeros(0))
        self.register_buffer('theta_unk', torch.zeros(1))
        self._register_load_state_dict_pre_hook(self._from_param_list)
        self.plan_cache = {}
        self.sym2param: Dict[str, nn.Parameter] = {}
        self.out_q: int = out_q
        self.precision: torch.dtype = precision
        self.discard = discard

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
        self.theta = nn.Parameter(param_values)
        self.sym2param = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.unk_param_index = num_symbols
        self._reset_records()

    REC_CAP = 2_000_000
    _rec_warned = False

    def _reset_records(self):
        # per-sentence records of the fused forward, built the first time a sentence is seen and kept in RAM for the run:
        # rec_cache[id(tensors)] = (tensors, (box instance ids, angle indices of the unfused gates)); box_inst[key] maps a
        # box's symbol names to its instance id, box_rows[key][id] holds that instance's angle indices
        self.rec_cache, self.box_inst, self.box_rows = {}, defaultdict(dict), defaultdict(list)

    def _record(self, plan, tensors):
        # the tensors list of a sentence is the same object every epoch (a cell of the compiled frame); holding it in
        # the cache keeps its id from being reused
        hit = self.rec_cache.get(id(tensors))
        if hit is not None and hit[0] is tensors:
            return hit[1]
        unk, s2p = self.unk_param_index, self.sym2param
        ids = []
        for b in plan['boxes']:
            names = tuple(tensors[p]['name'] for p in b['param_pos'])
            inst = self.box_inst[b['key']]
            i = inst.get(names)
            if i is None:
                i = inst[names] = len(inst)
                self.box_rows[b['key']].append([s2p.get(n, unk) for n in names])
            ids.append(i)
        rec = (ids, [s2p.get(tensors[p]['name'], unk) for p in plan['rest_param_src']])
        if len(self.rec_cache) >= self.REC_CAP:
            # the batches hold copies of the recipes (e.g. a pinning DataLoader), so ids never repeat: don't grow forever
            if not self._rec_warned:
                print(f"VQC: {self.REC_CAP} sentence records and still missing; the loader copies recipes, cache cleared")
                self._rec_warned = True
            self.rec_cache.clear()
        self.rec_cache[id(tensors)] = (tensors, rec)
        return rec

    def _rows(self, key, n_params, uniq):
        # angle indices of the given instances of a box structure, [len(uniq), n_params]
        rows = self.box_rows[key]
        return np.asarray([rows[u] for u in uniq], dtype=np.int64).reshape(len(uniq), n_params)

    @staticmethod
    def _from_param_list(state_dict, prefix, *args):
        # checkpoints from before the single-tensor angles store params.<i> (one per symbol) and params.<n> (unseen words)
        keys = sorted((k for k in state_dict if k.startswith(prefix + 'params.')), key=lambda k: int(k.rsplit('.', 1)[1]))
        if keys:
            values = [state_dict.pop(k).reshape(-1) for k in keys]
            state_dict[prefix + 'theta'] = torch.cat(values[:-1])
            state_dict[prefix + 'theta_unk'] = values[-1]

    def angles(self):
        # all angles, the unseen-word angle last (index unk_param_index), mapped to (-2pi, 2pi)
        return torch.tanh(torch.cat([self.theta, self.theta_unk])) * 2 * torch.pi

    def _get_params(self):
        with torch.no_grad():
            raw_tensor = torch.cat([self.theta, self.theta_unk])
            thetas = torch.tanh(raw_tensor) * 2 * torch.pi
            
            return {
                sym: float(thetas[idx].detach().cpu().item()) 
                for sym, idx in self.sym2param.items()
            }

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
        if self.fuse == 'off' or 'box' not in batch_recipes[0][1][0]:
            return self._forward_gates(batch_recipes)
        return self._forward_fused(batch_recipes)

    def _budget_wires(self, batch_size):
        # the largest box (in external wires) whose fused tensors fit the budget: complex64 entries, one per distinct box
        # of the batch (at most one per sentence), x3 for autograd copies
        if self.max_fuse_wires is None:
            dev = self.theta.device
            free = torch.cuda.mem_get_info(dev)[0] if dev.type == 'cuda' else 8 * 2**30
            budget = self.fuse_mem_frac * free
            entry = torch.empty((), dtype=self.precision).element_size()
            self.max_fuse_wires = max(1, int(math.log2(max(budget / (3 * entry * max(batch_size, 1)), 2))))
            print(f"VQC gate fusion: budget {budget / 2**30:.1f} GiB -> boxes with up to {self.max_fuse_wires} external wires fused")
        return self.max_fuse_wires

    def _struct(self, key, subs, out, tensors):
        # batched contraction of one box structure (canonical subscripts), shared by every box with that structure
        if key not in self.struct_cache:
            shapes, static, param_pos, by_type = [], [], [], defaultdict(list)
            for pos, gate in enumerate(tensors):
                if gate['name'] is None:
                    buf = getattr(self, f"gate_{gate['op_type']}")
                    static.append((pos, f"gate_{gate['op_type']}"))
                    shapes.append((64, *buf.shape))
                else:
                    by_type[gate['op_type']].append(len(param_pos))
                    param_pos.append(pos)
                    shapes.append((64, *[2] * (4 if gate['op_type'].startswith('C') else 2)))
            # chains of one-qubit operators along a wire ('ab','bc',... with b, c internal) are multiplied out first, by
            # stacked batched matmuls per chain length; the box contraction then sees one 2x2 per chain
            occ = Counter(''.join(subs) + out)
            one_q = [p for p, s in enumerate(subs) if len(s) == 2]
            starts = {subs[p][0]: p for p in one_q}
            nxt = {p: starts[subs[p][1]] for p in one_q if occ[subs[p][1]] == 2 and subs[p][1] in starts}
            heads = [p for p in one_q if p not in set(nxt.values())]
            chains = []
            for p in heads:
                chain = [p]
                while chain[-1] in nxt:
                    chain.append(nxt[chain[-1]])
                if len(chain) > 1:
                    chains.append(chain)
            by_len = defaultdict(list)
            for chain in chains:
                by_len[len(chain)].append(chain)
            red_chain = [(L, k) for L, chs in by_len.items() for k in range(len(chs))]
            in_chain = {p for chain in chains for p in chain}
            others = [p for p in range(len(subs)) if p not in in_chain]
            red_subs = [subs[ch[0]][0] + subs[ch[-1]][1] for L, chs in by_len.items() for ch in chs] + [subs[p] for p in others]
            red_shapes = [(64, 2, 2)] * len(red_chain) + [shapes[p] for p in others]
            expr = f"{','.join('$' + s for s in red_subs)}->${out}"
            self.struct_cache[key] = {'expr': contract_expression(expr, *red_shapes), **self._cost(expr, red_shapes),
                                      'n_ops': len(tensors), 'static': static,
                                      'param_pos': param_pos, 'by_type': dict(by_type), 'chains': dict(by_len),
                                      'red_chain': red_chain, 'others': others}
        return self.struct_cache[key]

    @staticmethod
    def _cost(expr, shapes):
        # entries per batch item: of the largest intermediate ('peak') and of all intermediates kept for backward ('saved')
        # (the path of contract_expression for these nominal-batch shapes; every intermediate carries the batch index)
        _, info = contract_path(expr, *shapes, shapes=True)
        nb = shapes[0][0]
        return {'peak': int(info.largest_intermediate) // nb, 'saved': int(sum(info.size_list)) // nb}

    def _run(self, plan, ops, n):
        # plan['expr'] on operands with leading batch dim n: in one go if it fits the budget, else in chunks (checkpointed
        # when training)
        item = torch.empty((), dtype=self.precision).element_size()
        if self.ckpt_bytes is None:
            dev = self.theta.device
            free = torch.cuda.mem_get_info(dev)[0] if dev.type == 'cuda' else 8 * 2**30
            self.ckpt_bytes = int(self.ckpt_mem_frac * free)
            print(f"VQC checkpoint budget: {self.ckpt_bytes / 2**30:.1f} GiB per contraction")
        train = torch.is_grad_enabled()
        per = (plan['saved'] if train else plan['peak']) * item
        if per * n <= self.ckpt_bytes or (n == 1 and not train):
            return plan['expr'](*ops, backend='torch')
        chunk = max(1, int(self.ckpt_bytes // per))
        run = lambda *x: plan['expr'](*x, backend='torch')
        outs = []
        for a in range(0, n, chunk):
            sub = [o[a:a + chunk] for o in ops]
            outs.append(checkpoint(run, *sub, use_reentrant=False) if train else run(*sub))
        return torch.cat(outs)

    def _contract_box(self, struct, operands, n):
        # one box structure for n distinct boxes: multiply out the gate chains, then contract
        products = {}
        for L, chs in struct['chains'].items():
            stack = torch.stack([operands[p] for ch in chs for p in ch], 1).reshape(n, len(chs), L, 2, 2)
            prod = stack[:, :, 0]
            for l in range(1, L):
                prod = prod @ stack[:, :, l]
            products[L] = prod
        red = [products[L][:, k] for L, k in struct['red_chain']] + [operands[p] for p in struct['others']]
        return self._run(struct, red, n)

    def _fplan(self, einsum_str, tensors, batch_size):
        # split a circuit into its word boxes: each box's operands contract to one tensor on its external wires (the
        # wires it shares with other boxes or the output; its fresh |0> inputs are absorbed). Boxes larger than the budget
        # stay gate by gate in the outer contraction.
        if einsum_str in self.fplan_cache:
            return self.fplan_cache[einsum_str]
        max_wires = self._budget_wires(batch_size)
        inputs, out = einsum_str.split('->')
        subs = inputs.split(',')
        count = Counter(''.join(subs) + out)
        members = defaultdict(list)
        for pos, t in enumerate(tensors):
            members[t.get('box')].append(pos)
        # a one-wire tensor outside every box (a post-selection <0|) joins the box whose wire it ends, so that wire is
        # internal to the box rather than leaving it and coming back
        owner = {c: b for b, pos in members.items() if b is not None for p in pos for c in subs[p]}
        for pos in list(members.get(None, [])):
            if len(subs[pos]) == 1 and subs[pos] in owner and count[subs[pos]] == 2:
                members[None].remove(pos)
                members[owner[subs[pos]]].append(pos)
        for b in members:
            members[b].sort()
        rest = list(members.pop(None, []))
        boxes = []
        for _, pos in sorted(members.items()):
            inside = Counter(''.join(subs[p] for p in pos))
            ext = list(dict.fromkeys(c for p in pos for c in subs[p] if count[c] > inside[c]))
            if len(ext) > max_wires:
                rest += pos
                continue
            relabel = {}
            for c in ''.join(subs[p] for p in pos):
                relabel.setdefault(c, get_symbol(len(relabel)))
            c_subs = [''.join(relabel[c] for c in subs[p]) for p in pos]
            c_out = ''.join(relabel[c] for c in ext)
            box_tensors = [tensors[p] for p in pos]
            key = (','.join(c_subs) + '->' + c_out, tuple((t['op_type'], t['name'] is None) for t in box_tensors))
            self._struct(key, c_subs, c_out, box_tensors)
            boxes.append({'key': key, 'ext': ''.join(ext),
                          'param_pos': [pos[i] for i in self.struct_cache[key]['param_pos']]})
        rest.sort()
        shapes = [(64, *[2] * len(b['ext'])) for b in boxes]
        static, param_pos, by_type = [], [], defaultdict(list)
        for j, pos in enumerate(rest):
            gate = tensors[pos]
            if gate['name'] is None:
                buf = getattr(self, f"gate_{gate['op_type']}")
                static.append((len(boxes) + j, f"gate_{gate['op_type']}"))
                shapes.append((64, *buf.shape))
            else:
                by_type[gate['op_type']].append(len(param_pos))
                param_pos.append(len(boxes) + j)
                shapes.append((64, *[2] * (4 if gate['op_type'].startswith('C') else 2)))
        outer = [b['ext'] for b in boxes] + [subs[p] for p in rest]
        expr = f"{','.join('$' + s for s in outer)}->${out}"
        plan = {'boxes': boxes, 'rest': rest, 'rest_param_src': [rest[i - len(boxes)] for i in param_pos],
                'expr': contract_expression(expr, *shapes), **self._cost(expr, shapes), 'n_ops': len(outer), 'static': static,
                'param_pos': param_pos, 'by_type': dict(by_type)}
        self.fplan_cache[einsum_str] = plan
        return plan

    def _gates(self, thetas, idx, struct, operands):
        # fill the parameterised operand slots of a (batched) contraction from rows of angle indices [n, n_params]
        angles = thetas[idx].to(dtype=self.precision)
        for op_type, cols in struct['by_type'].items():
            gate_batch = self.GATE_FNS[op_type](angles[:, cols])
            for j, c in enumerate(cols):
                operands[struct['param_pos'][c]] = gate_batch[:, j]

    def _forward_fused(self, batch_recipes):
        groups = defaultdict(list)
        for i, (einsum_str, tensors) in enumerate(batch_recipes):
            groups[einsum_str].append((i, tensors))
        thetas = self.angles()
        dev = thetas.device
        batch_size = len(batch_recipes)

        # all integer indices of the batch (angle rows, box lookups, output rows) go to the GPU as one host array, so the
        # batch makes one host-to-device copy; slots[name] = (start, shape) into it
        parts, slots, size = [], {}, 0
        def put(name, arr):
            nonlocal size
            parts.append(arr.reshape(-1))
            slots[name] = (size, arr.shape)
            size += arr.size

        # 1. per circuit structure: the sentences' records (built once per sentence and run)
        group_plans, by_key = [], defaultdict(list)
        for g, (einsum_str, items) in enumerate(groups.items()):
            plan = self._fplan(einsum_str, items[0][1], batch_size)
            recs = [self._record(plan, tensors) for _, tensors in items]
            m, nb = len(items), len(plan['boxes'])
            if nb:
                ids = np.asarray([r[0] for r in recs], dtype=np.int64).reshape(m, nb)
                for j, b in enumerate(plan['boxes']):
                    by_key[b['key']].append((g, j, ids[:, j]))
            if plan['param_pos']:
                put(('rest', g), np.asarray([r[1] for r in recs], dtype=np.int64).reshape(m, len(plan['rest_param_src'])))
            put(('out', g), np.fromiter((i for i, _ in items), dtype=np.int64, count=m))
            group_plans.append((g, items, plan))
        # 2. the distinct box instances of the batch, per box structure, and each sentence's position among them
        n_inst = {}
        for key, refs in by_key.items():
            uniq, inv = np.unique(np.concatenate([r for _, _, r in refs]), return_inverse=True)
            n_inst[key] = len(uniq)
            struct = self.struct_cache[key]
            if struct['param_pos']:
                put(('rows', key), self._rows(key, len(struct['param_pos']), uniq))
            start = 0
            for g, j, r in refs:
                put(('ref', g, j), inv[start:start + len(r)])
                start += len(r)
        flat = torch.from_numpy(np.concatenate(parts)).to(dev, non_blocking=False)
        def get(name):
            start, shape = slots[name]
            return flat[start:start + int(np.prod(shape))].reshape(shape)

        # 3. every distinct box contracted once per structure
        box_values = {}
        for key, n in n_inst.items():
            struct = self.struct_cache[key]
            operands = [None] * struct['n_ops']
            for pos, buf_name in struct['static']:
                buf = getattr(self, buf_name)
                operands[pos] = buf.expand(n, *buf.shape)
            if struct['param_pos']:
                self._gates(thetas, get(('rows', key)), struct, operands)
            box_values[key] = self._contract_box(struct, operands, n)

        # 4. each circuit structure: its boxes' tensors (looked up per sentence) and any unfused gates, contracted
        if self.discard:
            results = torch.zeros(batch_size, 2**self.out_q, 2**self.out_q, dtype=self.precision, device=dev)
        else:
            results = torch.zeros(batch_size, *[2]*self.out_q, dtype=self.precision, device=dev)
        for g, items, plan in group_plans:
            m = len(items)
            operands = [box_values[b['key']][get(('ref', g, j))] for j, b in enumerate(plan['boxes'])] + [None] * len(plan['rest'])
            for pos, buf_name in plan['static']:
                buf = getattr(self, buf_name)
                operands[pos] = buf.expand(m, *buf.shape)
            if plan['param_pos']:
                self._gates(thetas, get(('rest', g)), plan, operands)
            group_out = self._run(plan, operands, m)
            out_idx = get(('out', g))
            if self.discard:
                elems_per_item = group_out.numel() // m
                target_sent_dim = 2**self.out_q
                if elems_per_item >= target_sent_dim and elems_per_item % target_sent_dim == 0:
                    sent_dim, traced_dim = target_sent_dim, elems_per_item // target_sent_dim
                else:
                    sent_dim, traced_dim = elems_per_item, 1
                group_out = group_out.reshape(m, sent_dim, traced_dim)
                results[out_idx] = torch.bmm(group_out, group_out.conj().transpose(1, 2))
            else:
                results[out_idx] = group_out
        return results

    def _forward_gates(self, batch_recipes):
        groups = defaultdict(list)
        for i, (einsum_str, tensors) in enumerate(batch_recipes):
            groups[einsum_str].append((i, tensors))

        thetas = self.angles()
        dev = thetas.device

        batch_size = len(batch_recipes)
        if self.discard:
            results = torch.zeros(batch_size, 2**self.out_q, 2**self.out_q, dtype=self.precision, device=dev)
        else:
            results = torch.zeros(batch_size, *[2]*self.out_q, dtype=self.precision, device=dev)
        for einsum_str, items in groups.items():
            indices, tensor_lists = zip(*items)
            minibatch_size = len(indices)
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
                results[list(indices)] = rho
            else:    
                results[list(indices)] = group_out
        return results