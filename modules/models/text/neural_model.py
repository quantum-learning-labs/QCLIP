from typing import List, Union
import torch
import torch.nn as nn
from tqdm import tqdm

class WordMLP(nn.Module):
    """Dynamically sized gating logic for compositional words."""
    def __init__(self, num_inputs: int, hdim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs * hdim, hdim * 2),
            nn.LayerNorm(hdim * 2),
            nn.GELU(),
            nn.Linear(hdim * 2, hdim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPModel(nn.Module):
    def __init__(self, hdim: int, out_dim: int = 512, batched: bool = False):
        super().__init__()
        self.batched = batched   # step 5d: forward_batched (same outputs up to float rounding, one bmm per tree level and arity)
        self.out_dim = out_dim
        self.hdim = hdim
        self.leaves = nn.ParameterDict()
        self.mlps = nn.ModuleDict()
        self.final_projection = nn.Linear(hdim, out_dim)
        self.norm_head = nn.LayerNorm(out_dim)

    def from_plans(self, plans_arr: List[List[dict]]):
        unique_leaves = set()
        unique_mlps = {}
        target_device = next(self.parameters()).device if list(self.parameters()) else 'cuda'

        for row in tqdm(plans_arr):
            is_nested = isinstance(row, list) and len(row) > 0 and isinstance(row[0], list)
            plans = row if is_nested else [row]
            for plan in plans:
                for node in plan:
                    symbol = node['name']
                    inputs_required = len(node['idx']) - 1
                    
                    if inputs_required == 0:
                        unique_leaves.add(symbol)
                    else:
                        unique_mlps[symbol] = inputs_required
        # temp_mlps[sym] = WordMLP(inputs_required, self.dim)
        # temp_leaves[sym] = nn.Parameter(torch.randn(self.dim, device='cpu') * 0.02)
        temp_leaves, temp_mlps = {}, {}
        for symbol in tqdm(sorted(unique_leaves), desc="Allocating Leaf Parameters"):   # sorted: set order follows the per-process string hash, so the seed did not fix the init
            temp_leaves[symbol] = nn.Parameter(torch.randn(self.hdim) * 0.02)
        for key, inputs_required in tqdm(sorted(unique_mlps.items()), desc="Allocating WordMLP Modules"):
            temp_mlps[key] = WordMLP(inputs_required, self.hdim)
        self.leaves = nn.ParameterDict(temp_leaves)
        self.mlps = nn.ModuleDict(temp_mlps)

    def forward_batched(self, batch: List[Union[dict, tuple]]) -> torch.Tensor:
        # Same semantics as forward(): a node reads, for each input wire, the value last written to that wire earlier in the
        # plan (zeros if none: diagrams with loops, e.g. two words sharing two wires, give such dead wires; a tree MLP cannot
        # express them), OOV leaves and unknown MLP symbols give zeros, the sentence vector is the last value written to wire 0.
        # Nodes are grouped by (level, arity); each group gathers its nodes' WordMLP weights and runs as batched mat-vecs.
        device = self.final_projection.weight.device
        h = self.hdim
        syms, refs, levels, roots = [], [], [], []
        for plan, _ in batch:
            last = {}
            for node in plan:
                out = node['out']
                ins = [last.get(i) for i in node['idx'] if i != out]
                lv = 0 if len(node['idx']) == 1 else 1 + max([levels[r] for r in ins if r is not None], default=-1)
                k = len(syms); syms.append(node['name']); refs.append(ins if len(node['idx']) > 1 else None); levels.append(lv)
                last[out] = k
            roots.append(last.get(0))
        N = len(syms); Z = N   # row N of the value table stays zero
        # keep only nodes that reach a sentence vector: dead branches would otherwise get all-zero gradients (None in the loop),
        # which Adam's weight decay and momentum would still move
        live, todo = [False] * N, [r for r in roots if r is not None]
        while todo:
            k = todo.pop()
            if live[k]: continue
            live[k] = True   # a node with an unknown MLP symbol outputs zeros: its inputs do not reach the sentence vector
            if refs[k] is not None and syms[k] in self.mlps: todo += [r for r in refs[k] if r is not None]
        H = torch.zeros(N + 1, h, device=device)
        leaf = [k for k in range(N) if live[k] and refs[k] is None and syms[k] in self.leaves]
        if leaf:
            u = sorted({syms[k] for k in leaf}); pos = {s: i for i, s in enumerate(u)}
            L = torch.stack([self.leaves[s] for s in u])
            H = H.index_copy(0, torch.tensor(leaf, device=device), L[torch.tensor([pos[syms[k]] for k in leaf], device=device)])
        groups = {}
        for k in range(N):
            if live[k] and refs[k] is not None and syms[k] in self.mlps:
                groups.setdefault((levels[k], len(refs[k])), []).append(k)
        for (lv, ar) in sorted(groups):
            ks = groups[(lv, ar)]
            u = sorted({syms[k] for k in ks}); pos = {s: i for i, s in enumerate(u)}
            j = torch.tensor([pos[syms[k]] for k in ks], device=device)
            nets = [self.mlps[s].net for s in u]
            W1 = torch.stack([n[0].weight for n in nets])[j]; b1 = torch.stack([n[0].bias for n in nets])[j]
            g = torch.stack([n[1].weight for n in nets])[j]; be = torch.stack([n[1].bias for n in nets])[j]
            W2 = torch.stack([n[3].weight for n in nets])[j]; b2 = torch.stack([n[3].bias for n in nets])[j]
            idx = torch.tensor([r if r is not None else Z for k in ks for r in refs[k]], device=device)
            x = H[idx].reshape(len(ks), ar * h)
            y = torch.bmm(W1, x.unsqueeze(2)).squeeze(2) + b1
            mu = y.mean(1, keepdim=True); var = y.var(1, unbiased=False, keepdim=True)
            y = (y - mu) / torch.sqrt(var + nets[0][1].eps) * g + be
            y = torch.bmm(W2, torch.nn.functional.gelu(y).unsqueeze(2)).squeeze(2) + b2
            H = H.index_copy(0, torch.tensor(ks, device=device), y)
        output = H[torch.tensor([r if r is not None else Z for r in roots], device=device)]
        output = self.final_projection(output)
        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)

    def forward(self, batch: List[Union[dict, tuple]]) -> torch.Tensor:
        if self.batched:
            return self.forward_batched(batch)
        device = next(self.parameters()).device
        batch_outputs = []

        for plan, _ in batch:
            max_wire = max(max(node['idx']) for node in plan) if plan else 0
            wires = [None] * (max_wire + 1)

            for node in plan:
                sym = node['name']
                out_wire = node['out']
                inputs_required = len(node['idx']) - 1

                if inputs_required == 0:
                    # Leaf fetch with Out-Of-Vocabulary fallback (critical for eval sets)
                    wires[out_wire] = self.leaves.get(sym, torch.zeros(self.hdim, device=device))
                else:
                    in_vectors = []
                    for idx in node['idx']:
                        if idx != out_wire:
                            # Pull from wire, fallback to zero if structural parse error caused a dead wire
                            vec = wires[idx] if wires[idx] is not None else torch.zeros(self.hdim, device=device)
                            in_vectors.append(vec)

                    if sym in self.mlps:
                        x = torch.cat(in_vectors, dim=0)
                        wires[out_wire] = self.mlps[sym](x)
                    else:
                        wires[out_wire] = torch.zeros(self.hdim, device=device)

            # Safety fallback for catastrophic parser failures returning empty trees
            final_vector = wires[0] if (wires and wires[0] is not None) else torch.zeros(self.hdim, device=device)
            batch_outputs.append(final_vector)

        output = torch.stack(batch_outputs, dim=0)
        output = self.final_projection(output)
        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)