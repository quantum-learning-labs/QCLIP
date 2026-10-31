"""Step-4 constraints on TNModel tensors, applied after every optimiser step to the bank rows that step updated
(the rows RowAdam touched; untouched rows already satisfy them). Config block `text.constraint` (all off by default):

  norm: none | type | all   fix each type tensor / MPS core ('type') and also each word vector ('all') at its Frobenius
                            norm at init. The text output goes through LayerNorm, so every tensor's scale is redundant:
                            this fixes that gauge freedom and loses no expressivity.
  spectral_kappa: k         cap sigma_max <= k * sigma_rms of each type tensor's unfolding along its output leg (power
                            iteration with persistent vectors, then a rank-1 deflation of the excess)
  isometric: true           project each type tensor's output-leg unfolding onto the (scaled) isometries: polar factor
                            by Newton-Schulz, scale = init norm / sqrt(rank)
  mps_gauge: canonical      MPS type tensors in left-canonical form (core k = polar factor of its (left bond x leg, right
                            bond) matrix, the remainder pushed into core k+1): same represented tensor, gauge fixed
  init_orth: true           (once, at init) type tensors set to the polar factor of their norm init, same norm
  scope: all | root         root: the per-step constraints act only on the tensors carrying the sentence output leg (type
                            'out', i.e. the root types' cores); init_orth still applies to every type tensor

Everything is matmuls: CUDA svd/eigh are broken in our environments. "Output leg" = the leg towards the sentence output
(TNModel.out_votes majority, as for text.init: norm).
"""
import re
from collections import defaultdict
import torch


def polar(m, iters=15):
    # orthogonal polar factor of each matrix in a batch (B, r, c) by Newton-Schulz: Y <- 1.5 Y - 0.5 Y Y^T Y, started from
    # m / sqrt(||Gram||_F) (>= sigma_max, so the iteration converges); Gram on the short side
    tall = m.shape[1] > m.shape[2]
    y = m.transpose(1, 2) if tall else m   # wide: (B, r, c), r <= c
    g = y @ y.transpose(1, 2)
    y = y / g.flatten(1).norm(dim=1).sqrt().clamp_min(1e-30).view(-1, 1, 1)
    for _ in range(iters):
        y = 1.5 * y - 0.5 * (y @ y.transpose(1, 2)) @ y
    return y.transpose(1, 2) if tall else y


def unfold(x, leg):
    # (B, *shape) -> (B, shape[leg], rest)
    return x.movedim(leg + 1, 1).reshape(x.shape[0], x.shape[leg + 1], -1)


def fold(m, shape, leg):
    moved = (shape[leg],) + tuple(s for i, s in enumerate(shape) if i != leg)
    return m.reshape(m.shape[0], *moved).movedim(1, leg + 1)


class TNConstraints:
    def __init__(self, model, cfg, fresh=True):
        # fresh: a new model (init_orth applies); False for a resumed run (its weights already satisfy the constraints)
        self.model = model
        self.norm = cfg.get('norm', 'none')
        self.kappa = cfg.get('spectral_kappa')
        self.isometric = bool(cfg.get('isometric', False))
        self.gauge = cfg.get('mps_gauge', 'none')
        self.iters = int(cfg.get('ns_iters', 15))
        self.scope = cfg.get('scope', 'all')
        if self.scope not in ('all', 'root'):
            raise ValueError(f"text.constraint.scope must be 'all' or 'root', got {self.scope!r}")
        if self.norm not in ('none', 'type', 'all') or self.gauge not in ('none', 'canonical'):
            raise ValueError(f"text.constraint: bad norm/mps_gauge in {cfg}")
        if self.isometric and self.gauge != 'none':
            raise ValueError("text.constraint: isometric cores and mps_gauge conflict; use one")
        dev = model._bank(0).device
        nb = len(model.bank_shapes)
        self.is_word = [len(s) == 1 for s in model.bank_shapes]
        # output leg per bank row
        self.leg = [torch.zeros(model._bank(b).shape[0], dtype=torch.long, device=dev) for b in range(nb)]
        for name, (b, r) in model.sym_loc.items():
            votes = model.out_votes.get(name)
            if votes: self.leg[b][r] = votes.most_common(1)[0][0]
        # scope root: per bank, which rows belong to tensors of a root type ('out' among its types)
        self.in_scope = [torch.ones(model._bank(b).shape[0], dtype=torch.bool, device=dev) for b in range(nb)]
        if self.scope == 'root':
            for m in self.in_scope: m.zero_()
            for name, (b, r) in model.sym_loc.items():
                if 'out' in name.split(':')[-1].split('_core')[0].split('@'): self.in_scope[b][r] = True
        self.u = {}   # spectral: persistent power-iteration vectors per (bank, leg): (rows, shape[leg])
        # MPS symbols: cores t:..._core0.._coreK-1 -> per (core count, banks) signature, the (n_symbols, K) row matrix
        cores = defaultdict(dict)
        for name, (b, r) in model.sym_loc.items():
            m = re.fullmatch(r'(.*)_core(\d+)', name)
            if m: cores[m.group(1)][int(m.group(2))] = (b, r)
        sig = defaultdict(list)
        for sym, cs in cores.items():
            ks = sorted(cs)
            sig[tuple(cs[k][0] for k in ks)].append([cs[k][1] for k in ks])
        self.mps = [(banks, torch.tensor(rows, device=dev)) for banks, rows in sig.items()]
        if cfg.get('init_orth') and fresh:
            self.init_orth()
        if self.gauge == 'canonical':
            self._canonical(touched=None, iters=60)   # same represented tensors, canonical gauge from the start
        self.norm0 = [model._bank(b).flatten(1).norm(dim=1).clone() for b in range(nb)]   # target norms (after canonical)

    @property
    def active(self):
        return self.norm != 'none' or self.kappa or self.isometric or self.gauge != 'none'

    @torch.no_grad()
    def init_orth(self):
        # text.constraint.init_orth: type tensors -> polar factor of their unfolding, init norm kept
        for b in range(len(self.model.bank_shapes)):
            if not self.is_word[b]:
                rows = torch.arange(self.model._bank(b).shape[0], device=self.leg[b].device)
                self._isometric(b, rows, self.model._bank(b).flatten(1).norm(dim=1), iters=60)

    @torch.no_grad()
    def apply(self):
        touched = {}
        for bank, rows, _ in getattr(self.model, '_last_row_grads', []):
            b = next(i for i in range(len(self.model.bank_shapes)) if self.model._bank(i) is bank)
            touched[b] = rows
        if self.gauge == 'canonical':   # the remainders land in last cores the step may not have touched: renorm those too
            for b, r in self._canonical(touched).items():
                touched[b] = torch.unique(torch.cat(([touched[b]] if b in touched else []) + r))
        for b, rows in touched.items():
            if self.scope != 'all':
                rows = rows[self.in_scope[b][rows]]
                if len(rows) == 0: continue
            if self.is_word[b]:
                if self.norm == 'all': self._renorm(b, rows)
                continue
            if self.isometric:
                self._isometric(b, rows, self.norm0[b][rows])
                continue
            if self.kappa:
                self._spectral(b, rows)
            if self.norm != 'none':
                self._renorm(b, rows)

    @torch.no_grad()
    def _renorm(self, b, rows):
        bank = self.model._bank(b)
        p = bank.index_select(0, rows)
        scale = self.norm0[b][rows] / p.flatten(1).norm(dim=1).clamp_min(1e-30)
        bank.index_copy_(0, rows, p * scale.view(-1, *([1] * (p.dim() - 1))))

    def _by_leg(self, b, rows):
        legs = self.leg[b][rows]
        for leg in legs.unique().tolist():
            yield leg, rows[legs == leg]

    @torch.no_grad()
    def _isometric(self, b, rows, target_norm, iters=None):
        bank, shape = self.model._bank(b), self.model.bank_shapes[b]
        for leg, r in self._by_leg(b, rows):
            m = unfold(bank.index_select(0, r), leg)
            q = polar(m, iters or self.iters)
            q = q * (target_norm[self.leg[b][rows] == leg] / q.flatten(1).norm(dim=1).clamp_min(1e-30)).view(-1, 1, 1)
            bank.index_copy_(0, r, fold(q, shape, leg))

    @torch.no_grad()
    def _spectral(self, b, rows):
        bank, shape = self.model._bank(b), self.model.bank_shapes[b]
        for leg, r in self._by_leg(b, rows):
            m = unfold(bank.index_select(0, r), leg)   # (n, d, rest)
            key = (b, leg)
            if key not in self.u:
                g = torch.Generator(device='cpu').manual_seed(b * 1000 + leg)
                self.u[key] = torch.randn(bank.shape[0], shape[leg], generator=g).to(bank.device)
            u = self.u[key].index_select(0, r)
            for _ in range(3):
                v = torch.nn.functional.normalize((u.unsqueeze(1) @ m).squeeze(1), dim=1)
                u = torch.nn.functional.normalize((m @ v.unsqueeze(2)).squeeze(2), dim=1)
            self.u[key].index_copy_(0, r, u)
            sigma = (u.unsqueeze(1) @ m @ v.unsqueeze(2)).view(-1)
            k = min(m.shape[1], m.shape[2])
            cap = self.kappa * m.flatten(1).norm(dim=1) / k ** 0.5
            excess = (sigma - cap).clamp_min(0)
            m = m - excess.view(-1, 1, 1) * u.unsqueeze(2) * v.unsqueeze(1)
            bank.index_copy_(0, r, fold(m, shape, leg))

    @torch.no_grad()
    def _canonical(self, touched, iters=None):
        # left-canonical MPS: core k (left bond [x leg], right bond) -> polar factor Q, R = Q^T core_k pushed into core k+1's
        # left bond; touched=None: every MPS symbol, else the symbols with at least one touched core
        model, last = self.model, defaultdict(list)
        for banks, rows in self.mps:
            if touched is not None:
                hit = torch.zeros(len(rows), dtype=torch.bool, device=rows.device)
                for k, b in enumerate(banks):
                    if b in touched:
                        hit |= torch.isin(rows[:, k], touched[b])
                rows = rows[hit]
                if len(rows) == 0: continue
            for k in range(len(banks) - 1):
                bk, bn = model._bank(banks[k]), model._bank(banks[k + 1])
                c = bk.index_select(0, rows[:, k])
                m = c.reshape(len(rows), -1, c.shape[-1])   # (n, left [x leg], right bond)
                q = polar(m, iters or self.iters)
                rmat = q.transpose(1, 2) @ m                 # (n, right bond, right bond)
                nxt = bn.index_select(0, rows[:, k + 1])
                nxt = (rmat @ nxt.reshape(len(rows), nxt.shape[1], -1)).reshape(nxt.shape)
                bk.index_copy_(0, rows[:, k], q.reshape(c.shape))
                bn.index_copy_(0, rows[:, k + 1], nxt)
            last[banks[-1]].append(rows[:, -1])
        return last   # bank -> [rows of the last cores changed]
