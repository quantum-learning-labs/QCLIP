import torch
from torch import nn
from opt_einsum import contract_expression


class CPQuadRankLayer(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, rank, dropout_p=0.0, use_residual=True, gain_factor=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.dropout_p = dropout_p
        self.use_residual = use_residual

        self.factor_tl = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_tr = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_bl = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_br = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_out = nn.Parameter(torch.empty(num_nodes, rank, out_dim))
        self.gain = nn.Parameter(torch.full((num_nodes, 1), gain_factor))

        if use_residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        self._expression_cache = {}

        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for f in [self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br]:
                nn.init.orthogonal_(f)
            nn.init.orthogonal_(self.factor_out)

    def _rms_norm(self, t, eps=1e-6):
        rms = torch.sqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
        return t / rms

    def _compile(self, equation: str, *shapes):
        cache_key = (equation, tuple(shapes))
        if cache_key not in self._expression_cache:
            self._expression_cache[cache_key] = contract_expression(equation, *shapes)
        return self._expression_cache[cache_key]

    def compile_batch(self, batch_size: int, child_dim: int) -> None:
        self._compile("bni,nri->bnr", (batch_size, self.num_nodes, child_dim), (self.num_nodes, self.rank, child_dim))
        self._compile("bnr,nro->bno", (batch_size, self.num_nodes, self.rank), (self.num_nodes, self.rank, self.factor_out.shape[-1]))

    def forward(self, x):
        p_tl = self._compile("bni,nri->bnr", x[:, :, 0, :].shape, self.factor_tl.shape)(x[:, :, 0, :], self.factor_tl)
        p_tr = self._compile("bni,nri->bnr", x[:, :, 1, :].shape, self.factor_tr.shape)(x[:, :, 1, :], self.factor_tr)
        p_bl = self._compile("bni,nri->bnr", x[:, :, 2, :].shape, self.factor_bl.shape)(x[:, :, 2, :], self.factor_bl)
        p_br = self._compile("bni,nri->bnr", x[:, :, 3, :].shape, self.factor_br.shape)(x[:, :, 3, :], self.factor_br)

        p_tl, p_tr, p_bl, p_br = map(self._rms_norm, [p_tl, p_tr, p_bl, p_br])
        merged = p_tl * p_tr * p_bl * p_br
        merged = merged * self.gain.unsqueeze(0)

        if self.training and self.dropout_p > 0:
            merged = nn.functional.dropout(merged, p=self.dropout_p)

        out = self._compile("bnr,nro->bno", merged.shape, self.factor_out.shape)(merged, self.factor_out)

        if self.use_residual:
            return out + self.res_proj(x.mean(dim=2))
        return out
