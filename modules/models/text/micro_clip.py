from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SPECIAL = ['<pad>', '<unk>', '<sot>', '<eot>']

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    # CLIP's text-transformer block: pre-LayerNorm causal self-attention and a QuickGELU MLP, both residual
    def __init__(self, width, heads, ff_dim):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, ff_dim), QuickGELU(), nn.Linear(ff_dim, width))

    def forward(self, x, attn_mask, pad_mask):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=pad_mask, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))

class MicroCLIPModel(nn.Module):
    # MicroCLIP (LREC 2026 baseline): the CLIP text encoder (Radford et al. 2021) shrunk to `layers` blocks, `heads`
    # heads and width `width`, trained from scratch. Word-level vocabulary of the training (+ val) captions, built in
    # from_symbols (sorted, so a checkpoint's vocabulary is rebuilt identically); unseen words map to <unk>. A caption is
    # <sot> words <eot>; the sentence vector is the final-LayerNorm feature at <eot>, linearly projected to out_dim.
    def __init__(self, out_dim, width=32, layers=2, heads=2, ff_dim=None, max_len=32):
        super().__init__()
        self.out_dim, self.width, self.n_layers, self.heads = out_dim, width, layers, heads
        self.ff_dim = ff_dim or 4 * width
        self.max_len = max_len
        self.vocab = {}
        self.blocks = nn.ModuleList([ResidualAttentionBlock(width, heads, self.ff_dim) for _ in range(layers)])
        self.positional_embedding = nn.Parameter(torch.empty(max_len, width))
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, out_dim))
        self.token_embedding = None

    @property
    def pcount(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def from_symbols(self, token_lists, **_):
        words = sorted({t for toks in tqdm(token_lists, desc="MicroCLIP vocabulary") if isinstance(toks, list) for t in toks})
        self.vocab = {w: i for i, w in enumerate(SPECIAL + [w for w in words if w not in SPECIAL])}
        self.token_embedding = nn.Embedding(len(self.vocab), self.width)
        self.init_params()
        print(f"MicroCLIP: vocabulary {len(self.vocab)}, width {self.width}, layers {self.n_layers}, heads {self.heads}, "
              f"ff {self.ff_dim}: {self.pcount} parameters")

    def init_params(self):
        # CLIP's initialisation (scaled to this width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.width ** -0.5) * ((2 * self.n_layers) ** -0.5)
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=self.width ** -0.5)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=(2 * self.width) ** -0.5)
            nn.init.normal_(block.mlp[2].weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=self.width ** -0.5)

    def encode_ids(self, batch):
        unk, sot, eot = self.vocab['<unk>'], self.vocab['<sot>'], self.vocab['<eot>']
        seqs = [[sot] + [self.vocab.get(t, unk) for t in tokens][:self.max_len - 2] + [eot] for tokens, _ in batch]
        ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        return ids, torch.tensor([len(s) - 1 for s in seqs])

    def forward(self, batch: List[tuple]) -> torch.Tensor:
        device = self.positional_embedding.device
        ids, eot_pos = self.encode_ids(batch)
        ids, eot_pos = ids.to(device), eot_pos.to(device)
        L = ids.shape[1]
        x = self.token_embedding(ids) + self.positional_embedding[:L]
        causal = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        pad = ids == self.vocab['<pad>']
        for block in self.blocks:
            x = block(x, causal, pad)
        x = self.ln_final(x)
        return x[torch.arange(len(ids), device=device), eot_pos] @ self.text_projection
