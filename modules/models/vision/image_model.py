import math

import torch
from einops import rearrange
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn
from opt_einsum import contract_expression

from modules.models.vision.cp_node import CPQuadRankLayer

class ImageModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IMAGE_MODEL_")
    use_color: bool = True
    bond_dim: int = 64
    cp_rank: int = 32
    dropout: float = 0.3
    patch_size: int = 4
    image_size: int = 64


image_model_hyperparams = ImageModelSettings()

class TTNImageModel(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.in_channels = 3 if image_model_hyperparams.use_color else 1
        self.embedding_dim = embedding_dim
        self.bond_dim = image_model_hyperparams.bond_dim
        self.patch_size = image_model_hyperparams.patch_size

        num_patches_side = image_model_hyperparams.image_size // self.patch_size
        num_patches = num_patches_side**2

        assert math.log(num_patches, 4).is_integer(), "Number of patches must be a power of 4 for quad-tree structure"

        self.color_factor = nn.Parameter(torch.empty(self.in_channels, self.bond_dim))
        self.pixel_factor = nn.Parameter(torch.empty(self.patch_size**2, self.bond_dim))
        nn.init.xavier_uniform_(self.color_factor)
        nn.init.xavier_uniform_(self.pixel_factor)

        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, self.bond_dim))
        self.pos_scale = nn.Parameter(torch.tensor(0.05))

        self.depth = int(math.log(num_patches, 4))
        self.layers = nn.ModuleList()

        current_nodes = num_patches // 4
        in_dim = self.bond_dim
        gains = [2.0, 1.5, 1.0, 1.0]

        for i in range(self.depth):
            use_res = True if i > 1 else False
            gain = gains[i]
            self.layers.append(
                CPQuadRankLayer(
                    num_nodes=current_nodes,
                    in_dim=in_dim,
                    out_dim=in_dim * 2,
                    rank=image_model_hyperparams.cp_rank,
                    dropout_p=image_model_hyperparams.dropout,
                    use_residual=use_res,
                    gain_factor=gain,
                )
            )
            current_nodes //= 4
            in_dim *= 2

        self.final_norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, self.embedding_dim)
        self._expression_cache = {}

    def _compile(self, equation: str, *shapes):
        cache_key = (equation, tuple(shapes))
        if cache_key not in self._expression_cache:
            self._expression_cache[cache_key] = contract_expression(equation, *shapes)
        return self._expression_cache[cache_key]

    def compile_batch(self, batch_size: int) -> None:
        patch_features = self.patch_size**2
        self._compile("bncp,ck,pk->bnk", 
                      (batch_size, self._num_patches(), self.in_channels, patch_features), 
                      self.color_factor.shape,
                      self.pixel_factor.shape)

        in_dim = self.bond_dim
        for layer in self.layers:
            layer.compile_batch(batch_size=batch_size, child_dim=in_dim)
            in_dim *= 2

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict.pop("expression_cache", None)
        self._expression_cache = {}
        return super().load_state_dict(state_dict, strict=strict)

    def _num_patches(self) -> int:
        num_patches_side = image_model_hyperparams.image_size // self.patch_size
        return num_patches_side**2

    def forward(self, x):
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=self.patch_size, p2=self.patch_size)
        x = self._compile(
            "bncp,ck,pk->bnk", patches.shape, self.color_factor.shape, self.pixel_factor.shape
        )(patches, self.color_factor, self.pixel_factor)

        x = x + (self.positional_embedding * self.pos_scale)

        current_grid_dim = int(math.sqrt(x.shape[1]))
        for layer in self.layers:
            x = rearrange(x, "b (h w) c -> b c h w", h=current_grid_dim)
            x = rearrange(x, "b c (h h2) (w w2) -> b (h w) (h2 w2) c", h2=2, w2=2)
            x = layer(x)
            current_grid_dim //= 2

        x = x.squeeze(1)
        x = self.final_norm(x)
        x = self.head(x)
        return nn.functional.normalize(x, p=2, dim=-1)
