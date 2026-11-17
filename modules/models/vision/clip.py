import torch
import torch.nn as nn

class FrozenCLIP(nn.Module):
    def __init__(self, clip_model=None, out_dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.out_dim = out_dim
        self.params = nn.ParameterList([])

        if out_dim <= 0:
            raise ValueError("out_dim must be positive")

        if self.clip_model is not None:
            self.clip_model.eval()
            for parameter in self.clip_model.parameters():
                parameter.requires_grad_(False)

    def forward(self, img_vecs):
        if self.clip_model is not None:
            with torch.no_grad():
                img_vecs = self.clip_model(img_vecs)

        img_vecs = torch.as_tensor(img_vecs, dtype=torch.float32)
        if img_vecs.shape[-1] < self.out_dim:
            raise ValueError(
                f"Requested out_dim={self.out_dim}, "
                f"but image features have dimension {img_vecs.shape[-1]}"
            )
        return img_vecs[..., :self.out_dim]