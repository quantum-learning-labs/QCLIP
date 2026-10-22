import torch.nn as nn
import torch.nn.functional as F
from modules.utils.quantum_ops import amplitude_encoding

class FrozenCLIP(nn.Module):
    def __init__(self, clip_model = None, classical=True, normalize=False):
        super().__init__()
        self.clip_model = clip_model
        self.params = nn.ParameterList([])
        self.classical = classical
        self.normalize = normalize   # unit-norm vectors: cosine logits, as CLIP itself scores

    def forward(self, img_vecs):
        if self.classical:
            return F.normalize(img_vecs.float(), dim=-1) if self.normalize else img_vecs
        else:
            return amplitude_encoding(img_vecs)