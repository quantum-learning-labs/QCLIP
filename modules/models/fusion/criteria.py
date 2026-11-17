import torch
import torch.nn as nn
from abc import ABC, abstractmethod as AbstractMethod

class BaseInfoNCELoss(ABC, nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    @AbstractMethod
    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement compute_similarity().")

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)
        similarity = self.compute_similarity(text_emb, image_emb)
        logits = similarity / self.temperature
        return self.cross_entropy(logits, labels)

class InfoNCE(BaseInfoNCELoss):
    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return (text_emb.flatten(start_dim=1) @ image_emb.flatten(start_dim=1).conj().T).abs()

class QInfoNCE(BaseInfoNCELoss):
    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        txt_emb = text_emb.flatten(start_dim=1).to(torch.complex64)
        img_emb = image_emb.flatten(start_dim=1).to(torch.complex64)
        txt_norm = (txt_emb.real.square() + txt_emb.imag.square()).sum(dim=1).clamp_min(1e-12).sqrt()
        img_norm = (img_emb.real.square() + img_emb.imag.square()).sum(dim=1).clamp_min(1e-12).sqrt()
        txt_emb = txt_emb / txt_norm.unsqueeze(1)
        img_emb = img_emb / img_norm.unsqueeze(1)
        return (txt_emb @ img_emb.conj().T).abs()