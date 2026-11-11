import torch, math, itertools
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class BaseInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def describe(self) -> None:
        hyperparams = self.get_hyperparams()
        param_str = " | ".join([f"{k}: {v}" for k, v in hyperparams.items()])
        print(f"{self.__class__.__name__} {param_str}")

    def get_hyperparams(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "label_smoothing": self.label_smoothing,
        }

    def _ensure_density_matrices(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B = text_emb.size(0)
        if text_emb.dim() == 2:
            D = int(math.sqrt(text_emb.size(1)))
            rho_text = text_emb.view(B, D, D).to(torch.complex64)
            rho_image = image_emb.view(B, D, D).to(torch.complex64)
        else:
            D = text_emb.size(-1)
            rho_text = text_emb.to(torch.complex64)
            rho_image = image_emb.to(torch.complex64)
        return rho_text, rho_image, D

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement compute_similarity().")

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=text_emb.device)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor, swap_text_emb: torch.Tensor = None,
                swap_valid: torch.Tensor = None) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)

        # 1. Compute similarity matrix [B, B]
        similarity = self.compute_similarity(text_emb, image_emb)

        # 2. Scale by temperature
        logits = similarity / self.temperature

        # 3. Symmetrized Cross-Entropy Loss
        #    swap_text_emb (swap_negatives): each image also scores the batch's subject/object-swapped captions,
        #    except captions without a real swap (swap_valid False), which are dropped
        logits_i2t = logits.T
        if swap_text_emb is not None:
            swap_logits = self.compute_similarity(swap_text_emb, image_emb).T / self.temperature
            logits_i2t = torch.cat([logits_i2t, swap_logits[:, swap_valid.bool()]], dim=1)
        loss_t2i = self.cross_entropy(logits, labels)
        loss_i2t = self.cross_entropy(logits_i2t, labels)
        sym_loss = 0.5 * (loss_t2i + loss_i2t)

        # 4. Optional Regularization
        reg_loss = self.compute_regularization(text_emb, image_emb)

        return sym_loss + reg_loss

class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)
        text_emb = text_emb.flatten(start_dim=1)
        image_emb = image_emb.flatten(start_dim=1)

        logits = (text_emb @ image_emb.conj().T).abs() / self.temperature

        loss_t2i = self.cross_entropy(logits, labels)
        loss_i2t = self.cross_entropy(logits.T, labels)
        
        return 0.5 * (loss_t2i + loss_i2t)

class HS_InfoNCE(BaseInfoNCELoss):
    # similarity = Tr(rho sigma) / sqrt(P_rho P_sigma): Hilbert-Schmidt cosine *including* the identity component
    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.1, label_smoothing: float = 0.1, eps: float = 1e-7):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.lambda_reg = lambda_reg
        self.eps = eps

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"lambda_reg": self.lambda_reg, "eps": self.eps})
        return hp

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        text_flat = F.normalize(text_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        image_flat = F.normalize(image_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        return (text_flat @ image_flat.conj().T).abs()

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        if self.lambda_reg > 0 and B > 1:
            text_flat = F.normalize(text_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
            txt_overlap = torch.clamp((text_flat @ text_flat.conj().T).abs(), 0.0, 1.0 - self.eps)
            txt_thetas = torch.acos(txt_overlap)
            txt_logits = 1.0 - (txt_thetas / (math.pi / 2))
            
            mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
            return self.lambda_reg * txt_logits[mask].mean()
        return torch.tensor(0.0, device=text_emb.device)

class FS_InfoNCE(BaseInfoNCELoss):
    """InfoNCE on pure states with one overlap map f(o), o = |<a|b>|, used for BOTH the text-image similarity and the
    text-text spreading regulariser (lambda_reg * mean over caption pairs i != j of f(|<t_i|t_j>|)):
      'arccos' (default): f(o) = 1 - arccos(o) / (pi/2)        in [0, 1]: one minus the normalised Fubini-Study angle
      'arcsin'          : f(o) = arcsin(1/2 + o/2) / (pi/2)    in [1/3, 1]: a monotone transform of root fidelity
      'legacy'          : similarity 'arcsin', regulariser 'arccos' (the mixed form every run before 2026-11-11 used;
                          bit-identical to the old class, kept so existing configs, checkpoints and teachers reproduce)
    Note 1 - arccos(o)/(pi/2) == arcsin(o)/(pi/2): the two conventions differ only by the 1/2 + o/2 shift."""
    CONVENTIONS = ('arccos', 'arcsin', 'legacy')

    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.1, label_smoothing: float = 0.1, eps: float = 1e-7,
                 convention: str = 'arccos'):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        if convention not in self.CONVENTIONS:
            raise ValueError(f"FS_InfoNCE convention '{convention}' not in {self.CONVENTIONS}")
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.convention = convention

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"lambda_reg": self.lambda_reg, "eps": self.eps, "convention": self.convention})
        return hp

    def overlap_map(self, overlap: torch.Tensor, term: str) -> torch.Tensor:
        """f(o) for term 'similarity' or 'regularisation' under this convention (overlap already clamped to [0, 1 - eps])."""
        conv = self.convention if self.convention != 'legacy' else ('arcsin' if term == 'similarity' else 'arccos')
        if conv == 'arccos':
            return 1.0 - (torch.acos(overlap) / (math.pi / 2))
        return torch.asin(0.5 + (0.5 * overlap)) / (math.pi / 2)

    def _overlap(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_flat = F.normalize(a.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        b_flat = F.normalize(b.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        return torch.clamp((a_flat @ b_flat.conj().T).abs(), 0.0, 1.0 - self.eps)

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return self.overlap_map(self._overlap(text_emb, image_emb), 'similarity')

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        if self.lambda_reg > 0 and B > 1:
            txt_logits = self.overlap_map(self._overlap(text_emb, text_emb), 'regularisation')
            mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
            return self.lambda_reg * txt_logits[mask].mean()
        return torch.tensor(0.0, device=text_emb.device)

class UJ_InfoNCE(BaseInfoNCELoss):
    # similarity = arcsin(sqrt F) / (pi/2) with F = Tr(rho sigma), the Uhlmann-Jozsa fidelity for a pure image state sigma,
    # computed exactly in complex128 (a 12-step Newton-Schulz square root, used before 2026-10-17, was off by up to 0.26)
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, eps: float = 1e-6):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.eps = eps

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"eps": self.eps})
        return hp

    def _unit_trace(self, emb: torch.Tensor) -> torch.Tensor:
        D = int(round(math.sqrt(emb[0].numel())))
        rho = emb.reshape(emb.size(0), D, D).to(torch.complex128)
        return rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real.clamp(min=1e-12)[:, None, None]

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        fidelity = torch.einsum('aij,bji->ab', self._unit_trace(text_emb), self._unit_trace(image_emb)).real.clamp(0, 1).float()
        return torch.asin(fidelity.clamp(self.eps, 1 - self.eps).sqrt()) / (math.pi / 2.0)

class Pauli_InfoNCE(BaseInfoNCELoss):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, num_qubits: int = 3, eps: float = 1e-8, purity_weight: float = 0.1):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.num_qubits = num_qubits
        self.purity_weight = purity_weight
        self.dim_hilbert = 2 ** num_qubits
        self.eps = eps
        
        I = torch.eye(2, dtype=torch.complex64)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        paulis = [I, X, Y, Z]
        
        basis_list = []
        for p_tuple in itertools.product(paulis, repeat=num_qubits):
            if all(torch.equal(p, I) for p in p_tuple):
                continue
            p_tensor = p_tuple[0]
            for p in p_tuple[1:]:
                p_tensor = torch.kron(p_tensor, p)
            basis_list.append(p_tensor)
            
        self.register_buffer("pauli_basis", torch.stack(basis_list), persistent=False)

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({
            "purity_weight": self.purity_weight,
            "num_qubits": self.num_qubits,
            "eps": self.eps
        })
        return hp

    def _to_pauli_vector(self, rho: torch.Tensor) -> torch.Tensor:
        pauli_vec = torch.real(torch.einsum('bmn, knm -> bk', rho, self.pauli_basis.to(rho.device)))
        return pauli_vec

    def _compute_purity_from_pauli(self, v_pauli: torch.Tensor) -> torch.Tensor:
        v_norm_sq = torch.sum(v_pauli ** 2, dim=-1)
        purity = (1.0 + v_norm_sq) / self.dim_hilbert
        return torch.clamp(purity, min=1.0 / self.dim_hilbert, max=1.0)

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        rho_text, rho_image, _ = self._ensure_density_matrices(text_emb, image_emb)
        
        v_text = self._to_pauli_vector(rho_text)    # [B_txt, 63]
        v_image = self._to_pauli_vector(rho_image)  # [B_img, 63]
        
        v_text_norm = torch.nn.functional.normalize(v_text, dim=-1, eps=self.eps)
        v_image_norm = torch.nn.functional.normalize(v_image, dim=-1, eps=self.eps)
        
        return torch.matmul(v_text_norm, v_image_norm.T)

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        if self.purity_weight <= 0:
            return torch.tensor(0.0, device=text_emb.device)

        rho_text, rho_image, _ = self._ensure_density_matrices(text_emb, image_emb)
        v_text = self._to_pauli_vector(rho_text)
        
        text_purity = self._compute_purity_from_pauli(v_text)
        text_purity_loss = (1.0 - text_purity).mean()

        loss_purity = text_purity_loss

        return self.purity_weight * loss_purity





































