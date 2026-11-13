import torch, math, itertools
import torch.nn as nn
import torch.nn.functional as F

def unit_rows(emb: torch.Tensor) -> torch.Tensor:
    # each embedding flattened to a unit vector (a statevector, or a density matrix in Hilbert-Schmidt norm)
    return F.normalize(emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)

def density_matrices(emb: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
    # [B, D*D] or [B, D, D] -> [B, D, D]
    D = int(round(math.sqrt(emb[0].numel())))
    return emb.reshape(emb.size(0), D, D).to(dtype)

def fs_similarity(overlap: torch.Tensor, eps: float) -> torch.Tensor:
    # 1 - theta / (pi/2) for the Fubini-Study angle theta = arccos(overlap), written as arcsin(overlap) / (pi/2);
    # overlap = |<t|i>| for pure states, sqrt(Tr(rho sigma)) when the image state sigma is pure. Clamped below 1:
    # the gradient of arcsin diverges at 1
    return torch.asin(overlap.clamp(0.0, 1.0 - eps)) / (math.pi / 2)

class BaseInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement compute_similarity().")

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
        return 0.5 * (loss_t2i + loss_i2t)

class InfoNCE(BaseInfoNCELoss):
    # classical embeddings (tn, mlp): similarity = |<t, i>| of the unnormalised vectors
    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return (text_emb.flatten(start_dim=1) @ image_emb.flatten(start_dim=1).conj().T).abs()

class FS_InfoNCE(BaseInfoNCELoss):
    # pure states: similarity = 1 - theta / (pi/2), theta = arccos|<t|i>| the Fubini-Study angle (fs_similarity)
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.1, eps: float = 1e-7):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.eps = eps

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return fs_similarity((unit_rows(text_emb) @ unit_rows(image_emb).conj().T).abs(), self.eps)

class HS_InfoNCE(BaseInfoNCELoss):
    # density matrices: similarity = Tr(rho sigma) / sqrt(P_rho P_sigma), the Hilbert-Schmidt cosine including the identity component
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.1):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return (unit_rows(text_emb) @ unit_rows(image_emb).conj().T).abs()

class UJ_InfoNCE(BaseInfoNCELoss):
    # density matrices: the Fubini-Study similarity of FS_InfoNCE with overlap sqrt(F), F = Tr(rho sigma) the Uhlmann-Jozsa
    # fidelity, which needs no matrix square root because the image state sigma is pure (pure text states: FS_InfoNCE)
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, eps: float = 1e-6):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.eps = eps

    def _unit_trace(self, emb: torch.Tensor) -> torch.Tensor:
        rho = density_matrices(emb, torch.complex128)
        return rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real.clamp(min=1e-12)[:, None, None]

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        fidelity = torch.einsum('aij,bji->ab', self._unit_trace(text_emb), self._unit_trace(image_emb)).real.clamp(0, 1).float()
        return fs_similarity(fidelity.clamp(min=self.eps).sqrt(), self.eps)   # F >= eps: d sqrt(F)/dF is finite

class Pauli_InfoNCE(BaseInfoNCELoss):
    # density matrices: similarity = cosine of the Pauli vectors r_k = Tr(rho P_k) (identity dropped), i.e. of the traceless
    # parts rho - I/d; blind to purity, hence the optional purity regulariser (purity_weight)
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

    def _to_pauli_vector(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.einsum('bmn, knm -> bk', rho, self.pauli_basis.to(rho.device)))

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        v_text_norm = F.normalize(self._to_pauli_vector(density_matrices(text_emb)), dim=-1, eps=self.eps)
        v_image_norm = F.normalize(self._to_pauli_vector(density_matrices(image_emb)), dim=-1, eps=self.eps)
        return torch.matmul(v_text_norm, v_image_norm.T)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor, swap_text_emb: torch.Tensor = None,
                swap_valid: torch.Tensor = None) -> torch.Tensor:
        loss = super().forward(text_emb, image_emb, swap_text_emb, swap_valid)
        if self.purity_weight <= 0:
            return loss
        # purity penalty: purity_weight * (1 - Tr(rho^2)), Tr(rho^2) = (1 + |r|^2) / d
        v_norm_sq = torch.sum(self._to_pauli_vector(density_matrices(text_emb)) ** 2, dim=-1)
        purity = torch.clamp((1.0 + v_norm_sq) / self.dim_hilbert, min=1.0 / self.dim_hilbert, max=1.0)
        return loss + self.purity_weight * (1.0 - purity).mean()
