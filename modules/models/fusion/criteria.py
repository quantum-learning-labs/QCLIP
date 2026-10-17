import torch, math, itertools
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
class DB_Sqrt(nn.Module):
    def __init__(self, iters: int = 6, eps: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        orig_shape = A.shape
        D = orig_shape[-1]
        
        A_flat = A.reshape(-1, D, D)
        B = A_flat.size(0)
        device, dtype = A.device, A.dtype

        A_flat = complex_nan_to_num(A_flat, nan=0.0, posinf=1.0, neginf=-1.0)

        # 2. Hermitian Symmetrization: A = 0.5 * (A + A^H)
        A_flat = 0.5 * (A_flat + A_flat.conj().transpose(-2, -1))

        # 3. Trace Conditioning: Scale matrix to unit trace to optimize convergence rate
        tr_A = torch.real(torch.einsum('bii->b', A_flat)).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
        A_scaled = A_flat / tr_A

        # 4. Diagonal Ridge Jitter: Prevents singularity for zero/degenerate eigenvalues
        I = torch.eye(D, dtype=dtype, device=device).unsqueeze(0).expand(B, D, D)
        
        # Spectrum-splitting jitter to resolve degenerate spectra during backward pass
        splitting_vec = torch.linspace(self.eps, self.eps * 10, D, device=device, dtype=A_flat.real.dtype)
        jitter = torch.diag_embed(splitting_vec).to(dtype).unsqueeze(0).expand(B, D, D)
        
        Y = A_scaled + jitter
        Z = I.clone()

        # 5. Coupled Denman-Beavers Iterations
        for _ in range(self.iters):
            Y_inv = torch.linalg.inv(Y)
            Z_inv = torch.linalg.inv(Z)

            Y = 0.5 * (Y + Z_inv)
            Z = 0.5 * (Z + Y_inv)

            # Re-enforce Hermitian symmetry at each step to suppress floating-point drift
            Y = 0.5 * (Y + Y.conj().transpose(-2, -1))
            Z = 0.5 * (Z + Z.conj().transpose(-2, -1))

        # 6. Un-scale back to original tensor magnitude: sqrt(A) = sqrt(Tr(A)) * Y
        sqrt_A = Y * torch.sqrt(tr_A)
        
        return sqrt_A.reshape(orig_shape)

class NS_Sqrt(nn.Module):
    def __init__(self, iters: int = 8, eps: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        orig_shape = A.shape
        D = orig_shape[-1]
        A_flat = A.reshape(-1, D, D)
        B = A_flat.size(0)
        device, dtype = A.device, A.dtype

        # 1. Sanitize & Symmetrize
        A_flat = complex_nan_to_num(A_flat)
        A_flat = 0.5 * (A_flat + A_flat.conj().transpose(-2, -1))

        tr_A = torch.real(torch.einsum('bii->b', A_flat)).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
        A_norm = A_flat / tr_A

        # 2. Diagonal Ridge Jitter
        I = torch.eye(D, dtype=dtype, device=device).unsqueeze(0).expand(B, D, D)
        splitting_vec = torch.linspace(self.eps, self.eps * 10, D, device=device, dtype=A_flat.real.dtype)
        jitter = torch.diag_embed(splitting_vec).to(dtype).unsqueeze(0).expand(B, D, D)
        A_reg = A_norm + jitter

        # 3. Frobenius Norm Normalization (Guarantees Newton-Schulz Convergence)
        norm_A = A_reg.norm(p='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        norm_A_c = norm_A.to(dtype)

        Y = A_reg / norm_A_c
        Z = I.clone()

        # 4. Coupled Newton-Schulz Iterations
        for _ in range(self.iters):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z

            # Re-enforce Hermitian symmetry at each step to suppress floating-point drift
            Y = 0.5 * (Y + Y.conj().transpose(-2, -1))
            Z = 0.5 * (Z + Z.conj().transpose(-2, -1))

        # 5. Rescale back: sqrt(A) = Y * sqrt(||A||_F)
        sqrt_A = Y * torch.sqrt(norm_A_c * tr_A)
        return sqrt_A.reshape(orig_shape)

class ExactSpectralMatrixSqrt(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        orig_shape = A.shape
        D = orig_shape[-1]
        A_flat = A.reshape(-1, D, D)
        B = A_flat.size(0)
        orig_device = A.device

        # 1. Sanitize & Hermitian symmetrization
        A_flat = complex_nan_to_num(A_flat)
        A_flat = 0.5 * (A_flat + A_flat.conj().transpose(-2, -1))

        # 2. Spectrum-splitting jitter
        splitting_vec = torch.linspace(self.eps, self.eps * 100, D, device=orig_device, dtype=A_flat.real.dtype)
        jitter = torch.diag_embed(splitting_vec).to(A.dtype).unsqueeze(0).expand(B, D, D)
        A_reg = A_flat + jitter

        A_cpu = A_reg.cpu()

        try:
            evals, evecs = torch.linalg.eigh(A_cpu)
            evals = torch.clamp(evals, min=self.eps)
            sqrt_evals = torch.diag_embed(torch.sqrt(evals)).to(A.dtype)
            sqrt_A = evecs @ sqrt_evals @ evecs.conj().transpose(-2, -1)
        except torch._C._LinAlgError:
            U, S, Vh = torch.linalg.svd(A_cpu)
            sqrt_S = torch.diag_embed(torch.sqrt(torch.clamp(S, min=self.eps))).to(A.dtype)
            sqrt_A = U @ sqrt_S @ Vh

        return sqrt_A.to(orig_device).reshape(orig_shape)

class SafeArcsin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, max_grad: float = 5.0) -> torch.Tensor:
        x_clamped = torch.clamp(x, -0.999, 0.999)
        ctx.save_for_backward(x_clamped)
        ctx.max_grad = max_grad
        return torch.asin(x_clamped)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_clamped, = ctx.saved_tensors
        min_denom = 1.0 / ctx.max_grad
        denom = torch.sqrt(1.0 - x_clamped ** 2).clamp(min=min_denom)
        
        grad_x = grad_output / denom
        return grad_x, None

def complex_nan_to_num(z: torch.Tensor, nan: float = 0.0, posinf: float = 1.0, neginf: float = -1.0) -> torch.Tensor:
    real_part = torch.nan_to_num(z.real, nan=nan, posinf=posinf, neginf=neginf)
    imag_part = torch.nan_to_num(z.imag, nan=nan, posinf=posinf, neginf=neginf)
    return torch.complex(real_part, imag_part)


class BaseInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, learn_temperature: bool = False):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # CLIP-style learnable logit scale exp(s), s initialised at log(1/temperature), clamped at 100
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)), requires_grad=True) if learn_temperature else None

    def scale(self) -> torch.Tensor:
        if self.logit_scale is None:
            return torch.tensor(1.0 / self.temperature)
        return self.logit_scale.exp().clamp(max=100.0)

    def describe(self) -> None:
        hyperparams = self.get_hyperparams()
        param_str = " | ".join([f"{k}: {v}" for k, v in hyperparams.items()])
        print(f"{self.__class__.__name__} {param_str}")

    def get_hyperparams(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "label_smoothing": self.label_smoothing,
            "learn_temperature": self.logit_scale is not None,
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

    def positive_margin(self, similarity: torch.Tensor) -> torch.Tensor:
        return similarity   # subclasses may subtract a margin from the positives (diagonal)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor, neg_image_emb: torch.Tensor = None) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)
        scale = self.scale().to(text_emb.device)

        # 1. Similarity matrix [B, B] (positives on the diagonal), optional margin on the positives
        similarity = self.positive_margin(self.compute_similarity(text_emb, image_emb))

        # 2. text->image: in-batch images, plus (optionally) every caption's hard-negative image as extra columns
        logits_t2i = similarity * scale
        sim_neg = None
        if neg_image_emb is not None:
            sim_neg = self.compute_similarity(text_emb, neg_image_emb)
            logits_t2i = torch.cat([logits_t2i, sim_neg * scale], dim=1)
        loss_t2i = self.cross_entropy(logits_t2i, labels)
        loss_i2t = self.cross_entropy((similarity * scale).T, labels)
        sym_loss = 0.5 * (loss_t2i + loss_i2t)

        # 3. Optional regularisation (uniformity, purity, shot-aware margins, ...)
        reg_loss = self.compute_regularization(text_emb, image_emb)
        if sim_neg is not None and hasattr(self, 'hard_negative_regularization'):
            reg_loss = reg_loss + self.hard_negative_regularization(text_emb, image_emb, neg_image_emb)

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
    # similarity = arcsin(0.5 + 0.5 |<t|i>|) / (pi/2) in [1/3, 1]: a monotone transform of root fidelity,
    # not the Fubini-Study distance arccos|<t|i>| (use Bures_InfoNCE for the FS/Bures angle)
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

        overlap = torch.clamp((text_flat @ image_flat.conj().T).abs(), 0.0, 1.0 - self.eps)
        overlap_scaled = 0.5 + (0.5 * overlap)
        return torch.asin(overlap_scaled) / (math.pi / 2)

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

def _fidelity(text_emb: torch.Tensor, image_emb: torch.Tensor, text_dm: bool, image_dm: bool, image_pure: bool, eps: float = 1e-12) -> torch.Tensor:
    """Exact Uhlmann-Jozsa fidelity F[i, j] between text i and image j (states normalised first).
    Statevectors -> |<t|phi>|^2; pure image -> <phi|rho|phi> = Tr(rho sigma) (what the swap test measures);
    mixed images -> (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2 by eigendecomposition in float64."""
    B = text_emb.size(0)
    def as_dm(x, dm):
        if dm:
            D = int(round(math.sqrt(x[0].numel()))); r = x.reshape(x.size(0), D, D).to(torch.complex128)
        else:
            v = x.reshape(x.size(0), -1).to(torch.complex128); v = v / v.norm(dim=1, keepdim=True).clamp(min=eps)
            r = torch.einsum('bi,bj->bij', v, v.conj())
        return r / torch.diagonal(r, dim1=-2, dim2=-1).sum(-1).real.clamp(min=eps)[:, None, None]
    if not text_dm and not image_dm:
        t = text_emb.reshape(B, -1).to(torch.complex128); t = t / t.norm(dim=1, keepdim=True).clamp(min=eps)
        p = image_emb.reshape(image_emb.size(0), -1).to(torch.complex128); p = p / p.norm(dim=1, keepdim=True).clamp(min=eps)
        return ((t @ p.conj().T).abs() ** 2).float()
    rho, sig = as_dm(text_emb, text_dm), as_dm(image_emb, image_dm)
    if image_pure or not text_dm:   # one side pure: F = Tr(rho sigma)
        return torch.einsum('aij,bji->ab', rho, sig).real.clamp(0, 1).float()
    dev = rho.device
    try:
        w, V = torch.linalg.eigh(rho)
    except RuntimeError:            # cluster env: CUDA eigh unavailable (cusolver), fall back to CPU
        w, V = torch.linalg.eigh(rho.cpu()); w, V = w.to(dev), V.to(dev)
    sq = (V * w.clamp(min=0).sqrt().unsqueeze(-2).to(V.dtype)) @ V.conj().transpose(-1, -2)
    M = sq.unsqueeze(1) @ sig.unsqueeze(0) @ sq.unsqueeze(1); M = 0.5 * (M + M.conj().transpose(-1, -2))
    try:
        ev = torch.linalg.eigvalsh(M)
    except RuntimeError:
        ev = torch.linalg.eigvalsh(M.cpu()).to(dev)
    return (ev.clamp(min=0).sqrt().sum(-1) ** 2).clamp(0, 1).float()


class UJ_InfoNCE(BaseInfoNCELoss):
    """Similarity (2/pi) arcsin(sqrt F) = 1 - Bures angle/(pi/2), F the Uhlmann-Jozsa fidelity, computed exactly
    (_fidelity). Before 2026-10-17 this used a 12-step Newton-Schulz matrix square root in complex64, which was off
    by up to 0.26 in value and ~70% in gradient for pure images."""
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, eps: float = 1e-6,
                 image_pure: bool = True, learn_temperature: bool = False):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing, learn_temperature=learn_temperature)
        self.eps = eps
        self.image_pure = image_pure

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"eps": self.eps, "fidelity": "exact", "image_pure": self.image_pure})
        return hp

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        F = _fidelity(text_emb, image_emb, text_dm=True, image_dm=True, image_pure=self.image_pure)
        return torch.asin(F.clamp(self.eps, 1 - self.eps).sqrt()) / (math.pi / 2.0)


class Bures_InfoNCE(BaseInfoNCELoss):
    """InfoNCE on the Bures angle A = arccos(sqrt F) (the Fubini-Study angle for pure states), with F the exact
    fidelity, i.e. the swap-test score for pure images. Logit = (pi/2 - A - margin*[positive]) * scale, scale learnable.
    Works for statevectors (post-selected text) and density matrices (discard text), so one geometry serves both.
    Optional regularisers: HS-distance uniformity of text states (Wang & Isola form) and, with hard negatives,
    a shot-aware margin hinge relu(z0 - Delta/sigma_N), sigma_N = sqrt((2 - F_pos^2 - F_neg^2) / N)."""
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, text_dm: bool = True, image_dm: bool = True,
                 image_pure: bool = True, learn_temperature: bool = True, margin: float = 0.0, uniformity_weight: float = 0.0,
                 uniformity_t: float = 2.0, shot_weight: float = 0.0, shot_budget: int = 1024, shot_z: float = 2.0, eps: float = 1e-4):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing, learn_temperature=learn_temperature)
        self.text_dm, self.image_dm, self.image_pure = text_dm, image_dm, image_pure
        self.margin, self.eps = margin, eps
        self.uniformity_weight, self.uniformity_t = uniformity_weight, uniformity_t
        self.shot_weight, self.shot_budget, self.shot_z = shot_weight, shot_budget, shot_z

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"margin": self.margin, "uniformity_weight": self.uniformity_weight, "uniformity_t": self.uniformity_t,
                   "shot_weight": self.shot_weight, "shot_budget": self.shot_budget, "shot_z": self.shot_z,
                   "text_dm": self.text_dm, "image_pure": self.image_pure})
        return hp

    def fidelity(self, text_emb, image_emb):
        return _fidelity(text_emb, image_emb, self.text_dm, self.image_dm, self.image_pure)

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return math.pi / 2 - torch.acos(self.fidelity(text_emb, image_emb).clamp(self.eps, 1 - self.eps).sqrt())

    def positive_margin(self, similarity: torch.Tensor) -> torch.Tensor:
        if self.margin <= 0:
            return similarity
        return similarity - self.margin * torch.eye(similarity.size(0), device=similarity.device, dtype=similarity.dtype)

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        if self.uniformity_weight <= 0 or text_emb.size(0) < 2:
            return torch.tensor(0.0, device=text_emb.device)
        # uniformity of text states in Hilbert-Schmidt distance: d^2 = P_i + P_j - 2 Tr(rho_i rho_j)
        B = text_emb.size(0)
        if self.text_dm:
            D = int(round(math.sqrt(text_emb[0].numel()))); r = text_emb.reshape(B, D, D)
            r = r / torch.diagonal(r, dim1=-2, dim2=-1).sum(-1).real.clamp(min=1e-12)[:, None, None]
            G = torch.einsum('aij,bji->ab', r, r).real
        else:
            v = text_emb.reshape(B, -1); v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-12)
            G = (v @ v.conj().T).abs() ** 2
        d2 = G.diagonal()[:, None] + G.diagonal()[None, :] - 2 * G
        mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
        return self.uniformity_weight * torch.logsumexp(-self.uniformity_t * d2[mask], dim=0) - self.uniformity_weight * math.log(mask.sum().item())

    def hard_negative_regularization(self, text_emb, image_emb, neg_image_emb) -> torch.Tensor:
        if self.shot_weight <= 0:
            return torch.tensor(0.0, device=text_emb.device)
        Fp = self.fidelity(text_emb, image_emb).diagonal()
        Fn = self.fidelity(text_emb, neg_image_emb).diagonal()
        sigma = torch.sqrt((2 - Fp ** 2 - Fn ** 2).clamp(min=1e-6) / self.shot_budget)
        return self.shot_weight * torch.relu(self.shot_z - (Fp - Fn) / sigma).mean()


class Pauli_InfoNCE(BaseInfoNCELoss):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, num_qubits: int = 3, eps: float = 1e-8, purity_weight = 0.1, spread_weight: float = 0.0):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.spread_weight = spread_weight  # same text-spreading term as FS/HS_InfoNCE (lambda_reg), off by default
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
            "spread_weight": self.spread_weight,
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

    def _spread(self, text_emb: torch.Tensor) -> torch.Tensor:
        # FS/HS_InfoNCE's regulariser on the flattened (Hilbert-Schmidt) text states: pushes sentences apart
        B = text_emb.size(0)
        text_flat = torch.nn.functional.normalize(text_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        overlap = torch.clamp((text_flat @ text_flat.conj().T).abs(), 0.0, 1.0 - 1e-7)
        logits = 1.0 - (torch.acos(overlap) / (math.pi / 2))
        mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
        return logits[mask].mean()

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        spread = self.spread_weight * self._spread(text_emb) if self.spread_weight > 0 and text_emb.size(0) > 1 \
            else torch.tensor(0.0, device=text_emb.device)
        if self.purity_weight <= 0:
            return spread

        rho_text, rho_image, _ = self._ensure_density_matrices(text_emb, image_emb)
        v_text = self._to_pauli_vector(rho_text)
        
        text_purity = self._compute_purity_from_pauli(v_text)
        text_purity_loss = (1.0 - text_purity).mean()

        loss_purity = text_purity_loss

        return self.purity_weight * loss_purity + spread





































