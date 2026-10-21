# Numerical checks of the discard losses (CPU, double precision references).
import torch, math, sys
sys.path.insert(0, '.')
from modules.models.fusion.criteria import UJ_InfoNCE, HS_InfoNCE, Pauli_InfoNCE
torch.manual_seed(0); d = 8; B = 64
def rand_mixed(B, rank):
    X = torch.randn(B, d, rank, dtype=torch.complex128); r = X @ X.conj().transpose(1, 2)
    return r / r.diagonal(dim1=1, dim2=2).sum(-1).real[:, None, None]
def rand_pure(B):
    v = torch.randn(B, d, dtype=torch.complex128); v = v / v.norm(dim=1, keepdim=True); return torch.einsum('bi,bj->bij', v, v.conj())
uj = UJ_InfoNCE(label_smoothing=0.1, temperature=0.07); hs = HS_InfoNCE(label_smoothing=0.1, temperature=0.07); pa = Pauli_InfoNCE(label_smoothing=0.1, temperature=0.07, purity_weight=0.0)
for rank in (1, 2, 4, 8):
    rho, sig = rand_mixed(B, rank), rand_pure(B)
    P = torch.einsum('bij,bji->b', rho, rho).real
    exact_F = torch.einsum('aij,bji->ab', rho, sig).real            # pure sigma: Uhlmann fidelity = Tr(rho sigma)
    exact = torch.asin(exact_F.clamp(0, 1).sqrt()) / (math.pi / 2)
    got = uj.compute_similarity(rho.to(torch.complex64), sig.to(torch.complex64)).double()
    # gradient check on one pair: d similarity / d (real part of a rho entry) via autograd on both
    r = rho[:4].clone().to(torch.complex64).requires_grad_(True); s4 = sig[:4].to(torch.complex64)
    g_got = torch.autograd.grad(uj.compute_similarity(r, s4).diagonal().sum(), r)[0]
    r2 = rho[:4].clone().requires_grad_(True)
    g_ex = torch.autograd.grad((torch.asin(torch.einsum('bij,bji->b', r2, sig[:4]).real.clamp(0, 1).sqrt()) / (math.pi / 2)).sum(), r2)[0]
    rel_g = ((g_got.double() - g_ex).abs().max() / g_ex.abs().max()).item()
    err = (got - exact).abs()
    # ranking agreement for image choice (fix text, compare two images) and text choice (fix image, two texts)
    tr = exact_F; idx = torch.randperm(B)
    img_choice = ((got - got[:, idx]) * (tr - tr[:, idx]) >= 0).double().mean().item()
    pcos = pa.compute_similarity(rho.to(torch.complex64), sig.to(torch.complex64)).double()
    txt_choice_pauli = ((pcos - pcos[idx]) * (tr - tr[idx]) >= 0).double().mean().item()
    img_choice_pauli = ((pcos - pcos[:, idx]) * (tr - tr[:, idx]) >= 0).double().mean().item()
    nhs = hs.compute_similarity(rho.to(torch.complex64), sig.to(torch.complex64)).double()
    txt_choice_hs = ((nhs - nhs[idx]) * (tr - tr[idx]) >= 0).double().mean().item()
    print(f"rank {rank}: median purity {P.median():.3f} | UJ vs exact: max abs err {err.max():.4f}, median {err.median():.5f}; "
          f"grad rel err {rel_g:.3f} | ranking agreement with Tr(rho sigma): UJ image-choice {img_choice:.3f}; "
          f"Pauli image-choice {img_choice_pauli:.3f}, text-choice {txt_choice_pauli:.3f}; normHS text-choice {txt_choice_hs:.3f}")
# Pauli cosine identity: cos = (d Tr - 1)/sqrt((d P_r - 1)(d P_s - 1))
rho, sig = rand_mixed(B, 3), rand_pure(B)
pc = pa.compute_similarity(rho.to(torch.complex64), sig.to(torch.complex64)).double()
Tr = torch.einsum('aij,bji->ab', rho, sig).real; Pr = torch.einsum('bij,bji->b', rho, rho).real
pred = (d * Tr - 1) / torch.sqrt((d * Pr[:, None] - 1) * (d - 1))
print(f"Pauli cosine identity max abs err {(pc - pred).abs().max():.2e}")
