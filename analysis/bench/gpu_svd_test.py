# Which GPU SVD/QR/eigh backends work in this env? Prints one line per backend: ok + reconstruction error + time, or the error.
# usage: python analysis/bench/gpu_svd_test.py
import time, traceback, os
import numpy as np, torch

N = 1024
rng = np.random.default_rng(0)
A_np = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex64)
dev = torch.device('cuda')
A = torch.tensor(A_np, device=dev)
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
print('LD_PRELOAD', os.environ.get('LD_PRELOAD', ''), flush=True)


def run(name, fn):
    try:
        fn(); torch.cuda.synchronize(); t0 = time.time(); err = fn(); torch.cuda.synchronize()
        print(f'OK   {name:28s} err {err:.2e}  {1e3 * (time.time() - t0):.1f} ms', flush=True)
    except Exception as e:
        print(f'FAIL {name:28s} {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ""}', flush=True)


def t_svd():
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return float(torch.linalg.norm((U * S.to(U.dtype)) @ Vh - A) / torch.linalg.norm(A))
def t_qr():
    Q, R = torch.linalg.qr(A); return float(torch.linalg.norm(Q @ R - A) / torch.linalg.norm(A))
def t_eigh():
    H = A @ A.mH; w, V = torch.linalg.eigh(H); return float(torch.linalg.norm((V * w.to(V.dtype)) @ V.mH - H) / torch.linalg.norm(H))


for lib in ('default', 'cusolver', 'magma'):
    try:
        if lib != 'default': torch.backends.cuda.preferred_linalg_library(lib)
    except Exception as e:
        print('cannot select', lib, e); continue
    run(f'torch svd [{lib}]', t_svd); run(f'torch qr [{lib}]', t_qr); run(f'torch eigh [{lib}]', t_eigh)
torch.backends.cuda.preferred_linalg_library('default')

try:
    import cupy as cp
    Ac = cp.asarray(A_np)
    def c_svd():
        U, S, Vh = cp.linalg.svd(Ac, full_matrices=False); return float(cp.linalg.norm((U * S) @ Vh - Ac) / cp.linalg.norm(Ac))
    def c_qr():
        Q, R = cp.linalg.qr(Ac); return float(cp.linalg.norm(Q @ R - Ac) / cp.linalg.norm(Ac))
    run('cupy svd', c_svd); run('cupy qr', c_qr)
except Exception as e:
    print('FAIL cupy import', e)

try:
    from cuquantum import cutensornet as ctn
    from cuquantum.cutensornet.tensor import decompose, SVDMethod
    def q_svd():
        U, S, V = decompose('ij->ik,kj', A, method=SVDMethod())
        return float(torch.linalg.norm((U * S.to(U.dtype)) @ V - A) / torch.linalg.norm(A))
    def q_qr():
        Q, R = decompose('ij->ik,kj', A); return float(torch.linalg.norm(Q @ R - A) / torch.linalg.norm(A))
    run('cutensornet svd', q_svd); run('cutensornet qr', q_qr)
except Exception as e:
    print('FAIL cutensornet import', type(e).__name__, e)


def gram_svd():
    # fallback: Gram on GPU, eigh on CPU (float64), back to GPU
    G = (A.mH @ A).cpu().to(torch.complex128); w, V = torch.linalg.eigh(G)
    w = w.clamp_min(0).flip(0); V = V.flip(1).to(dev).to(A.dtype); S = w.sqrt().to(dev).to(torch.float32)
    U = (A @ V) / S.clamp_min(1e-30).to(A.dtype)
    return float(torch.linalg.norm((U * S.to(U.dtype)) @ V.mH - A) / torch.linalg.norm(A))
run('gram+cpu eigh', gram_svd)
