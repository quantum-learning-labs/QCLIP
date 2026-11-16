import torch, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def amplitude_encoding(vector):
    is_numpy = isinstance(vector, np.ndarray)
    is_1d = (vector.ndim == 1)

    if is_numpy: vector = torch.from_numpy(vector)
    vector = vector.to(torch.float64)
    device = vector.device
    if is_1d: vector = vector.unsqueeze(0)

    batch_size, dim = vector.shape

    num_qubits = math.ceil(math.log2(dim))
    target_dim = 1 << num_qubits

    if dim < target_dim:
        vector = F.pad(vector, (0, target_dim - dim), value=0.0)

    norm = torch.linalg.vector_norm(vector, ord=2, dim=-1, keepdim=True)

    is_zero = (norm == 0) 
    safe_norm = torch.where(is_zero, torch.ones_like(norm), norm)
    state_vector = vector / safe_norm

    state_vector[:, 0] = torch.where(is_zero.squeeze(-1), 
                                     torch.tensor(1.0, dtype=torch.float64, device=device), 
                                     state_vector[:, 0])
    
    final_norm = torch.linalg.vector_norm(state_vector, ord=2, dim=-1, keepdim=True)
    state_vector = state_vector / final_norm
    
    if is_1d: state_vector = state_vector.squeeze(0)
    if is_numpy: state_vector = state_vector.detach().cpu().numpy()

    return state_vector

class FrozenCLIP(nn.Module):
    # the CLIP image vector itself, no trained parameters. pca_dim (< 512): projected onto the top pca_dim principal
    # components of the training images (fit_pca; centred, fixed, saved in the checkpoint as buffers). classical=False:
    # amplitude-encoded as a state on log2(dim) qubits; classical=True: the vector itself
    def __init__(self, clip_model = None, classical=True, pca_dim=None, in_dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.params = nn.ParameterList([])
        self.classical = classical
        self.pca_dim = pca_dim if pca_dim is not None and pca_dim < in_dim else None
        if self.pca_dim is not None:
            self.register_buffer('pca_mean', torch.zeros(in_dim))
            self.register_buffer('pca_components', torch.zeros(self.pca_dim, in_dim))

    @torch.no_grad()
    def fit_pca(self, vectors):
        # exact PCA of the training image vectors (float64 eigh on CPU: CUDA eigh is broken in our envs); each component's
        # sign is fixed so its largest-magnitude entry is positive, so the projection is deterministic
        x = vectors.detach().to('cpu', torch.float64)
        mean = x.mean(0)
        evals, evecs = torch.linalg.eigh((x - mean).T @ (x - mean) / max(len(x) - 1, 1))
        comps = evecs[:, evals.argsort(descending=True)[:self.pca_dim]].T
        idx = comps.abs().argmax(dim=1)
        comps = comps * torch.sign(comps[torch.arange(len(comps)), idx]).unsqueeze(1)
        self.pca_mean.copy_(mean.to(self.pca_mean.dtype))
        self.pca_components.copy_(comps.to(self.pca_components.dtype))
        return (evals.sort(descending=True).values[:self.pca_dim].sum() / evals.sum()).item()   # explained variance

    def forward(self, img_vecs):
        if self.pca_dim is not None:
            img_vecs = (img_vecs - self.pca_mean.to(img_vecs.dtype)) @ self.pca_components.to(img_vecs.dtype).T
        if self.classical:
            return img_vecs
        else:
            return amplitude_encoding(img_vecs)