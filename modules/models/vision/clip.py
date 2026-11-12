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
    def __init__(self, clip_model = None, classical=True):
        super().__init__()
        self.clip_model = clip_model
        self.params = nn.ParameterList([])
        self.classical = classical

    def forward(self, img_vecs):
        if self.classical:
            return img_vecs
        else:
            return amplitude_encoding(img_vecs)