import torch.nn as nn
from itertools import count
import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
from opt_einsum import contract_expression
from modules.compilation.quantum.gates import *

class QuantumFeatureMap(nn.Module):
    def __init__(self, k: int, layers: int,  vdim: int, batch_size: int):
        super().__init__()
        self.k = k
        self.layers = layers
        self.nftr = self.k * self.layers
        self.vdim = vdim
        self.batch_size = batch_size
        self.params = nn.ParameterList([])
        self.register_buffer('_device_anchor', torch.empty(0), persistent=False)
        self.compile_fmap()

    def get_device(self):
        return self._device_anchor.device

    def reset_char(self):
        self.char_idx = count(0)

    def get_char(self):
        i = next(self.char_idx)
        chars = "acdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return chars[i] if i < len(chars) else chr(192 + i - len(chars))

    def compile_fmap(self):
        self.reset_char()
        gate_arr = [{'name': None, 'op_type': '0'} for _ in range(self.k)]
        shape_arr = [[2]] * self.k

        current_wires = [self.get_char() for _ in range(self.k)]
        input_indices = list(current_wires)

        for l in range(self.layers):
            for i in range(self.k):
                nxt = self.get_char()
                input_indices.append('b' + current_wires[i] + nxt)
                symbol = f"ftr_Ry_l{l}_q{i}"
                gate_arr.append({'name': symbol, 'op_type': 'Ry'})
                shape_arr.append([self.batch_size, 2, 2])
                current_wires[i] = nxt
        
        einsum_str = f"{','.join(input_indices)}->b{''.join(current_wires)}"
        self.gate_arr = gate_arr
        self.einsum_expr = einsum_str
        self.contraction_path = contract_expression(einsum_str, *shape_arr)

    def get_features(self, img_vecs):
        img_vecs = torch.as_tensor(img_vecs, dtype=torch.float32, device=self.get_device())
        if img_vecs.ndim == 1:
            img_vecs = img_vecs.unsqueeze(0)
        if img_vecs.shape[0] < 2:
            top_idx = torch.arange(self.nftr, device=img_vecs.device)
        else:
            top_idx = torch.var(img_vecs, dim=0).topk(self.nftr).indices
        return img_vecs.index_select(1, top_idx)

    def encode_features(self, img_vec):
        img_vec = torch.as_tensor(img_vec, dtype=torch.float32, device=self.get_device())
        features = self.get_features(img_vec).squeeze(0)
        sym2ftr = {}
        for item in self.gate_arr:
            symbol, gate = item['name'], item['op_type']
            if gate == '0':
                continue
            parts = symbol.split('_')
            layer_idx = int(parts[2][1:])
            qubit_idx = int(parts[3][1:])
            flat_idx = layer_idx * self.k + qubit_idx
            feature_tensor = features[flat_idx]
            sym2ftr[symbol] = feature_tensor
        return sym2ftr

    def forward(self, img_vecs):
        features = self.get_features(img_vecs)
        tensor_arr = []

        for item in self.gate_arr:
            symbol, gate = item['name'], item['op_type']
            if gate == '0':
                tensor_arr.append(torch.tensor([1, 0], dtype=torch.complex64, device=self.get_device()))
                continue

            parts = symbol.split('_')
            layer_idx = int(parts[2][1:])
            qubit_idx = int(parts[3][1:])

            flat_idx = layer_idx * self.k + qubit_idx
            feature_tensor = features[:, flat_idx]

            tensor_arr.append(Ry(feature_tensor))

        return self.contraction_path(*tensor_arr)
