from types import SimpleNamespace
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from modules.compilation.quantum.ansatz import Ansatz
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS
from modules.compilation.classical.neural import MLPCompiler

from modules.models.text.einsum_quantum import VQCModel
from modules.models.text.einsum_classical import TNModel
from modules.models.text.neural_model import MLPModel

from modules.models.vision.quantum_map import QuantumFeatureMap
from modules.models.vision.clip import FrozenCLIP
from modules.models.vision.image_model import TTNImageModel

from modules.models.fusion.criteria import FS_InfoNCE, InfoNCE, UJ_InfoNCE, Pauli_InfoNCE, HS_InfoNCE
from modules.models.fusion.run import CheckpointManager
from modules.data_pipeline.engine import DataEngine

def build_loss(config):
    discard = config['text'].get('discard', False)
    loss_name = config.get('loss', 'pauli' if discard else 'fs')
    if loss_name in ('pauli', 'uj') and not discard:
        raise ValueError(f"loss '{loss_name}' needs density matrices: set text.discard: true")
    ls, temp = config.get('label_smoothing', 0.1), config.get('temperature', 0.07)
    if loss_name == 'pauli':
        return Pauli_InfoNCE(label_smoothing=ls, temperature=temp, num_qubits=config['embedding_qubits'],
                             purity_weight=config.get('purity_weight', 0.1))
    if loss_name == 'uj':
        return UJ_InfoNCE(label_smoothing=ls, temperature=temp)
    if loss_name == 'hs':
        return HS_InfoNCE(label_smoothing=ls, temperature=temp)
    if loss_name == 'fs':
        return FS_InfoNCE(label_smoothing=ls, temperature=temp)
    raise ValueError(f"Unknown loss '{loss_name}' (pauli, uj, hs, fs)")

def build_experiment(config, device):
    model_type = config['model_type']
    
    if model_type == 'tn':
        compiler = config['text']
        obmap = {'n': compiler['n'], 's': compiler['s'], 'p': compiler['p'], 'out': config['embedding_dim']}
        mps_proc = MPS(bond_dim=config['text']['bond_dim'], max_order=config['text']['max_order'])
        ansatz = TNCompiler(obmap=obmap, decomp_fn=mps_proc)
        
        if config['vision']['method'] == 'amp':
            image_model = FrozenCLIP().to(device)
        else:
            image_model = TTNImageModel(embedding_dim=config['embedding_dim']).to(device)

        text_model = TNModel(out_dim=config['embedding_dim']).to(device)
        loss_fn = InfoNCE()
        
    elif model_type == "vqc":
        compiler = config['text']
        obmap = {'n': compiler['n'], 's': compiler['s'], 'p': compiler['p'], 'out': config['embedding_qubits']}
        ansatz = Ansatz(obmap=obmap, layers=compiler['layers'])
        
        if config['vision']['method'] == 'amp':
            image_model = FrozenCLIP(classical=False).to(device)
        else:
            image_model = QuantumFeatureMap(k=config['embedding_qubits'], 
                                            layers=config['vision']['layers'], 
                                            batch_size=config['batch_size'], 
                                            out_dim=config['out_dim'],
                                            id_init=False, 
                                            discard=config['text'].get('discard', False),
                                            method=config['vision']['method']).to(device)

        text_model = VQCModel(out_q=config['embedding_qubits'], discard=config['text'].get('discard', False)).to(device)
        loss_fn = build_loss(config)
        
    elif model_type == "mlp":
        ansatz = MLPCompiler()
        if config['vision']['method'] == 'amp':
            image_model = FrozenCLIP().to(device)
        else:
            image_model = TTNImageModel(embedding_dim=config['embedding_dim']).to(device)
        text_model = MLPModel(hdim=config['text']['hidden_dim'], out_dim=config['embedding_dim']).to(device)
        loss_fn = InfoNCE()
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return ansatz, image_model, text_model, loss_fn


def load_experiment(config, device, checkpoint=None):
    # models, loss and data of a config with every symbol of train + val initialised, then the checkpoint's weights if given
    ansatz, image_model, text_model, loss_fn = build_experiment(config, device)
    data = DataEngine(config, ansatz, device)
    train, val = data.compile_text('train'), data.compile_text('val')
    data.text_init(text_model, pd.concat([train, val], ignore_index=True))
    data.image_init(image_model)
    state = CheckpointManager.load_model(checkpoint, image_model, text_model, device) if checkpoint else {}
    return SimpleNamespace(ansatz=ansatz, image_model=image_model, text_model=text_model, loss_fn=loss_fn,
                           data=data, train=train, val=val, state=state)

def build_optimizer(config, text_model, image_model):
    # Adam (circuit angles at qlr, classical image projector at clr), linear warm-up then cosine decay
    quantum_params = list(text_model.parameters()) + list(image_model.params)
    classical_params = list(image_model.projector.parameters()) if config['vision']['method'] == 'mlp' else []
    optimizer = torch.optim.Adam(
        [{'params': quantum_params, 'lr': config['qlr']},
         {'params': classical_params, 'lr': config['clr']}],
        betas=(0.9, 0.999), eps=1e-08, weight_decay=config.get('weight_decay', 1e-4)
    )
    warmup_epochs = config.get('warmup_epochs', 5)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    return optimizer, scheduler
