# factory.py
from modules.compilation.quantum.ansatz import CustomV5Ansatz, CustomV6Ansatz
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS
from modules.compilation.classical.neural import MLPCompiler

from modules.models.text.einsum_quantum import VQCModel
from modules.models.text.einsum_classical import TNModel
from modules.models.text.neural_model import MLPModel

from modules.models.vision.quantum_map import QuantumFeatureMap, QFMap_CPTP
from modules.models.vision.clip import FrozenCLIP
from modules.models.vision.image_model import TTNImageModel

from modules.models.fusion.criteria import FS_InfoNCE, InfoNCE, UJ_InfoNCE, Pauli_InfoNCE, HS_InfoNCE, Bures_InfoNCE
import math
import torch.nn as nn
import importlib

def load_obj(import_str: str):
    module_path, attr_name = import_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)

def build_dataset(config: dict):
    ds_config = config['dataset']
    
    dataset_class = load_obj(ds_config['class'])
    collate_fn = load_obj(ds_config['collate_fn'])
    eval_mapper = load_obj(ds_config['eval_mapper'])
    
    return dataset_class, collate_fn, eval_mapper

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
        ansatz = CustomV5Ansatz(obmap=obmap, layers=compiler['layers'])
        
        if config['vision']['method'] == 'amp':
            image_model = FrozenCLIP(classical=False).to(device)
        else:
            image_model = QuantumFeatureMap(k=config['embedding_qubits'], 
                                            layers=config['vision']['layers'], 
                                            batch_size=config['batch_size'], 
                                            out_dim=config['out_dim'],
                                            id_init=False, 
                                            discard=config['text'].get('discard', False),
                                            method=config['vision']['method'],
                                            env_qubits=config['vision'].get('env_qubits', 0)).to(device)

        text_model = VQCModel(out_q=config['embedding_qubits'], discard=config['text'].get('discard', False),
                              renorm=bool(config['text'].get('postselect_types'))).to(device)
        discard = config['text'].get('discard', False)
        loss_name = config.get('loss', 'pauli' if discard else 'fs')
        if loss_name in ('pauli', 'uj') and not discard:
            raise ValueError(f"loss '{loss_name}' needs density matrices: set text.discard: true")
        ls, temp = config.get('label_smoothing', 0.1), config.get('temperature', 0.07)
        image_pure = not config['vision'].get('env_qubits', 0)
        if loss_name == 'bures':
            loss_fn = Bures_InfoNCE(label_smoothing=ls, temperature=temp, text_dm=discard, image_dm=discard, image_pure=image_pure,
                                    learn_temperature=config.get('learn_temperature', True), margin=config.get('margin', 0.0),
                                    uniformity_weight=config.get('uniformity_weight', 0.0), uniformity_t=config.get('uniformity_t', 2.0),
                                    shot_weight=config.get('shot_weight', 0.0), shot_budget=config.get('shot_budget', 1024),
                                    shot_z=config.get('shot_z', 2.0))
        elif loss_name == 'pauli':
            loss_fn = Pauli_InfoNCE(label_smoothing=ls, temperature=temp, num_qubits=config['embedding_qubits'],
                                    purity_weight=config.get('purity_weight', 0.1), spread_weight=config.get('spread_weight', 0.0))
        elif loss_name == 'uj':
            loss_fn = UJ_InfoNCE(label_smoothing=ls, temperature=temp, image_pure=image_pure)
        elif loss_name == 'hs':
            loss_fn = HS_InfoNCE(label_smoothing=ls, temperature=temp, lambda_reg=config.get('spread_weight', 0.1))
        elif loss_name == 'fs':
            loss_fn = FS_InfoNCE(label_smoothing=ls, temperature=temp, lambda_reg=config.get('spread_weight', 0.1))
        else:
            raise ValueError(f"Unknown loss '{loss_name}' (bures, pauli, uj, hs, fs)")
        if loss_name != 'bures' and config.get('learn_temperature'):
            loss_fn.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))
        
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
