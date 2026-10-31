# factory.py
from modules.compilation.quantum.ansatz import CustomV5Ansatz, CustomV6Ansatz
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS
from modules.compilation.classical.neural import MLPCompiler

from modules.models.text.einsum_quantum import VQCModel
from modules.models.text.einsum_classical import TNModel
from modules.models.text.row_adam import RowAdam
from modules.models.text.neural_model import MLPModel

from modules.models.vision.quantum_map import QuantumFeatureMap, QFMap_CPTP
from modules.models.vision.clip import FrozenCLIP
from modules.models.vision.image_model import TTNImageModel

from modules.models.fusion.criteria import FS_InfoNCE, InfoNCE, UJ_InfoNCE, Pauli_InfoNCE, HS_InfoNCE, Bures_InfoNCE
import math, torch
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
        obmap = {'n': compiler['n'], 's': compiler['s'], 'p': compiler['p'], 'out': config['embedding_dim'],
                 'sem': compiler.get('word_dim', 128)}
        mps_proc = MPS(bond_dim=config['text']['bond_dim'], max_order=config['text']['max_order'], root_bond=config['text'].get('root_bond_dim'))
        ansatz = TNCompiler(obmap=obmap, decomp_fn=mps_proc, decouple=compiler.get('decouple', False),
                            orient=compiler.get('init', 'glorot') == 'norm')
        
        method = config['vision']['method']
        if method == 'clip':   # precomputed, L2-normalised CLIP ViT-B/32 vectors (embeddings.embed_images)
            image_model = FrozenCLIP(normalize=True).to(device)
        elif method == 'ttn':
            image_model = TTNImageModel(embedding_dim=config['embedding_dim']).to(device)
        else:
            raise ValueError(f"tn: vision.method must be 'clip' or 'ttn', got '{method}'")

        text_model = TNModel(out_dim=config['embedding_dim'], engine=config['text'].get('tn_engine', 'grouped'),
                             init=config['text'].get('init', 'glorot'), scale_free=config['text'].get('scale_free', False),
                             bow=config['text'].get('arch') == 'bow').to(device)
        loss_fn = InfoNCE(temperature=config.get('temperature', 0.07), label_smoothing=config.get('label_smoothing', 0.0))
        
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

def build_optimizer(config, text_model, image_model, loss_fn):
    # Adam over text + image (+ learnable loss) parameters, linear warmup then cosine (per epoch); shared by
    # scripts/train.py and tools/tn/profile_train.py so both build the identical optimiser
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]   # e.g. a learnable logit scale
    # image towers with a `params` list keep their projector in a second group; others train all their parameters
    image_params = list(image_model.params) if hasattr(image_model, 'params') else [p for p in image_model.parameters() if p.requires_grad]
    quantum_params = list(text_model.parameters()) + image_params
    classical_params = list(image_model.projector.parameters()) if config['vision']['method'] == 'mlp' else []
    lr = config.get('qlr', config.get('lr'))

    # models with packed parameter banks (TNModel) update only the rows a step used: RowAdam, same maths as Adam
    row_kw = {'row_model': text_model} if hasattr(text_model, 'pop_row_grads') else {}
    groups = [{'params': quantum_params, 'lr': lr}, {'params': classical_params, 'lr': config.get('clr', lr)}]
    # step 4 (TN): type tensors / MPS cores (banks of order >= 2) in their own group with type_lr_scale x LR and
    # type_weight_decay; decoupled_wd: AdamW-style decay for every group. Unset = the single group above.
    split = row_kw and ('type_lr_scale' in config or 'type_weight_decay' in config)
    if split:
        type_banks = [text_model._bank(b) for b, s in enumerate(text_model.bank_shapes) if len(s) > 1]
        ids = {id(p) for p in type_banks}
        groups[0]['params'] = [p for p in quantum_params if id(p) not in ids]
        groups.insert(1, {'params': type_banks, 'lr': lr * config.get('type_lr_scale', 1.0),
                          'weight_decay': config.get('type_weight_decay', config.get('weight_decay', 1e-4))})
    optimizer = (RowAdam if row_kw else torch.optim.Adam)(
        groups
        + ([{'params': loss_params, 'lr': config.get('temperature_lr', 1e-3), 'weight_decay': 0.0}] if loss_params else []),
        betas=(0.9, 0.999), eps=1e-08, weight_decay=config.get('weight_decay', 1e-4), **row_kw,
        **({'decoupled_weight_decay': True} if config.get('decoupled_wd') else {})
    )
    warmup_epochs = config.get('warmup_epochs', 5)
    total_epochs = config['epochs']
    if config.get('lr_schedule', 'cosine') == 'plateau':
        return optimizer, PlateauSchedule(optimizer, warmup_epochs, factor=config.get('lr_factor', 0.1),
                                          patience=config.get('lr_patience', 4), threshold=config.get('min_delta', 1e-3))
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    return optimizer, scheduler


class PlateauSchedule:
    """lr_schedule: plateau. Linear warmup over warmup_epochs, then constant LR, times `factor` whenever the selection
    metric (RunManager.fit passes it: needs_metric) hasn't improved by > threshold for `patience` epochs. With early
    stopping (config patience > lr_patience) a run trains until its validation metric plateaus."""
    needs_metric = True

    def __init__(self, optimizer, warmup_epochs, factor=0.1, patience=4, threshold=1e-3):
        from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
        self.warmup_epochs, self.epoch = warmup_epochs, 0
        self.warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs) if warmup_epochs else None
        self.plateau = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience,
                                         threshold=threshold, threshold_mode='abs')
        self.optimizer = optimizer

    def step(self, metric):
        self.epoch += 1
        if self.warmup is not None and self.epoch <= self.warmup_epochs:
            self.warmup.step()
        else:
            self.plateau.step(metric)

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {'epoch': self.epoch, 'warmup': self.warmup.state_dict() if self.warmup is not None else None,
                'plateau': self.plateau.state_dict()}

    def load_state_dict(self, sd):
        self.epoch = sd['epoch']
        if self.warmup is not None: self.warmup.load_state_dict(sd['warmup'])
        self.plateau.load_state_dict(sd['plateau'])
