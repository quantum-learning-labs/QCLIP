import torch, argparse
import pandas as pd

from modules.utils.factory import build_experiment
from modules.utils.general import log_phase, setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import ContrastiveTrainer, MMEvaluator, RunManager, adaptive_optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# uv run python train.py --config configs/tensor_network.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', "--config", type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument("--seed", type=int, default=None, help='Overrides the config seed')
    parser.add_argument("--run_name", type=str, default=None, help='Checkpoint directory name (default: generated from the config)')
    parser.add_argument("--epochs", type=int, default=None, help='Overrides the config epochs (smoke tests)')
    parser.add_argument("--init_from", type=str, default=None,
                        help='Initialise text+image weights from this checkpoint (.pt), e.g. a non-discard model')
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help='Override a config value, e.g. --set qlr=3e-3 --set text.layers=3 (repeatable)')
    args = parser.parse_args()

    config, DEV, SEED = setup_exp(args.config, seed=args.seed, overrides=args.set)
    if args.init_from: config['init_from'] = args.init_from
    if args.run_name: config['run_name'] = args.run_name
    if args.epochs: config['epochs'] = args.epochs
    DATASET = config['dataset']['name']
    BATCH_SIZE = config['batch_size']

    log_phase("Environment Initialized")
    print(f" Target Device : {DEV} | Random Seed : {SEED} | Run ID : {DATASET}")

    log_phase("Setting Up Experiment Components")
    ansatz, image_model, text_model, loss_fn = build_experiment(config, DEV)
    print(f"Model type: {config['model_type']} | CPTP: {config['text'].get('discard', False)} | Latent Qubits: {config['embedding_qubits']}")
    print(f"Text Model: {text_model.__class__.__name__} ansatz: {ansatz.__class__.__name__} | qubits: (n: {config['text']['n']}, s: {config['text']['s']}, p: {config['text']['p']}) | layers: {config['text']['layers']} | curry: {config['text']['curry']}")
    print(f"Image Model: {image_model.__class__.__name__} img_dim: {config['out_dim']} | method: {config['vision']['method']} | layers: {config['vision']['layers']}")

    log_phase("Compiling Symbolic Datasets")
    data_engine = DataEngine(config, ansatz, DEV)
    compiled_train = data_engine.compile_text('train')
    compiled_val = data_engine.compile_text('val')
    print(f"Datasets compiled: Train={len(compiled_train)} | Val={len(compiled_val)}")

    log_phase("Extracting Dataset Statistics")
    data_engine.describe_einsum(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.describe_einsum(image_model)

    log_phase("Initializing Model Parameters")
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)
    if args.init_from:
        ck = CheckpointManager.load_model(args.init_from, image_model, text_model, DEV)
        print(f"Initialised weights from {args.init_from} (epoch {ck.get('epoch')})")
    print(f"Model Parameter Counts: Image={sum(p.numel() for p in image_model.parameters()):,} | Text={sum(p.numel() for p in text_model.parameters()):,}")

    log_phase("Preparing Pipeline Execution")
    train_loader = data_engine.get_loader(compiled_train, split='train')
    val_loader = data_engine.get_loader(compiled_val, split='val')

    loss_fn.to(DEV)
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]   # e.g. a learnable logit scale
    quantum_params = list(text_model.parameters()) + list(image_model.params)
    classical_params = list(image_model.projector.parameters()) if config['vision']['method'] == 'mlp' else []

    # optimizer = adaptive_optimizer(text_model, image_model, base_qlr=config['qlr'], base_clr=config['clr'], weight_decay=1.8)
    optimizer = torch.optim.Adam(
        [{'params': quantum_params, 'lr': config['qlr']},
         {'params': classical_params, 'lr': config['clr']}]
        + ([{'params': loss_params, 'lr': config.get('temperature_lr', 1e-3), 'weight_decay': 0.0}] if loss_params else []), 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=config.get('weight_decay', 1e-4)
    )
    warmup_epochs = config.get('warmup_epochs', 5)
    total_epochs = config['epochs']

    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    print(f"Optimizer: {optimizer.__class__.__name__} | Quantum LR: {config['qlr']} | Classical LR: {config['clr']} | Weight Decay: {config.get('weight_decay', 1e-4)}")
    print(f"Scheduler: {scheduler.__class__.__name__} | Warmup Epochs: {warmup_epochs} | Total Epochs: {total_epochs} | Patience : {config['patience']} | Delta : {config['min_delta']}")
    loss_fn.describe()
    print(f"Training for {total_epochs} epochs with batch size {BATCH_SIZE}")

    trainer = ContrastiveTrainer(image_model, text_model, optimizer, loss_fn, DEV)
    trainer.hard_negatives = bool(config.get('hard_negatives', False))   # add each caption's negative image to the softmax
    if config.get('distill_from'):   # frozen post-selected teacher: match its scores (KD) and/or its states
        from modules.models.fusion.distill import Teacher
        trainer.teacher = Teacher(config['distill_from'], config, compiled_train, DEV, kd_weight=config.get('distill_kd_weight', 1.0),
                                  kd_tau=config.get('distill_tau', 1.0), state_weight=config.get('distill_state_weight', 0.0))
    evaluator = MMEvaluator(image_model, text_model, DEV)

    manager = RunManager(config, trainer, evaluator, DEV, SEED, scheduler=scheduler)
    manager.fit(train_loader, val_loader, data_engine.eval_mapper)