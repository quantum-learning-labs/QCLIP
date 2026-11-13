import pandas as pd

from modules.common import log_phase
from modules.config import parse_args, setup_exp
from modules.factory import load_experiment, build_optimizer
from modules.models.fusion.trainer import ContrastiveTrainer, Teacher
from modules.models.fusion.evaluator import MMEvaluator
from modules.models.fusion.run import RunManager

# python scripts/train.py -cfg configs/final/vqc_nodiscard.yaml [--seed 0] [-cp init.pt] [--set run_name=x --set epochs=2]
if __name__ == "__main__":
    args = parse_args("Train a model from a config")
    if args.config is None:
        raise ValueError("train.py needs a config (-cfg)")
    overrides = args.set + ([f"init_from={args.checkpoint}"] if args.checkpoint else [])
    config, DEV, SEED = setup_exp(args.config, seed=args.seed, overrides=overrides)

    log_phase("Environment Initialized")
    print(f" Target Device : {DEV} | Random Seed : {SEED} | Run ID : {config['dataset']['name']}")

    log_phase("Setting Up Experiment Components")
    exp = load_experiment(config, DEV, config.get('init_from'))
    text_model, image_model = exp.text_model, exp.image_model
    vqc = config['model_type'] == 'vqc'
    if vqc:
        print(f"Model type: vqc | CPTP: {config['text'].get('discard', False)} | Latent Qubits: {config['embedding_qubits']}")
        print(f"Text Model: {text_model.__class__.__name__} ansatz: {exp.ansatz.__class__.__name__} | qubits: (n: {config['text']['n']}, s: {config['text']['s']}, p: {config['text']['p']}) | layers: {config['text']['layers']} | curry: {config['text']['curry']}")
        print(f"Image Model: {image_model.__class__.__name__} img_dim: {config['out_dim']} | method: {config['vision']['method']} | layers: {config['vision']['layers']}")
    else:
        print(f"Model type: {config['model_type']} | Text Model: {text_model.__class__.__name__} | Image Model: {image_model.__class__.__name__}")
    print(f"Datasets compiled: Train={len(exp.train)} | Val={len(exp.val)}")
    if vqc:   # circuit size metrics
        exp.data.describe_einsum(text_model, pd.concat([exp.train, exp.val], ignore_index=True))
        exp.data.describe_einsum(image_model)
    if config.get('init_from'):
        print(f"Initialised weights from {config['init_from']} (epoch {exp.state.get('epoch')})")
    print(f"Model Parameter Counts: Image={sum(p.numel() for p in image_model.parameters()):,} | Text={sum(p.numel() for p in text_model.parameters()):,}")

    log_phase("Preparing Pipeline Execution")
    train_loader = exp.data.get_loader(exp.train, split='train')
    val_loader = exp.data.get_loader(exp.val, split='val')
    optimizer, scheduler = build_optimizer(config, text_model, image_model)
    print(f"Optimizer: {optimizer.__class__.__name__} | LR: {[g['lr'] for g in optimizer.param_groups]} | Weight Decay: {config.get('weight_decay', 1e-4)}")
    if scheduler is not None:
        print(f"Scheduler: {scheduler.__class__.__name__} | Warmup Epochs: {config.get('warmup_epochs', 5)} | Total Epochs: {config['epochs']}")
    print(f"Early stopping: patience {config.get('patience', 15)} | delta {config.get('min_delta', 1e-3)}")
    print(f"Loss: {exp.loss_fn.__class__.__name__} | temperature: {exp.loss_fn.temperature} | label smoothing: {getattr(exp.loss_fn, 'label_smoothing', None)}")
    print(f"Training for {config['epochs']} epochs with batch size {config['batch_size']}")

    teacher = None
    if config.get('distill_from'):   # frozen post-selected teacher: match its in-batch scores
        teacher = Teacher(config['distill_from'], exp.train, DEV,
                          kd_weight=config.get('distill_kd_weight', 1.0), kd_tau=config.get('distill_tau', 1.0))
    trainer = ContrastiveTrainer(image_model, text_model, optimizer, exp.loss_fn, DEV,
                                 swap_negatives=bool(config.get('swap_negatives', False)), teacher=teacher)
    evaluator = MMEvaluator(image_model, text_model, DEV)

    manager = RunManager(config, trainer, evaluator, DEV, SEED, scheduler=scheduler)
    manager.fit(train_loader, val_loader, exp.data.eval_mapper)
