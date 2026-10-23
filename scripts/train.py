import torch, argparse
import pandas as pd

from modules.utils.factory import build_experiment, build_optimizer
from modules.utils.general import log_phase, setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import ContrastiveTrainer, MMEvaluator, RunManager, adaptive_optimizer

# uv run python train.py --config configs/tensor_network.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', "--config", type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument("--seed", type=int, default=None, help='Overrides the config seed')
    parser.add_argument("--run_name", type=str, default=None, help='Checkpoint directory name (default: generated from the config)')
    parser.add_argument("--epochs", type=int, default=None, help='Overrides the config epochs (smoke tests)')
    parser.add_argument("--init_from", type=str, default=None,
                        help='Initialise text+image weights from this checkpoint (.pt), e.g. a non-discard model')
    parser.add_argument("--train_frac", type=float, default=None, help='Train on a fixed random fraction of the train split (smoke tests)')
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help='Override a config value, e.g. --set qlr=3e-3 --set text.layers=3 (repeatable)')
    args = parser.parse_args()

    config, DEV, SEED = setup_exp(args.config, seed=args.seed, overrides=args.set)
    if args.init_from: config['init_from'] = args.init_from
    if args.run_name: config['run_name'] = args.run_name
    if args.epochs: config['epochs'] = args.epochs
    if args.train_frac: config['train_frac'] = args.train_frac
    DATASET = config['dataset']['name']
    BATCH_SIZE = config['batch_size']

    log_phase("Environment Initialized")
    print(f" Target Device : {DEV} | Random Seed : {SEED} | Run ID : {DATASET}")

    log_phase("Setting Up Experiment Components")
    ansatz, image_model, text_model, loss_fn = build_experiment(config, DEV)
    print(f"Model type: {config['model_type']} | CPTP: {config['text'].get('discard', False)} | Latent: {config.get('embedding_qubits', config.get('embedding_dim'))}")
    print(f"Text Model: {text_model.__class__.__name__} ansatz: {ansatz.__class__.__name__} | dims: (n: {config['text']['n']}, s: {config['text']['s']}, p: {config['text']['p']}) | layers: {config['text'].get('layers')} | curry: {config['text'].get('curry')}")
    print(f"Image Model: {image_model.__class__.__name__} img_dim: {config.get('out_dim', config.get('embedding_dim'))} | method: {config['vision']['method']} | layers: {config['vision'].get('layers')}")

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
    if args.init_from and config.get('init_by_symbol'):
        # copy parameters by symbol name from a checkpoint of a DIFFERENT circuit family (e.g. post-selected -> discard with
        # mixer/sinks): the source model is rebuilt from its own config; symbols the source lacks keep their init
        import yaml as _yaml
        from pathlib import Path as _Path
        src_cfg = _yaml.safe_load(open(_Path(args.init_from).parent / 'config.yaml')); src_cfg['num_workers'] = 0
        s_ansatz, s_img, s_txt, _ = build_experiment(src_cfg, DEV)
        s_de = DataEngine(src_cfg, s_ansatz, DEV)
        s_de.text_init(s_txt, pd.concat([s_de.compile_text('train'), s_de.compile_text('val')], ignore_index=True)); s_de.image_init(s_img)
        ck = CheckpointManager.load_model(args.init_from, s_img, s_txt, DEV)
        copied = 0
        with torch.no_grad():
            for sym, i in text_model.sym2param.items():
                j = s_txt.sym2param.get(sym)
                if j is not None:
                    text_model.params[i].copy_(s_txt.params[j].to(text_model.params[i].device)); copied += 1
                else:   # new symbols (identity-initialisable mixer, text.mixer_ansatz: cr) start at exactly 0 = identity gates
                    text_model.params[i].zero_()
        image_model.load_state_dict(s_img.state_dict())
        print(f"Initialised by symbol from {args.init_from} (epoch {ck.get('epoch')}): {copied}/{len(text_model.symbols)} text symbols copied, "
              f"{len(text_model.symbols) - copied} new (mixer/sinks) set to 0; image model copied")
        del s_txt, s_img
    elif args.init_from:
        ck = CheckpointManager.load_model(args.init_from, image_model, text_model, DEV)
        print(f"Initialised weights from {args.init_from} (epoch {ck.get('epoch')})")
    text_model.to(DEV)   # text_init creates the parameters on the CPU
    print(f"Model Parameter Counts: Image={sum(p.numel() for p in image_model.parameters()):,} | Text={sum(p.numel() for p in text_model.parameters()):,}"
          + (f" {text_model.param_breakdown()}" if hasattr(text_model, 'param_breakdown') else ""))

    log_phase("Preparing Pipeline Execution")
    train_loader = data_engine.get_loader(compiled_train, split='train')
    val_loader = data_engine.get_loader(compiled_val, split='val')

    loss_fn.to(DEV)
    optimizer, scheduler = build_optimizer(config, text_model, image_model, loss_fn)
    lr = config.get('qlr', config.get('lr'))
    warmup_epochs = config.get('warmup_epochs', 5)
    total_epochs = config['epochs']

    print(f"Optimizer: {optimizer.__class__.__name__} | LR: {lr} | Classical LR: {config.get('clr', lr)} | Weight Decay: {config.get('weight_decay', 1e-4)}")
    print(f"Scheduler: {scheduler.__class__.__name__} | Warmup Epochs: {warmup_epochs} | Total Epochs: {total_epochs} | Patience : {config.get('patience', 15)} | Delta : {config.get('min_delta', 1e-3)}")
    loss_fn.describe()
    print(f"Training for {total_epochs} epochs with batch size {BATCH_SIZE}")

    trainer = ContrastiveTrainer(image_model, text_model, optimizer, loss_fn, DEV)
    trainer.hard_negatives = bool(config.get('hard_negatives', False))   # add each caption's negative image to the softmax
    trainer.swap_negatives = bool(config.get('swap_negatives', False))   # add swapped captions to the image->text softmax (train text_path from make_swap.py --keep_all)
    if config.get('distill_from'):   # frozen post-selected teacher: match its scores (KD) and/or its states
        from modules.models.fusion.distill import Teacher
        trainer.teacher = Teacher(config['distill_from'], config, compiled_train, DEV, kd_weight=config.get('distill_kd_weight', 1.0),
                                  kd_tau=config.get('distill_tau', 1.0), state_weight=config.get('distill_state_weight', 0.0))
    evaluator = MMEvaluator(image_model, text_model, DEV)

    manager = RunManager(config, trainer, evaluator, DEV, SEED, scheduler=scheduler)
    manager.fit(train_loader, val_loader, data_engine.eval_mapper)