import argparse, time, torch, json
from pathlib import Path
import pandas as pd

# Local framework imports
from modules.utils.factory import build_experiment, load_obj
from modules.models.fusion.engine import MMEvaluator
from modules.utils.general import log_phase, setup_exp, CheckpointManager, resolve_eval_config, git_state
from modules.data_pipeline.engine import DataEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default=None, help="Config YAML (default: the config.yaml saved next to the checkpoint)")
    parser.add_argument("-cp", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated test blocks (default: all)")
    parser.add_argument("-o", "--out", type=str, default=None, help="Results JSON (default: benchmark_<checkpoint>.json next to the checkpoint); never overwritten")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = resolve_eval_config(args.config, checkpoint_path)
    out_path = Path(args.out) if args.out else checkpoint_path.parent / f"benchmark_{checkpoint_path.stem}.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")
    config, DEV, SEED = setup_exp(config_path)

    log_phase("Evaluation Environment")
    print(f" Target Device   : {DEV} | Weights : {checkpoint_path} | Config : {config_path}")

    log_phase("Instantiating Architecture")
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)

    data_engine = DataEngine(config, ansatz, DEV)
    startup, t0 = {}, time.time()   # seconds per startup phase (step-2 efficiency report)
    compiled_train = data_engine.compile_text('train')
    compiled_val = data_engine.compile_text('val')

    startup['compile_train_val'] = time.time() - t0; t0 = time.time()
    log_phase("Restoring Dynamic Parameter Spaces")
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)

    startup['text_init'] = time.time() - t0; t0 = time.time()
    log_phase("Loading Model Checkpoint Weights")
    checkpoint = CheckpointManager.load_model(checkpoint_path, image_model, text_model, DEV)
    text_model.to(DEV)
    startup['load_checkpoint'] = time.time() - t0
    print(f" Startup (s): {startup}")
    print(f" Recovered from Epoch: {checkpoint.get('epoch', 'N/A')} | Historical Loss: {checkpoint.get('train_loss', 'N/A')}")

    image_model.eval()
    text_model.eval()
    evaluator = MMEvaluator(image_model, text_model, DEV)

    log_phase("Compiling Evaluation Graph Structures")
    test_sets = config["dataset"]["test"]
    if args.only:
        test_sets = {k: v for k, v in test_sets.items() if k in args.only.split(',')}
    results, details = {}, {}
    for split_name, split_info in test_sets.items():
        t0 = time.time()
        compiled_eval = data_engine.compile_text(split_name)
        data_engine.describe_einsum(text_model, compiled_eval)
        if hasattr(text_model, 'extend_symbols'):   # word@type tensors unseen in training: seeded, untrained init
            sym_cols = [c for c in compiled_eval.columns if c.endswith('_symbols')]
            oov = text_model.extend_symbols([v for c in sym_cols for v in compiled_eval[c].tolist() if not isinstance(v, float)], seed=SEED)
            for key, val in oov.items(): results[f"{split_name}_{key}"] = val
            print(f" OOV '{split_name}': {oov}")
        eval_loader = data_engine.get_loader(compiled_eval, split=split_name)
        startup[f'prepare_{split_name}'] = time.time() - t0
        eval_fn = getattr(evaluator, split_info['task'], None)
        eval_args = (load_obj(split_info.get('eval_mapper', config['dataset']['eval_mapper'])),) if split_info['task'] == 'global_retrieval' else ()

        with torch.no_grad():
            if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start_task = time.time()
            out = eval_fn(eval_loader, *eval_args)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            task_metrics = out[0] if isinstance(out, tuple) else out
            if isinstance(out, tuple): details[split_name] = out[1]
            elapsed = time.time() - start_task
            print(f" -> Evaluated '{split_name}' task in {elapsed:.2f}s")
            for key, val in task_metrics.items(): results[f"{split_name}_{key}"] = val
            results[f"{split_name}_n"] = len(compiled_eval)
            results[f"{split_name}_eval_s"] = elapsed
            results[f"{split_name}_rows_per_s"] = len(compiled_eval) / elapsed
            if torch.cuda.is_available(): results[f"{split_name}_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30

    log_phase("Final Benchmark Scoreboard")
    print(f"{'Split & Metric Key':<35} | {'Value / Score':<15}")
    print("—" * 53)
    for key, val in results.items():
        print(f"{key:<35} | {val:.6f}")
    print("—" * 53 + "\n")

    payload = {"checkpoint": str(checkpoint_path), "config": config_path, "epoch": checkpoint.get('epoch'),
               "seed": SEED, "discard": config['text'].get('discard', False), "git_commit": git_state(),
               "train_git_commit": config.get('git_commit'), "startup_s": startup, "results": results, "details": details}
    with open(out_path, 'x') as f:
        json.dump(payload, f, indent=1, default=float)
    print(f" Results written to {out_path}")
