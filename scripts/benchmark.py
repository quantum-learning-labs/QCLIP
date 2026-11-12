from modules.common import log_phase
from modules.config import parse_args, setup_eval, git_state, write_results
from modules.factory import load_experiment
from modules.models.fusion.evaluator import MMEvaluator

# python scripts/benchmark.py -cp checkpoints/.../best.pt [-o results.json]: exact scores on every test set of the config
if __name__ == "__main__":
    args = parse_args("Exact evaluation of a checkpoint on the config's test sets")
    if args.checkpoint is None:
        raise ValueError("benchmark.py needs a checkpoint (-cp)")
    config, DEV, SEED, config_path, checkpoint_path, out_path = setup_eval(args, 'benchmark')

    log_phase("Evaluation Environment")
    print(f" Target Device   : {DEV} | Weights : {checkpoint_path} | Config : {config_path}")
    exp = load_experiment(config, DEV, checkpoint_path)
    print(f" Recovered from Epoch: {exp.state.get('epoch', 'N/A')} | Historical Loss: {exp.state.get('train_loss', 'N/A')}")

    evaluator = MMEvaluator(exp.image_model, exp.text_model, DEV)
    results, details = evaluator.benchmark(exp.data, config["dataset"]["test"])

    log_phase("Final Benchmark Scoreboard")
    print(f"{'Split & Metric Key':<35} | {'Value / Score':<15}")
    print("—" * 53)
    for key, val in results.items():
        print(f"{key:<35} | {val:.6f}")
    print("—" * 53 + "\n")

    write_results(out_path, {"checkpoint": str(checkpoint_path), "config": config_path, "epoch": exp.state.get('epoch'),
                             "seed": SEED, "discard": config['text'].get('discard', False), "git_commit": git_state(),
                             "train_git_commit": config.get('git_commit'), "results": results, "details": details})
