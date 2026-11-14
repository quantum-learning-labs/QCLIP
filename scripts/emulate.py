from modules.common import log_phase
from modules.config import parse_args, setup_eval, git_state, write_results
from modules.factory import load_experiment
from modules.compilation.quantum.backends import BackendManager
from modules.compilation.quantum.circuits import compile_circuits
from modules.compilation.quantum.emul import Emulator

# python scripts/emulate.py -cp checkpoints/.../best.pt --set emulate.noise=true --set emulate.profile=kingston
#   --set emulate.shots=[64,256,1024]   shot budgets per circuit (circuits compiled once); default from emulate.resolution
#   --set emulate.limit=40              first N pairs only
if __name__ == "__main__":
    args = parse_args("Shot-based emulation (Aer) of a checkpoint on emulate.set")
    config, DEV, SEED, config_path, checkpoint_path, out_path = setup_eval(args, 'emulate')
    em = config['emulate']

    log_phase("Evaluation Environment")
    print(f" Target Device   : {DEV} | Config : {config_path} | Seed : {SEED}")
    exp = load_experiment(config, DEV, checkpoint_path)
    if checkpoint_path is None:
        print(" [Warning] No checkpoint given: emulating untrained (randomly initialised) weights")
    else:
        print(f" Recovered from ({checkpoint_path.name}) Epoch: {exp.state.get('epoch', 'N/A')} | Historical Loss: {exp.state.get('train_loss', 'N/A')}")
    exp.image_model.eval()
    exp.text_model.eval()

    compiled_eval = exp.data.compile_text(em['set'])
    einsum_data = exp.data.describe_einsum(exp.text_model, compiled_eval, return_metrics=True)
    qlimit = einsum_data['max'][0] + (0 if em['measurement_method'] == 'compute_uncompute' else config['embedding_qubits'])
    backend_manager = BackendManager(config, qlimit)

    log_phase("Compiling Circuits")
    eval_dataset = exp.data.get_dataset(compiled_eval, em['set'])
    pos_circs, neg_circs = compile_circuits(eval_dataset, exp.text_model, exp.image_model, config, qlimit, em.get('limit'))
    pos_circs = backend_manager.transpile(pos_circs, batch_size=256, optimization_level=em['optimization_level'])
    neg_circs = backend_manager.transpile(neg_circs, batch_size=256, optimization_level=em['optimization_level'])

    log_phase("Executing Emulator Engine Pipeline")
    emulator = Emulator(config, backend_manager.backend)
    if emulator.dm_backend is not None:
        widths = [emulator.active_qubits(qc) for qc in pos_circs + neg_circs]
        print(f" Hybrid emulation: {sum(w <= emulator.dm_max_qubits for w in widths)}/{len(widths)} circuits as density matrices "
              f"(<= {emulator.dm_max_qubits} active qubits); widths {min(widths)}-{max(widths)}")
    max_nq = max(qc.num_qubits for qc in pos_circs + neg_circs)
    nq_out = config['embedding_qubits']
    emulator.shot_estimation(nq_out, max_nq - nq_out, resolution=em['resolution'])
    print(f" Simulating quantum states across {len(pos_circs)} pairs with {emulator.shots} shot resolution")
    num_runs = em.get('num_runs', 5)
    shots = em.get('shots')
    if shots is not None:
        by_shots = {}
        for s in (shots if isinstance(shots, list) else [shots]):
            print(f" --- {s} shots per circuit ---")
            by_shots[s] = emulator.run_experiment(pos_circs, neg_circs, shots=s, num_runs=num_runs)
            print(f" shots {s}: accuracy {by_shots[s]['accuracy']:.4f} | accepted fraction {by_shots[s]['accept_frac_mean']:.4f}")
        data = dict(by_shots[max(by_shots)])
        data['by_shots'] = by_shots
    else:
        data = emulator.run_experiment(pos_circs, neg_circs, num_runs=num_runs)

    log_phase("Benchmark Emulation Results")
    print(f" {'Metric Key':<35} | {'Value / Score':<25}")
    print(" " + "—" * 63)
    print(f"  {'Evaluated Dataset Pairs':<34} | {len(pos_circs):.6f}")
    print(f"  {'Evaluation Runs':<34} | {num_runs}")
    print(f"  {'Pipeline Cumulative Accuracy':<34} | \033[92m{data['accuracy']:.6f}\033[0m")
    if num_runs > 1:
        print(f"  {'Accuracy Std Deviation':<34} | ±{data['accuracy_std']:.6f}")
        print(f"  {'Accuracy Sample Variance':<34} | {data['accuracy_var']:.6f}")
        print(f"  {'Accuracy 95% Conf. Interval':<34} | ±{data['accuracy_ci95']:.6f}")
        print(f"  {'Pos Pair Fidelity (Mean ± Std)':<34} | {data['pos_fidelity_mean']:.4f} ± {data['pos_fidelity_std']:.4f}")
        print(f"  {'Neg Pair Fidelity (Mean ± Std)':<34} | {data['neg_fidelity_mean']:.4f} ± {data['neg_fidelity_std']:.4f}")
    print(" " + "—" * 63 + "\n")

    if out_path is not None:
        write_results(out_path, {"checkpoint": str(checkpoint_path), "config": config_path, "epoch": exp.state.get('epoch'),
                                 "seed": SEED, "discard": config['text'].get('discard', False), "git_commit": git_state(),
                                 "train_git_commit": config.get('git_commit'), "pairs": len(pos_circs), "shots": emulator.shots,
                                 "limit": em.get('limit'), "split": config['dataset'].get('split_record'), "emulate": em,
                                 "results": data})
