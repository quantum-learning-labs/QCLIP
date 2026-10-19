import argparse, json
import yaml
from pathlib import Path
import pandas as pd

from modules.utils.factory import build_experiment
from modules.utils.general import log_phase, setup_exp, CheckpointManager, resolve_eval_config, git_state
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager, Emulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, default=None, help='Config YAML (default: the config.yaml saved next to the checkpoint)')
    parser.add_argument('-cp',"--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt file)")
    parser.add_argument('-l', "--limit", type=int, default=None, help="Debug Flag: Truncate execution to top N samples")
    parser.add_argument("--seed", type=int, default=None, help="Overrides the config seed (shot sampling, transpiler)")
    parser.add_argument("--shots", type=str, default=None,
                        help="Comma-separated shot budgets per circuit (e.g. 64,256,1024); circuits are compiled once and run at each")
    parser.add_argument("--eval_set", type=str, default=None, help="Overrides emulate.set (e.g. std, swap)")
    parser.add_argument("--noise", action="store_true", help="Overrides emulate.noise: hardware noise model (FakeMiami)")
    parser.add_argument("--noise_profile", type=str, default=None, choices=["backend", "honest"],
                        help="backend = Aer NoiseModel.from_backend (default); honest = scaled errors + coherent ZZ + idle decoherence")
    parser.add_argument("--num_runs", type=int, default=None, help="Overrides emulate.num_runs")
    parser.add_argument("--method", type=str, default=None, choices=["statevector", "density_matrix", "tensor_network", "matrix_product_state"],
                        help="Overrides emulate.backend (Aer method; GPU + cuStateVec when CUDA is available)")
    parser.add_argument("--aer", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra Aer backend option (YAML value, repeatable), applied last, e.g. --aer batched_shots_gpu=false")
    parser.add_argument("--profile", type=str, default=None,
                        help="Device calibration snapshot (qiskit_ibm_runtime fake backend), e.g. fake_kingston; default fake_miami")
    parser.add_argument("--honest", action="append", default=[], metavar="KEY=VALUE",
                        help="Override an honest-profile knob (scale_2q, scale_1q, scale_ro, scale_t, zz_rad, idle), repeatable")
    parser.add_argument("--circuit_batch", type=int, default=64, help="Circuits per Aer job (default 64)")
    parser.add_argument("--save_counts", action="store_true", help="Store the full counts of every circuit (first shot run)")
    parser.add_argument("--mlflow_db", type=str, default=None,
                        help="Also log this emulation to MLflow (sqlite file); on the cluster use one file per job")
    parser.add_argument("--mlflow_experiment", type=str, default=None,
                        help="MLflow experiment (default: emulation_noise with --noise, else emulation_shot_sweep)")
    parser.add_argument("-o", "--out", type=str, default=None, help="Results JSON (default: emulate_<checkpoint>.json next to the checkpoint); never overwritten")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else None
    config_path = resolve_eval_config(args.config, checkpoint_path)
    out_path = Path(args.out) if args.out else (checkpoint_path.parent / f"emulate_{checkpoint_path.stem}.json" if checkpoint_path else None)
    if out_path is not None and out_path.exists():
        raise FileExistsError(f"{out_path} already exists")
    config, DEV, SEED = setup_exp(config_path, seed=args.seed)
    if args.eval_set: config['emulate']['set'] = args.eval_set
    if args.noise: config['emulate']['noise'] = True
    if args.noise_profile: config['emulate']['noise_profile'] = args.noise_profile
    if args.num_runs: config['emulate']['num_runs'] = args.num_runs
    if args.method: config['emulate']['backend'] = args.method
    if args.honest:
        config['emulate']['honest'] = {**(config['emulate'].get('honest') or {}), **{k: yaml.safe_load(v) for k, v in (a.split('=', 1) for a in args.honest)}}
    if args.profile: config['emulate']['profile'] = args.profile
    if args.aer:
        config['emulate']['aer_options'] = {k: yaml.safe_load(v) for k, v in (a.split('=', 1) for a in args.aer)}

    log_phase("Evaluation Environment")
    print(f" Target Device   : {DEV} | Config : {config_path} | Seed : {SEED}")

    log_phase("Instantiating Architecture")
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)

    data_engine = DataEngine(config, ansatz, DEV)
    compiled_train = data_engine.compile_text('train')
    compiled_val = data_engine.compile_text('val')
    emulation_set = config['emulate']['set']
    compiled_eval = data_engine.compile_text(emulation_set)
    einsum_data = data_engine.describe_einsum(text_model, compiled_eval, return_metrics=True)
    if config['emulate']['measurement_method'] == 'compute_uncompute':
        qlimit = einsum_data['max'][0] # (einsum_data['max'][0] + einsum_data['avg'][0]) // 2
    else:
        qlimit = einsum_data['max'][0] + config['embedding_qubits']

    log_phase("Restoring Dynamic Parameter Spaces")
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)

    log_phase("Loading Model Checkpoint Weights")
    checkpoint = {}
    if checkpoint_path is not None:
        checkpoint = CheckpointManager.load_model(checkpoint_path, image_model, text_model, DEV)
        print(f" Recovered from ({checkpoint_path.name}) Epoch: {checkpoint.get('epoch', 'N/A')} | Historical Loss: {checkpoint.get('train_loss', 'N/A')}")
    else:
        print(" [Warning] No checkpoint given: emulating untrained (randomly initialised) weights")

    image_model.eval()
    text_model.eval()
    if config['emulate'].get('profile'):
        import qiskit_ibm_runtime.fake_provider as fake_provider
        cls = {c.lower(): c for c in dir(fake_provider) if c.startswith('Fake')}['fake' + config['emulate']['profile'].lower().replace('fake_', '').replace('fake', '')]
        backend_manager = BackendManager(config, qlimit, hardware_profile=getattr(fake_provider, cls)())
    else:
        backend_manager = BackendManager(config, qlimit)

    log_phase("Extracting Parameter Angle Maps")
    eval_dataset = data_engine.get_dataset(compiled_eval, emulation_set)
    pos_circs, neg_circs = backend_manager.compile_circuits(eval_dataset, text_model, image_model, args.limit, opt_lvl=config['emulate']['optimization_level'])

    # 8. Execution and metric compiling
    log_phase("Executing Emulator Engine Pipeline")
    emulator = Emulator(config, backend_manager.backend)
    emulator.save_counts = args.save_counts
    max_nq = max(qc.num_qubits for qc in pos_circs + neg_circs)
    nq_out = config['embedding_qubits']
    emulator.shot_estimation(nq_out, max_nq - nq_out, resolution=config['emulate']['resolution'])
    print(f" Simulating quantum states across {len(pos_circs)} pairs with {emulator.shots} shot resolution")
    num_runs = config['emulate'].get('num_runs', 5)
    if args.shots:
        by_shots = {}
        for s in [int(x) for x in args.shots.split(',')]:
            print(f" --- {s} shots per circuit ---")
            by_shots[s] = emulator.run_experiment(pos_circs, neg_circs, batch_size=args.circuit_batch, shots=s, num_runs=num_runs)
            print(f" shots {s}: accuracy {by_shots[s]['accuracy']:.4f} | accepted fraction {by_shots[s]['accept_frac_mean']:.4f}")
        data = dict(by_shots[max(by_shots)]); data['by_shots'] = by_shots
    else:
        data = emulator.run_experiment(pos_circs, neg_circs, batch_size=args.circuit_batch, num_runs=num_runs)

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

    if args.mlflow_db:   # same schema as the central importer (qclip_snapshots/tools/central_import.py)
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{Path(args.mlflow_db).resolve()}")
        em = config['emulate']
        mlflow.set_experiment(args.mlflow_experiment or ('emulation_noise' if em.get('noise') else 'emulation_shot_sweep'))
        name = out_path.stem if out_path is not None else Path(args.checkpoint).parent.name
        with mlflow.start_run(run_name=name, tags={'source_key': f"emu:direct:{name}", 'analysis': 'E6 emulation',
                                                   'step_meaning': 'shots per circuit (total, incl. rejected post-selection shots)'}):
            mlflow.log_params({'checkpoint': str(checkpoint_path), 'trained_method': 'discard' if config['text'].get('discard') else 'nodiscard',
                               'seed': SEED, 'pairs': len(pos_circs), 'eval_set': em.get('set'), 'noise': bool(em.get('noise')),
                               'noise_profile': em.get('noise_profile', 'backend') if em.get('noise') else 'none', 'aer_method': em.get('backend'),
                               'measurement_method': em.get('measurement_method'), 'git_commit': git_state(), 'honest_knobs': str(em.get('honest') or 'defaults')})
            for s, r in (data.get('by_shots') or {emulator.shots: data}).items():
                for k, v in (('accuracy', r['accuracy']), ('accuracy_std_over_shot_runs', r.get('accuracy_std')),
                             ('accept_fraction', r.get('accept_frac_mean')), ('pos_fidelity_mean', r.get('pos_fidelity_mean')),
                             ('neg_fidelity_mean', r.get('neg_fidelity_mean'))):
                    if v is not None: mlflow.log_metric(k, float(v), step=int(s))

    if out_path is not None:
        payload = {"checkpoint": str(checkpoint_path), "config": config_path, "epoch": checkpoint.get('epoch'),
                   "seed": SEED, "discard": config['text'].get('discard', False), "git_commit": git_state(),
                   "train_git_commit": config.get('git_commit'), "pairs": len(pos_circs), "shots": emulator.shots,
                   "limit": args.limit, "emulate": config['emulate'], "results": data}
        with open(out_path, 'x') as f:
            json.dump(payload, f, indent=1, default=float)
        print(f" Results written to {out_path}")

if __name__ == "__main__":
    main()
