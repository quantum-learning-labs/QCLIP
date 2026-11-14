import argparse, json, logging, os, subprocess, yaml
from datetime import datetime
from pathlib import Path
from modules.common import get_device, set_seed

def parse_args(description: str):
    # the one command line of train.py, benchmark.py and emulate.py; everything else is a config key (--set)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-cfg', '--config', type=str, default=None,
                        help='Config YAML (evaluation default: the config.yaml saved next to the checkpoint)')
    parser.add_argument('-cp', '--checkpoint', type=str, default=None,
                        help='Weights (.pt): evaluated by benchmark/emulate, the starting point of train')
    parser.add_argument('-o', '--out', type=str, default=None,
                        help='Results JSON (default: <script>_<checkpoint>.json next to the checkpoint); never overwritten')
    parser.add_argument('--seed', type=int, default=None, help='Overrides the config seed')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a config value (repeatable), e.g. --set text.layers=3 --set emulate.noise=true')
    return parser.parse_args()

def serialise_subsapce(sub_config):
    identity_keys = ['layers', 'hidden_dim', 'n', 'p', 's', 'bond_dim', 'max_order']
    if 'method' in sub_config:
        if sub_config['method'] == 'amp':
            identity_keys = ['hidden_dim', 'n', 'p', 's', 'bond_dim', 'max_order']
        parts = [f"{sub_config['method']}"]
    elif 'curry' in sub_config and sub_config['curry']:
        parts = ['cur']
    else:
        parts = []
    for key in identity_keys:
        if key in sub_config:
            val = sub_config[key]
            if type(val) is int and not isinstance(val, bool):
                label = key[:3] if len(key) > 3 else key
                parts.append(f"{label}{val}")
    return "_".join(parts)

def gen_id(config):
    q = str(config['embedding_qubits'])
    if config.get('text', {}).get('discard', False):
        q += 'd'
    else:
        q += 'p'
    t_str = f"t_{serialise_subsapce(config['text'])}"
    v_str = f"v_{serialise_subsapce(config['vision'])}"
    timestamp = datetime.now().strftime('%m%d_%H%M')
    return f"q{q}__{t_str}__{v_str}__{timestamp}"

def git_state():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout.strip()
        return commit + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        # code copies without .git (e.g. on the HPC cluster) pass the commit in the environment
        return os.environ.get('QCLIP_GIT_COMMIT', 'unknown')

def apply_overrides(config, overrides):
    # --set "a.b=value" -> config['a']['b'] = yaml.safe_load(value); missing sections are created (e.g. emulate.honest)
    for item in overrides or []:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f"--set expects key=value, got '{item}'")
        *parents, leaf = key.split('.')
        node = config
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise KeyError(f"--set {key}: '{p}' is not a config section")
        node[leaf] = yaml.safe_load(value)
    if overrides:
        config['overrides'] = list(overrides)
    return config

def setup_exp(config_path: str, seed=None, overrides=None):
    # must be set before cuBLAS initialises for deterministic GPU matmuls
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    apply_overrides(config, overrides)

    # the readout is text.discard; a top-level discard key is not read
    if 'discard' in config:
        text_discard = config.get('text', {}).get('discard')
        if text_discard != config['discard']:
            raise ValueError(f"{config_path}: top-level 'discard: {config['discard']}' is not read; "
                             f"set text.discard (currently {text_discard}) instead.")

    device = get_device()
    if seed is None:
        seed = config.get("seed", int.from_bytes(os.urandom(4), "big"))
    config["seed"] = seed  # saved with the checkpoint config, read by the emulator
    set_seed(seed)
    
    return config, device, seed

# keys that define the model: an eval config must agree with the training config on these
MODEL_KEYS = ['model_type', 'embedding_qubits', 'out_dim', 'text', 'vision']

def resolve_eval_config(config_arg, checkpoint_path):
    saved = Path(checkpoint_path).parent / 'config.yaml' if checkpoint_path is not None else None
    if saved is None or not saved.exists():
        if config_arg is None:
            raise ValueError("No --config given and no config.yaml next to the checkpoint.")
        return config_arg
    if config_arg is None:
        return str(saved)
    with open(config_arg) as f:
        given_cfg = yaml.safe_load(f)
    with open(saved) as f:
        saved_cfg = yaml.safe_load(f)
    diff = [k for k in MODEL_KEYS if given_cfg.get(k) != saved_cfg.get(k)]
    if diff:
        raise ValueError(f"--config {config_arg} differs from {saved} in {diff}; drop --config to use the checkpoint's own.")
    return config_arg

def setup_eval(args, script: str):
    # benchmark/emulate: the checkpoint's config (or a matching --config), and an output file that does not exist yet
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    config_path = resolve_eval_config(args.config, checkpoint)
    out = Path(args.out) if args.out else (checkpoint.parent / f"{script}_{checkpoint.stem}.json" if checkpoint else None)
    if out is not None and out.exists():
        raise FileExistsError(f"{out} already exists")
    config, device, seed = setup_exp(config_path, seed=args.seed, overrides=args.set)
    return config, device, seed, config_path, checkpoint, out

def write_results(out, payload):
    with open(out, 'x') as f:
        json.dump(payload, f, indent=1, default=float)
    print(f" Results written to {out}")
