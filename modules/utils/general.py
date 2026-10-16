import torch, pickle, gc, random, logging, yaml, os, subprocess
from datetime import datetime
import numpy as np
from pathlib import Path

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            _ = test_tensor + 1
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"\n[Warning] CUDA is physically available but incompatible or broken.")
            print(f"Error detail: {e}")
            print("Falling back to CPU execution for stability...\n")
    return torch.device("cpu")

def store_pkl(data, fpathname):
    with open(fpathname, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(fpathname):
    gc.disable()
    try:
        with open(fpathname, 'rb') as f:
            data = pickle.load(f)
    finally:
        gc.enable()
    return data

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # deterministic kernels where torch has them (e.g. index_put accumulate in the text backward)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def log_phase(name: str):
    print(f"\n[{name.upper()}] " + "—" * (60 - len(name)))

def serialise_subsapce(sub_config):
    identity_keys = ['layers', 'hidden_dim', 'n', 'p', 's', 'bond_dim', 'max_order']
    tags = ['method']
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
        return 'unknown'

def setup_exp(config_path: str, seed=None):
    # must be set before cuBLAS initialises for deterministic GPU matmuls
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # discard lives under text:, old configs set it at the top level
    if 'discard' in config:
        text_discard = config.get('text', {}).get('discard')
        if text_discard != config['discard']:
            raise ValueError(f"{config_path}: top-level 'discard: {config['discard']}' is no longer read; "
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
    with open(config_arg) as f: given_cfg = yaml.safe_load(f)
    with open(saved) as f: saved_cfg = yaml.safe_load(f)
    diff = [k for k in MODEL_KEYS if given_cfg.get(k) != saved_cfg.get(k)]
    if diff:
        raise ValueError(f"--config {config_arg} differs from {saved} in {diff}; drop --config to use the checkpoint's own.")
    return config_arg

class CheckpointManager:
    @staticmethod
    def upgrade_checkpoint(old_state_dict):
        new_state_dict = {}
        for key, value in old_state_dict.items():
            if key == "params":
                for i in range(value.size(0)):
                    new_state_dict[f"params.{i}"] = value[i:i+1]
            else:
                new_state_dict[key] = value
        return new_state_dict

    @classmethod
    def load_model(cls, checkpoint_path: Path, image_model, text_model, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        text_params = checkpoint["text"]
        image_params = checkpoint["image"]

        if "params" in text_params and not any("params." in k for k in text_params):
            text_params = cls.upgrade_checkpoint(text_params)
        if "params" in image_params and not any("params." in k for k in image_params):
            image_params = cls.upgrade_checkpoint(image_params)

        missing, unexpected = text_model.load_state_dict(text_params, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Text weights in {checkpoint_path} do not match the model built from the config: "
                               f"{len(missing)} missing, {len(unexpected)} unexpected (e.g. {(missing + unexpected)[:3]})")
        image_model.load_state_dict(image_params)
        
        return checkpoint