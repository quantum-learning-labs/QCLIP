import torch, pickle, gc, random, importlib
import numpy as np

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

def log_phase(name: str):
    print(f"\n[{name.upper()}] " + "—" * (60 - len(name)))

# import paths of objects that moved, as saved in the configs of older checkpoints
MOVED = {
    'modules.models.fusion.engine.svo_mapper': 'modules.data_pipeline.datasets.svo_mapper',
    'modules.models.fusion.engine.aro_mapper': 'modules.data_pipeline.datasets.aro_mapper',
    'modules.models.fusion.engine.mscoco_mapper': 'modules.data_pipeline.datasets.mscoco_mapper',
}

def load_obj(import_str: str):
    module_path, attr_name = MOVED.get(import_str, import_str).rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
