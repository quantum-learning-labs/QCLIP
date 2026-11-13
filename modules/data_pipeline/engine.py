import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from modules.common import load_pkl, load_obj
from modules.compilation.einsum import tn_metadata, analyse_einsum
from modules.data_pipeline.datasets import build_swap_set
from modules.symbolic.root import apply_root

class DataEngine:
    def __init__(self, config, ansatz, device):
        self.config = config
        self.ansatz = ansatz
        self.device = device
        
        ds_config = config['dataset']
        self.DatasetClass = load_obj(ds_config['class'])
        self.collate_fn = load_obj(ds_config['collate_fn'])
        self.eval_mapper = load_obj(ds_config['eval_mapper'])
        if config['vision']['method'] == 'amp':
            self.img_transform = v2.Compose([
                v2.ToImage(),               
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((64, 64))
            ])
        else:
            self.img_transform = None

    def split_config(self, split):
        return self.config['dataset'][split] if split in ('train', 'val') else self.config['dataset']['test'][split]

    def load_text(self, split):
        # derive: swap builds the subject/object swap set of the split; swap_negatives adds swapped captions to train
        entry = self.split_config(split)
        data = load_pkl(entry["text_path"])
        if entry.get('derive') == 'swap':
            data = build_swap_set(data)
        if split == 'train' and self.config.get('swap_negatives', False):
            data = build_swap_set(data, keep_all=True)
        return data

    def compile_text(self, split):
        compile_kwargs = {}
        if self.config["model_type"] == "vqc":
            compile_kwargs["curry"] = self.config["text"].get("curry", False)
            compile_kwargs["discard"] = self.config["text"].get("discard", False)
            compile_kwargs["spider"] = self.config["text"].get("spider", False)
        data = self.load_text(split)
        root = self.config.get('text', {}).get('root', 'word')   # functor.apply_root
        for col in [c for c in data.columns if c.endswith('_diagram')]:
            data[col] = [[apply_root(t, root) for t in d] if d and isinstance(d[0], list) else apply_root(d, root) for d in data[col]]
        return self.ansatz.compile_dataset(data, **compile_kwargs)

    def text_init(self, text_model, compiled_data):
        if hasattr(text_model, "from_symbols"):
            sym_cols = [c for c in compiled_data.columns if c.endswith("_symbols")]
            symbol_arr = []
            for col in sym_cols:   # skip NaN cells: a column only one split has (e.g. swapped training captions)
                symbol_arr += [v for v in compiled_data[col].tolist() if not isinstance(v, float)]
            
            sym_kwargs = {"id_init": True} if self.config["model_type"] == "vqc" else {}
            text_model.from_symbols(symbol_arr, **sym_kwargs)

        if hasattr(text_model, "from_plans"):
            cols = [col for col in compiled_data.columns if col.endswith('_einsum')]
            plan_stream = []
            for col in cols:
                plan_stream += [v for v in compiled_data[col].tolist() if not isinstance(v, float)]
            text_model.from_plans(list(plan_stream))
        
    def image_init(self, image_model):
        if self.config['vision']['method'] == 'pca':
            train_embeddings = torch.load(self.config["dataset"]["train"]["img_path"])
            val_embeddings = torch.load(self.config["dataset"]["val"]["img_path"])
            raw_tensor_stack = torch.stack(list(train_embeddings.values()) + list(val_embeddings.values())).to(self.device)
            image_model.fit_image_pca(raw_tensor_stack)

    def describe_einsum(self, model, data=None, return_metrics=False):
        if hasattr(model, "from_symbols"):
            sym_cols = [c for c in data.columns if c.endswith("_symbols")]
            einsum_cols = [c for c in data.columns if c.endswith("_einsum")]
            symbol_arr, einsum_arr = [], []
            for einsum_col, sym_col in zip(einsum_cols, sym_cols):
                symbol_arr += data[sym_col].tolist()
                einsum_arr += data[einsum_col].tolist()
            tn_arr = [(e, t) for e, t in zip(einsum_arr, symbol_arr) if not isinstance(e, float)]
            metrics = tn_metadata(tn_arr)
            print(f"Metrics for {model.__class__.__name__}:")
            print(f" (Max) Qubits: {metrics['max'][0]:.4f} | Gates: {metrics['max'][1]:.4f} | Depth: {metrics['max'][2]:.4f} | Rank: {metrics['max'][3]:.4f} | 2-Qubit Gates: {metrics['max'][4]:.4f}")
            print(f" (Avg) Qubits: {metrics['avg'][0]:.4f} | Gates: {metrics['avg'][1]:.4f} | Depth: {metrics['avg'][2]:.4f} | Rank: {metrics['avg'][3]:.4f} | 2-Qubit Gates: {metrics['avg'][4]:.4f}")
        elif hasattr(model, "einsum_expr"):
            metrics = analyse_einsum(model.einsum_expr.replace('b', ''), model.gate_arr)
            print(f"Metrics for {model.__class__.__name__}:")
            print(f" Qubits: {metrics[0]:.4f} | Gates: {metrics[1]:.4f} | Depth: {metrics[2]:.4f} | Rank: {metrics[3]:.4f} | 2-Qubit Gates: {metrics[4]:.4f}")
        else:
            raise ValueError(f"Model {model.__class__.__name__} does not support einsum description.")
        if return_metrics:
            return metrics

    def get_dataset(self, compiled_data, split='train'):
        DatasetClass = self.DatasetClass if split in ('train', 'val') else load_obj(self.split_config(split)['class'])
        return DatasetClass(compiled_data, self.split_config(split)['img_path'], image_transform=self.img_transform,
                            mode=split if split in ('train', 'val') else 'test')

    def get_loader(self, compiled_data, split='train'):
        collate_fn = self.collate_fn if split in ('train', 'val') else load_obj(self.split_config(split)['collate_fn'])
        return DataLoader(self.get_dataset(compiled_data, split=split), batch_size=self.config['batch_size'],
                          collate_fn=collate_fn, shuffle=(split == 'train'), num_workers=self.config.get("num_workers", 0),
                          pin_memory=True)
