import torch, gc, os
from torch.utils.data import DataLoader
from modules.utils.general import load_pkl
from modules.utils.analysis import tn_metadata, analyse_einsum
from torchvision.transforms import v2
from modules.utils.factory import build_dataset, load_obj

class DataEngine:
    def __init__(self, config, ansatz, device):
        self.config = config
        self.ansatz = ansatz
        self.device = device
        
        self.DatasetClass, self.collate_fn, self.eval_mapper = build_dataset(config)
        if config['vision']['method'] in ('amp', 'ttn'):   # pixel towers (TTN) need 64x64 tensors; .pt stores ignore it
            self.img_transform = v2.Compose([
                v2.ToImage(),               
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((64, 64))
            ])
        else:
            self.img_transform = None

    def compile_text(self, split):
        compile_kwargs = {}
        if self.config["model_type"] == "vqc":
            compile_kwargs["curry"] = self.config["text"].get("curry", False)
            compile_kwargs["discard"] = self.config["text"].get("discard", False)
            compile_kwargs["spider"] = self.config["text"].get("spider", False)
            postselect = tuple(self.config["text"].get("postselect_types") or ())
            if postselect and not compile_kwargs["discard"]:
                raise ValueError("text.postselect_types only applies with text.discard: true")
            compile_kwargs["postselect"] = postselect
            compile_kwargs["mixer"] = int(self.config["text"].get("mixer_layers", 0) or 0)
            compile_kwargs["sinks"] = int(self.config["text"].get("sinks", 0) or 0)
            self.ansatz.mixer_type = self.config["text"].get("mixer_ansatz", "default")   # 'cr': identity-initialisable mixer
            self.ansatz.phi_block = bool(self.config["text"].get("phi_block", False))   # Theorem A's Phi as a fixed discard block
            if compile_kwargs["sinks"] and not compile_kwargs["mixer"]:
                raise ValueError("text.sinks needs text.mixer_layers >= 1 (the sinks are coupled in by the mixer)")
            
        if split == 'train' or split == 'val':
            data = load_pkl(self.config["dataset"][split]["text_path"])
            if split == 'train' and self.config.get('train_frac'):   # fixed random subset (smoke tests); saved in the run config
                data = data.sample(frac=self.config['train_frac'], random_state=0).reset_index(drop=True)
            return self.ansatz.compile_dataset(data, **compile_kwargs)
        elif split not in ['train', 'val']:
            data = load_pkl(self.config["dataset"]["test"][split]["text_path"])
            compiled_data = self.ansatz.compile_dataset(data, **compile_kwargs)
            return compiled_data
        else:
            raise ValueError(f"Unsupported split '{split}' for text compilation. Must be 'train' or 'val'.")

    def text_init(self, text_model, compiled_data):
        if hasattr(text_model, "from_symbols"):
            sym_cols = [c for c in compiled_data.columns if c.endswith("_symbols")]
            symbol_arr = []
            for col in sym_cols:   # skip NaN cells: a column only one split has (e.g. swapped training captions)
                symbol_arr += [v for v in compiled_data[col].tolist() if not isinstance(v, float)]
            
            sym_kwargs = {"id_init": True} if self.config["model_type"] == "vqc" else {}
            text_model.from_symbols(symbol_arr, **sym_kwargs)
            self.precompute_paths(text_model, compiled_data)
            if os.environ.get('TN_GC_FREEZE', '1') == '1':
                # the compiled captions and plans are millions of long-lived Python objects: freeze them so the cyclic
                # GC stops rescanning them on every full collection (training steps allocate many temporaries)
                gc.collect(); gc.freeze()

        if hasattr(text_model, "from_plans"):
            cols = [col for col in compiled_data.columns if col.endswith('_einsum')]
            plan_stream = []
            for col in cols:
                plan_stream += [v for v in compiled_data[col].tolist() if not isinstance(v, float)]
            text_model.from_plans(list(plan_stream))
        
    def precompute_paths(self, text_model, compiled_data):
        # TN models: contraction paths (and levelwise plans) of every structure in compiled_data, searched in parallel
        if not hasattr(text_model, "precompute_paths"):
            return
        sym_cols = [c for c in compiled_data.columns if c.endswith("_symbols")]
        ein_cols = [c[:-len("_symbols")] + "_einsum" for c in sym_cols]
        pairs = [(e, s) for ec, sc in zip(ein_cols, sym_cols) for es, ss in zip(compiled_data[ec], compiled_data[sc])
                 if not isinstance(ss, float) for e, s in (zip(es, ss) if isinstance(es, list) else [(es, ss)])]
        n = text_model.precompute_paths([e for e, _ in pairs], [s for _, s in pairs])
        print(f"Contraction paths searched up front: {n} structures")

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
            first = tn_arr[0][1] if tn_arr else None
            if isinstance(first, list) and first and isinstance(first[0], list): first = first[0]
            if first and 'op_type' not in first[0]:   # plain tensors (no gates): count sentences and contraction structures
                exprs = [x for e, _ in tn_arr for x in (e if isinstance(e, list) else [e])]
                print(f"{model.__class__.__name__}: {len(exprs)} sentences | {len(set(exprs))} distinct contraction structures")
                return None
            metrics = tn_metadata(tn_arr)
            print(f"Metrics for {model.__class__.__name__}:")
            print(f" (Max) Qubits: {metrics['max'][0]:.4f} | Gates: {metrics['max'][1]:.4f} | Depth: {metrics['max'][2]:.4f} | Rank: {metrics['max'][3]:.4f} | 2-Qubit Gates: {metrics['max'][4]:.4f}")
            print(f" (Avg) Qubits: {metrics['avg'][0]:.4f} | Gates: {metrics['avg'][1]:.4f} | Depth: {metrics['avg'][2]:.4f} | Rank: {metrics['avg'][3]:.4f} | 2-Qubit Gates: {metrics['avg'][4]:.4f}")
        elif hasattr(model, "einsum_expr"):
            metrics = analyse_einsum(model.einsum_expr.replace('b', ''), model.gate_arr)
            print(f"Metrics for {model.__class__.__name__}:")
            print(f" Qubits: {metrics[0]:.4f} | Gates: {metrics[1]:.4f} | Depth: {metrics[2]:.4f} | Rank: {metrics[3]:.4f} | 2-Qubit Gates: {metrics[4]:.4f}")
        else:
            print(f"{model.__class__.__name__}: no einsum description")
            return None
        if return_metrics:
            return metrics

    def get_dataset(self, compiled_data, split='train'):
        if split == 'train' or split == 'val':
            return self.DatasetClass(
                compiled_data, 
                self.config['dataset'][split]['img_path'], 
                image_transform=self.img_transform, 
                mode=split
            )
        elif split not in ['train', 'val']:
            BenchDatasetClass = load_obj(self.config['dataset']['test'][split].get('class', None))
            if BenchDatasetClass is None:
                raise ValueError(f"No dataset class specified for test split '{split}' in config.")
            return BenchDatasetClass(
                compiled_data, 
                self.config['dataset']['test'][split]['img_path'], 
                image_transform=self.img_transform, 
                mode='test'
            )
        else:
            raise ValueError(f"Unsupported split '{split}' for loader. Must be 'train', 'val', or a test split name.")
        
    def get_loader(self, compiled_data, split='train'):
        if split == 'train' or split == 'val':
            dataset = self.get_dataset(compiled_data, split=split)
            loader = DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                collate_fn=self.collate_fn, 
                shuffle=(split=='train'), 
                num_workers=self.config.get("num_workers", 0),  # workers deadlocked runs at start (2026-10-17); data is in memory
                pin_memory=self.config['model_type'] != 'tn'   # TN batches are nested caption dicts + one small tensor: pinning only walks the dicts
            )
            return loader
        elif split not in ['train', 'val']:
            dataset = self.get_dataset(compiled_data, split=split)
            collate_fn = load_obj(self.config['dataset']['test'][split]['collate_fn'])
            loader = DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                collate_fn=collate_fn, 
                shuffle=False, 
                num_workers=self.config.get("num_workers", 0),  # workers deadlocked runs at start (2026-10-17); data is in memory
                pin_memory=self.config['model_type'] != 'tn'   # TN batches are nested caption dicts + one small tensor: pinning only walks the dicts
            )
            return loader
        else:
            raise ValueError(f"Unsupported split '{split}' for loader. Must be 'train', 'val', or a test split name.")