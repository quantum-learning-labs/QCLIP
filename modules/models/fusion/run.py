import json
import torch, yaml, mlflow, time, socket
from pathlib import Path
from tqdm import tqdm
from modules.common import log_phase
from modules.config import gen_id, git_state

class RunManager:
    def __init__(self, config, trainer, evaluator, device, seed, scheduler=None):
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator
        self.device = device
        self.seed = seed
        self.scheduler = scheduler

        self.dataset_name = config['dataset']['name']
        self.model_type = config['model_type']
        self.run_name = config.get('run_name') or gen_id(config)
        self.config['git_commit'] = git_state()

        self.checkpoint_dir = Path(f"./checkpoints/{self.dataset_name}/{self.model_type}/{self.run_name}")
        self.checkpoint_path = self.checkpoint_dir / f"last.pt"
        self.best_checkpoint_path = self.checkpoint_dir / f"best.pt"

        self.eval_tasks = config['dataset']['val'].get('diagnostics', ['global_retrieval'])

    def _setup_environment(self):
        if self.checkpoint_dir.exists():
            raise FileExistsError(f"{self.checkpoint_dir} already exists; choose another run_name")
        self.checkpoint_dir.mkdir(parents=True)
        
        with open(self.checkpoint_dir / f"config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        mlf_db_path = Path.cwd() / self.config.get('mlflow_db', f"mlf_dbs/{self.dataset_name}.db")
        mlf_db_path.parent.mkdir(parents=True, exist_ok=True)

        mlflow.pytorch.autolog(log_models=False)
        mlflow.set_tracking_uri(f"sqlite:///{mlf_db_path}")
        mlflow.set_experiment(self.dataset_name+'_train')

    def _save_checkpoint(self, epoch, loss, metrics, is_best=False):
        payload = {
            "image": self.trainer.image_model.state_dict(),
            "text": self.trainer.text_model.state_dict(),
            "epoch": epoch,
            "train_loss": loss,
            "val_metrics": metrics,
            "text_symbols": list(getattr(self.trainer.text_model, 'symbols', [])),   # names of params.<i> (VQC)
        }
        torch.save(payload, self.checkpoint_path)
        if is_best:
            torch.save(payload, self.best_checkpoint_path)

    def _save_snapshot(self, name, epoch, loss, metrics):
        torch.save({"image": self.trainer.image_model.state_dict(), "text": self.trainer.text_model.state_dict(),
                    "epoch": epoch, "train_loss": loss, "val_metrics": metrics,
                    "text_symbols": list(getattr(self.trainer.text_model, 'symbols', []))}, self.checkpoint_dir / name)

    def fit(self, train_loader, val_loader, eval_mapper):
        self._setup_environment()
        hostname = socket.gethostname()
        log_phase(f"Model #{self.run_name}: optimization started on \"{hostname}\"...")
        
        best_metric_value = float("-inf")
        target_metric = "i2tR1" if 'global_retrieval' in self.eval_tasks else 'hard_neg_acc'

        patience = self.config.get('patience', 15)
        min_delta = self.config.get('min_delta', 1e-3)
        patience_counter = 0
        stopped_epoch = 0

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_params({
                "epochs": self.config['epochs'],
                "batch_size": self.config['batch_size'],
                "learning_rate_quantum": self.config['qlr'],
                "learning_rate_classical": self.config['clr'],
                "embedding_qubits": self.config.get('embedding_qubits', 0),
                "temperature_parameter": self.trainer.loss_fn.temperature,
                "device_target": str(self.device),
                "seed": self.seed,
                "text_tower": type(self.trainer.text_model).__name__,
                "image_tower": type(self.trainer.image_model).__name__,
                "execution_host": hostname,
                "model_path": str(self.checkpoint_path),
                "early_stopping_patience": patience,
                "early_stopping_min_delta": min_delta,
                "discard": self.config['text'].get('discard', False),
                "loss": type(self.trainer.loss_fn).__name__,
                "git_commit": self.config['git_commit'],
                "label_smoothing": getattr(self.trainer.loss_fn, 'label_smoothing', None),
                "weight_decay": self.config.get('weight_decay', 1e-4),
                "warmup_epochs": self.config.get('warmup_epochs', 5),
                "purity_weight": getattr(self.trainer.loss_fn, 'purity_weight', None),
                "text_layers": self.config['text']['layers'],
                "vision_layers": self.config['vision']['layers'],
                "overrides": " ".join(self.config.get('overrides', [])) or "none",
                "init_from": self.config.get('init_from') or "none",
                "swap_negatives": bool(self.config.get('swap_negatives', False)),
                "distill_from": self.config.get('distill_from') or "none",
                "distill_kd_weight": self.config.get('distill_kd_weight', 1.0) if self.config.get('distill_from') else 0.0,
                "distill_tau": self.config.get('distill_tau', 1.0),
            })

            checkpoint_every = self.config.get('checkpoint_every')   # extra per-epoch snapshots (training dynamics)
            if checkpoint_every:
                self._save_snapshot("epoch_init.pt", -1, None, {})
            epoch_pbar = tqdm(range(self.config['epochs']), desc="Training Pipeline", unit="epoch")
            current_lr = self.config.get('qlr', 1e-3)
            
            for epoch in epoch_pbar:
                start_time = time.time()
                loss = self.trainer.train_epoch(train_loader, eval_mapper)
                elapsed_time = time.time() - start_time

                grad_norm_dict = self.trainer.grad_norm_by_type()

                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)

                metrics = self.evaluator.eval_set(
                    dataloader=val_loader,
                    tasks=self.eval_tasks,
                    eval_mapper=eval_mapper
                )

                mlflow.log_metrics(metrics, step=epoch)
                for k, v in grad_norm_dict.items():
                    mlflow.log_metric(f"gradient_norm_{k}", v, step=epoch)
                metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                tqdm.write(f"\nEpoch {epoch:02d} | Loss: {loss:.4f} | Gradient Norms: {', '.join([f'{k}: {v:.4f}' for k, v in grad_norm_dict.items()])} | {metrics_str} | Time: {elapsed_time:.1f}s")

                current_metric = metrics.get(target_metric, 0.0)
                if current_metric > (best_metric_value + min_delta):
                    best_metric_value = current_metric
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False

                self._save_checkpoint(epoch, loss, metrics, is_best=is_best)
                mlflow.log_metric("train_loss", loss, step=epoch)
                if checkpoint_every and epoch % checkpoint_every == 0:
                    self._save_snapshot(f"epoch_{epoch:03d}.pt", epoch, loss, metrics)

                if patience_counter >= patience:
                    stopped_epoch = epoch
                    tqdm.write(
                        f"\n[EARLY STOPPING TRIGGERED] Metric '{target_metric}' failed to improve by > {min_delta} "
                        f"for {patience} consecutive epochs. Best Score: {best_metric_value:.4f} (at epoch {epoch - patience})."
                    )
                    mlflow.log_metric("early_stopped_epoch", stopped_epoch)
                    break
                
        log_phase("Experiment Run Concluded")

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

    @staticmethod
    def params_by_name(text_params, names, text_model, checkpoint_path):
        if set(names) != set(text_model.symbols):
            raise RuntimeError(f"{checkpoint_path}: its words differ from the model's: {len(set(text_model.symbols) - set(names))} "
                               f"missing, {len(set(names) - set(text_model.symbols))} extra (e.g. {sorted(set(names) ^ set(text_model.symbols))[:3]})")
        old_index = {name: i for i, name in enumerate(names)}
        remapped = {k: v for k, v in text_params.items() if not k.startswith('params.')}
        for name, j in text_model.sym2param.items():
            remapped[f"params.{j}"] = text_params[f"params.{old_index[name]}"]
        remapped[f"params.{text_model.unk_param_index}"] = text_params[f"params.{len(names)}"]   # the fixed unknown-word angle
        return remapped

    @classmethod
    def load_model(cls, checkpoint_path: Path, image_model, text_model, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        text_params = checkpoint["text"]
        image_params = checkpoint["image"]

        if "params" in text_params and not any("params." in k for k in text_params):
            text_params = cls.upgrade_checkpoint(text_params)
        if "params" in image_params and not any("params." in k for k in image_params):
            image_params = cls.upgrade_checkpoint(image_params)
        # VQC word parameters are params.<i> in order of first appearance in the training data: map them by name, so a
        # checkpoint loads onto data whose words come in another order (checkpoints from before 2026-11-13 have no names
        # stored; weights/ has them in symbols.json next to best.pt)
        names = checkpoint.get("text_symbols")
        names_file = Path(checkpoint_path).parent / "symbols.json"
        if not names and names_file.exists():
            names = json.load(open(names_file))
        if names and hasattr(text_model, 'sym2param'):
            text_params = cls.params_by_name(text_params, names, text_model, checkpoint_path)

        missing, unexpected = text_model.load_state_dict(text_params, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Text weights in {checkpoint_path} do not match the model built from the config: "
                               f"{len(missing)} missing, {len(unexpected)} unexpected (e.g. {(missing + unexpected)[:3]})")
        image_model.load_state_dict(image_params)
        
        return checkpoint
