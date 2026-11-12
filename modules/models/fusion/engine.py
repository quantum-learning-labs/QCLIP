import torch, yaml, mlflow, time, socket, math
from collections import defaultdict
from typing import Callable
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from modules.utils.general import gen_id, log_phase, git_state, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine
from pathlib import Path
from tqdm import tqdm

mscoco_mapper = lambda batch: (batch["image"], batch["caption"])
aro_mapper = lambda batch: (batch["image"], batch["pos_caption"])
svo_mapper = lambda batch: (batch["pos_image"], batch["caption"])


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
            "val_metrics": metrics
        }
        torch.save(payload, self.checkpoint_path)
        if is_best:
            torch.save(payload, self.best_checkpoint_path)

    def _save_snapshot(self, name, epoch, loss, metrics):
        torch.save({"image": self.trainer.image_model.state_dict(), "text": self.trainer.text_model.state_dict(),
                    "epoch": epoch, "train_loss": loss, "val_metrics": metrics}, self.checkpoint_dir / name)

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

class Teacher:
    # Frozen post-selected teacher for score distillation (distill_from). Its circuits compile differently from the
    # student's, so its text and image embeddings are computed once for every training row and looked up by row index.
    def __init__(self, checkpoint_path, student_train_df, device, kd_weight=1.0, kd_tau=1.0):
        self.kd_weight = kd_weight
        self.kd_tau = kd_tau

        with open(Path(checkpoint_path).parent / 'config.yaml') as f:
            config = yaml.safe_load(f)
        ansatz, image_model, text_model, self.loss_fn = build_experiment(config, device)
        data_engine = DataEngine(config, ansatz, device)
        compiled_train = data_engine.compile_text('train')
        compiled_val = data_engine.compile_text('val')
        if len(compiled_train) != len(student_train_df) or (compiled_train['pos_image_id'].values != student_train_df['pos_image_id'].values).any():
            raise RuntimeError("Teacher and student training rows are not aligned (row count or image ids differ)")

        data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
        data_engine.image_init(image_model)
        CheckpointManager.load_model(checkpoint_path, image_model, text_model, device)
        image_model.to(device).eval()
        text_model.to(device).eval()

        loader = DataLoader(data_engine.get_dataset(compiled_train, split='train'), batch_size=config['batch_size'],
                            shuffle=False, collate_fn=data_engine.collate_fn, num_workers=0)
        text_arr, image_arr = [], []
        with torch.no_grad():
            for batch in loader:
                image_arr.append(image_model(batch['pos_image'].to(device)))
                text_arr.append(text_model(batch['caption']))
        self.text_emb = torch.cat(text_arr)
        self.image_emb = torch.cat(image_arr)
        print(f"Teacher {checkpoint_path}: {len(self.text_emb)} rows | KD weight: {kd_weight} | KD tau: {kd_tau}")

    def kl_div(self, student_logits, teacher_logits):
        return F.kl_div(F.log_softmax(student_logits / self.kd_tau, dim=1), F.log_softmax(teacher_logits / self.kd_tau, dim=1),
                        log_target=True, reduction='batchmean')

    def loss(self, idx, student_loss_fn, text_emb, image_emb):
        # symmetric KL between the student's and the teacher's in-batch score matrices (both directions), times tau^2
        with torch.no_grad():
            teacher_logits = self.loss_fn.compute_similarity(self.text_emb[idx], self.image_emb[idx]) / self.loss_fn.temperature
        student_logits = student_loss_fn.compute_similarity(text_emb, image_emb) / student_loss_fn.temperature
        kd = 0.5 * (self.kl_div(student_logits, teacher_logits) + self.kl_div(student_logits.T, teacher_logits.T))
        return self.kd_weight * self.kd_tau ** 2 * kd


class ContrastiveTrainer:
    def __init__(self, image_model, text_model, optimizer, loss_fn, device, swap_negatives=False, teacher=None):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.swap_negatives = swap_negatives   # swapped captions as extra image->text negatives (needs swap_caption in the batch)
        self.teacher = teacher                 # Teacher: score distillation from a frozen post-selected model

    @torch.no_grad()
    def grad_norm_by_type(self) -> dict:
        verb_grad_sq = 0.0
        noun_grad_sq = 0.0
        other_grad_sq = 0.0

        sym2param = getattr(self.text_model, 'sym2param', {})
        param2sym = {idx: sym for sym, idx in sym2param.items()}

        for name, param in self.text_model.named_parameters():
            if param.grad is None:
                continue

            g_norm_sq = param.grad.data.norm(2).item() ** 2

            if "params" in name:
                try:
                    idx = int(name.split(".")[-1])
                    sym_name = param2sym.get(idx, "")
                except ValueError:
                    sym_name = ""
            else:
                sym_name = name

            # print(sym_name.split('__')[1].split('_')[0].split('@'))
            cmplx_type = sym_name.split('__')[1].split('_')[0].split('@')
            op_arity = len(cmplx_type)
            cmplx_type = ''.join(cmplx_type)

            if cmplx_type == 'n':
                noun_grad_sq += g_norm_sq
            elif op_arity == 3:
                verb_grad_sq += g_norm_sq
            else:
                other_grad_sq += g_norm_sq

        return {
            "verb_grad_norm": verb_grad_sq ** 0.5,
            "noun_grad_norm": noun_grad_sq ** 0.5,
            "other_grad_norm": other_grad_sq ** 0.5,
            "total_text_grad_norm": (verb_grad_sq + noun_grad_sq) ** 0.5
        }

    def train_epoch(self, dataloader, batch_mapper: Callable) -> float:
        self.image_model.train()
        self.text_model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            images, texts = batch_mapper(batch)
            images = images.to(self.device)

            image_emb = self.image_model(images)
            text_emb = self.text_model(texts)

            if self.swap_negatives and 'swap_caption' in batch:
                loss = self.loss_fn(text_emb, image_emb, self.text_model(batch['swap_caption']), batch['swap_valid'].to(self.device))
            else:
                loss = self.loss_fn(text_emb, image_emb)
            if self.teacher is not None:
                loss = loss + self.teacher.loss(batch['idx'].to(self.device), self.loss_fn, text_emb, image_emb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.text_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_loss += loss.item() * images.shape[0]
        
        return epoch_loss / len(dataloader.dataset)

class MMEvaluator:
    def __init__(self, image_model, text_model, device):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.device = device
        self._sims_cache = None  # set by eval_set so several tasks share one encoding pass

    @torch.no_grad()
    def _pair_sims(self, dataloader, choice):
        key = (id(dataloader), choice)
        if self._sims_cache is not None and key in self._sims_cache:
            return self._sims_cache[key]
        is_dm = getattr(self.text_model, 'discard', False)
        def unit(emb):
            # unit norm (statevectors) or unit trace (flattened density matrices), as the emulator measures
            if is_dm:
                d = int(round(emb.shape[1] ** 0.5))
                return emb / emb.view(-1, d, d).diagonal(dim1=1, dim2=2).sum(-1).real.clamp(min=1e-12).unsqueeze(1)
            return emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-12)
        pos_arr, neg_arr, pos_n_arr, neg_n_arr = [], [], [], []
        for batch in dataloader:
            if choice == 'text':
                m1_emb = self._encode_img(batch["image"])
                pos_m2_emb = self._encode_txt(batch["pos_caption"])
                neg_m2_emb = self._encode_txt(batch["neg_caption"])
            if choice == 'image':
                m1_emb = self._encode_txt(batch["caption"])
                pos_m2_emb = self._encode_img(batch["pos_image"])
                neg_m2_emb = self._encode_img(batch["neg_image"])
            pos_arr.append(torch.sum(m1_emb.conj() * pos_m2_emb, dim=1).abs())
            neg_arr.append(torch.sum(m1_emb.conj() * neg_m2_emb, dim=1).abs())
            m1_u = unit(m1_emb)
            pos_n_arr.append(torch.sum(m1_u.conj() * unit(pos_m2_emb), dim=1).abs())
            neg_n_arr.append(torch.sum(m1_u.conj() * unit(neg_m2_emb), dim=1).abs())
        sims = tuple(torch.cat(a).cpu() for a in (pos_arr, neg_arr, pos_n_arr, neg_n_arr))
        if self._sims_cache is not None:
            self._sims_cache[key] = sims
        return sims

    @torch.no_grad()
    def _encode_txt(self, texts):
        return self.text_model(texts).flatten(1).contiguous()
    
    @torch.no_grad()
    def _encode_img(self, images):
        return self.image_model(images.to(self.device)).flatten(1).contiguous()
    
    def _calculate_recall(self, scores: torch.Tensor, mask: torch.Tensor, prefix: str):
        rankings = scores.argsort(dim=1, descending=True)
        matched_positions = mask.gather(1, rankings).float()
        positions = torch.argmax(matched_positions, dim=1)
        #positions = (rankings == target_indices[:, None]).nonzero(as_tuple=False)[:, 1]
        
        results = {}
        results[f"{prefix}R1"] = (positions < 1).float().mean().item()
        results[f"{prefix}R5"] = (positions < 5).float().mean().item()
        results[f"{prefix}R10"] = (positions < 10).float().mean().item()
        results[f"{prefix}_mrr"] = (1.0 / (positions.float() + 1.0)).mean().item()
        return results

    @torch.no_grad()
    def global_retrieval(self, dataloader, batch_mapper) -> dict:
        self.image_model.eval()
        self.text_model.eval()
        all_img, all_txt = [], []
        global_img_idx, global_txt_idx = 0, 0
        match_coordinates = []

        for batch in dataloader:
            img_batch, txt_batch = batch_mapper(batch)
            img_emb = self._encode_img(img_batch)
            all_img.append(img_emb.cpu())

            text_payload_batch = []
            for item_captions in txt_batch:
                if isinstance(item_captions[0], str) or isinstance(item_captions[0][0], dict):
                    # Single element, either (str, [dict]) or ([dict], None)
                    item_captions = [item_captions] 

                num_captions = len(item_captions)
                for _ in range(num_captions):
                    match_coordinates.append((global_img_idx, global_txt_idx))
                    global_txt_idx += 1
                global_img_idx += 1

                for c in item_captions:
                    text_payload_batch.append((c[0], c[1]))
            txt_emb = self._encode_txt(text_payload_batch)
            all_txt.append(txt_emb.cpu())

        similarity = (torch.cat(all_img, dim=0) @ torch.cat(all_txt, dim=0).conj().T).abs()
        n_imgs, m_txts = similarity.shape

        mask = torch.zeros((n_imgs, m_txts), dtype=torch.bool, device=similarity.device)
        for i, j in match_coordinates:
            mask[i, j] = True

        metrics = {}

        # --- 1. Directional Retrieval Metrics (i2t and t2i) ---
        metrics.update(self._calculate_recall(similarity, mask, "i2t"))
        metrics.update(self._calculate_recall(similarity.T, mask.T, "t2i"))

        # --- 2. Advanced Geometric & Structural Latent Metrics ---
        avg_positive = similarity[mask].mean().item()
        avg_negative = similarity[~mask].mean().item()

        # --- 3. Global Embedding Margin ---
        metrics["gamma"] = avg_positive - avg_negative

        # --- 4. Directional Hard-Negative Margins ---
        mask_matrix = similarity.clone()
        mask_matrix[mask] = float("-inf")
        
        max_neg_i2t, _ = mask_matrix.max(dim=1)
        max_neg_t2i, _ = mask_matrix.max(dim=0)

        pos_matrix = similarity.clone()
        pos_matrix[~mask] = 0.0
        avg_pos_per_img = pos_matrix.sum(dim=1) / mask.sum(dim=1).float()
        avg_pos_per_txt = pos_matrix.sum(dim=0) / mask.sum(dim=0).float()
        
        metrics["i2t_hnm"] = (avg_pos_per_img - max_neg_i2t).mean().item()
        metrics["t2i_hnm"] = (avg_pos_per_txt - max_neg_t2i).mean().item()

        # --- 5. Spatial Alignment Asymmetry ---
        metrics["delta"] = abs(metrics["i2tR1"] - metrics["t2iR1"])
            
        return metrics
    
    @torch.no_grad()
    def hard_neg_eval(self, dataloader, choice = 'text') -> dict:
        self.image_model.eval()
        self.text_model.eval()
        pos_sim, neg_sim, pos_n, neg_n = self._pair_sims(dataloader, choice)
        margin = pos_sim - neg_sim
        correct = (margin > 0).float()
        correct_n = (pos_n > neg_n).float()
        return ({"hard_neg_acc": correct.sum().item() / len(correct),
                 "hard_neg_acc_norm": correct_n.sum().item() / len(correct_n)},
                {"pos_scores": list(pos_sim.numpy()), "neg_scores": list(neg_sim.numpy()),
                 "margins": list(margin.numpy()), "correct": list(correct.numpy()), "correct_norm": list(correct_n.numpy())})
            
    def evaluate_text_choice(self, dataloader) -> dict:
        return self.hard_neg_eval(dataloader, choice='text')

    def evaluate_image_choice(self, dataloader) -> dict:
        return self.hard_neg_eval(dataloader, choice='image')

    def diagnostic(self, dataloader, choice="text") -> dict:
        self.image_model.eval()
        self.text_model.eval()

        metrics = {}
        pos_sim, neg_sim, _, _ = self._pair_sims(dataloader, choice)
        total_samples = len(pos_sim)
        if total_samples == 0:
            return {}

        metrics["diag_mean_pos_overlap"] = pos_sim.double().sum().item() / total_samples
        metrics["diag_mean_neg_overlap"] = neg_sim.double().sum().item() / total_samples
        metrics["diag_collapse_gap"] = (pos_sim - neg_sim).abs().double().sum().item() / total_samples

        return metrics
    
    def diagnostic_text_choice(self, dataloader) -> dict:
        return self.diagnostic(dataloader, choice="text")
    
    def diagnostic_image_choice(self, dataloader) -> dict:
        return self.diagnostic(dataloader, choice="image")

    @torch.no_grad()
    def eval_set(self, dataloader, tasks, eval_mapper) -> dict:
        self.image_model.eval()
        self.text_model.eval()
        metrics = {}
        self._sims_cache = {}
        for task_name in tasks:
            eval_fn = getattr(self, task_name, None)
            if eval_fn is None:
                print(f"Warning: Evaluation method '{task_name}' not found on MMEvaluator. Skipping.")
                continue

            if task_name == "global_retrieval":
                task_metrics = eval_fn(dataloader, eval_mapper)
            else:
                out = eval_fn(dataloader)
                task_metrics = out[0] if isinstance(out, tuple) else out
                
            metrics.update(task_metrics)

        self._sims_cache = None
        return metrics