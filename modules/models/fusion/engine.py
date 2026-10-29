import torch, yaml, mlflow, time, socket, math, json, csv
from collections import defaultdict
from typing import Callable
import torch.nn.functional as F
from modules.utils.general import gen_id, log_phase, git_state
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
        self.perf = []

    def _setup_environment(self):
        # never reuse a run directory (runs started in the same minute used to share one)
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

    def _log_perf(self, epoch, train_s, n_samples, val_s):
        # steps.csv: per-step losses (the curve a speed-up must reproduce); run_meta.json: speed, memory, parameter counts
        with open(self.checkpoint_dir / "steps.csv", 'a', newline='') as f:
            w = csv.writer(f)
            if epoch == 0: w.writerow(["epoch", "step", "loss", "step_s"])
            for i, (l, dt) in enumerate(self.trainer.step_log): w.writerow([epoch, i, repr(l), f"{dt:.4f}"])
        self.trainer.step_log = []
        self.perf.append({"epoch": epoch, "train_s": train_s, "train_samples_per_s": n_samples / train_s, "val_eval_s": val_s,
                          "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None})
        tm, im = self.trainer.text_model, self.trainer.image_model
        meta = {"run_name": self.run_name, "seed": self.seed, "git_commit": self.config['git_commit'], "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "params_text": sum(p.numel() for p in tm.parameters()),
                "params_text_by_order": tm.param_breakdown() if hasattr(tm, 'param_breakdown') else None,
                "params_image_trainable": sum(p.numel() for p in im.parameters() if p.requires_grad), "epochs": self.perf}
        with open(self.checkpoint_dir / "run_meta.json", 'w') as f:
            json.dump(meta, f, indent=1)

    def _save_checkpoint(self, epoch, loss, metrics, is_best=False):
        payload = {
            "image": self.trainer.image_model.state_dict(),
            "text": self.trainer.text_model.state_dict(),
            "epoch": epoch,
            "train_loss": loss,
            "val_metrics": metrics
        }
        if self.config.get('save_last', True):   # save_last: false keeps only best.pt (2.2 GB per TN checkpoint)
            torch.save(payload, self.checkpoint_path)
        if is_best and self.config.get('save_best', True):   # fixed-budget runs report last.pt and can skip best.pt
            torch.save(payload, self.best_checkpoint_path)

    def fit(self, train_loader, val_loader, eval_mapper):
        self._setup_environment()
        hostname = socket.gethostname()
        log_phase(f"Model #{self.run_name}: optimization started on \"{hostname}\"...")
        
        best_metric_value = float("-inf")
        target_metric = "i2tR1" if 'global_retrieval' in self.eval_tasks else 'hard_neg_acc'
        # select_metric: mrr_mean = mean of i2t_mrr and t2i_mrr (less noisy than R@1 on 5k val images)
        target_metric = self.config.get('select_metric', target_metric)
        metric_needed = getattr(self.scheduler, 'needs_metric', False)   # lr_schedule: plateau steps on the metric

        patience = self.config.get('patience', 15)
        min_delta = self.config.get('min_delta', 1e-3)
        patience_counter = 0
        stopped_epoch = 0

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_params({
                "epochs": self.config['epochs'],
                "batch_size": self.config['batch_size'],
                "learning_rate_quantum": self.config.get('qlr', self.config.get('lr')),
                "learning_rate_classical": self.config.get('clr'),
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
                "label_smoothing": self.trainer.loss_fn.label_smoothing if hasattr(self.trainer.loss_fn, 'label_smoothing') else None,
                "weight_decay": self.config.get('weight_decay', 1e-4),
                "warmup_epochs": self.config.get('warmup_epochs', 5),
                "purity_weight": getattr(self.trainer.loss_fn, 'purity_weight', None),
                "text_layers": self.config['text'].get('layers'),
                "vision_layers": self.config['vision'].get('layers'),
                "overrides": " ".join(self.config.get('overrides', [])) or "none",
                "postselect_types": ",".join(self.config['text'].get('postselect_types') or []) or "none",
                "env_qubits": self.config['vision'].get('env_qubits', 0),
                "mixer_layers": self.config['text'].get('mixer_layers', 0),
                "spread_weight": getattr(self.trainer.loss_fn, 'spread_weight', getattr(self.trainer.loss_fn, 'lambda_reg', None)),
                "init_from": self.config.get('init_from') or "none",
                "hard_negatives": bool(self.config.get('hard_negatives', False)),
                "learn_temperature": getattr(self.trainer.loss_fn, 'logit_scale', None) is not None,
                "margin": self.config.get('margin', 0.0),
                "uniformity_weight": self.config.get('uniformity_weight', 0.0),
                "shot_weight": self.config.get('shot_weight', 0.0),
                "purity_anneal_epochs": self.config.get('purity_anneal_epochs', 0),
                "swap_negatives": bool(self.config.get('swap_negatives', False)),
                "distill_from": self.config.get('distill_from') or "none",
                "distill_kd_weight": self.config.get('distill_kd_weight', 1.0) if self.config.get('distill_from') else 0.0,
                "distill_tau": self.config.get('distill_tau', 1.0),
                "distill_state_weight": self.config.get('distill_state_weight', 0.0) if self.config.get('distill_from') else 0.0,
            })

            if self.config.get('checkpoint_every'):   # the initialisation, as epoch -1
                torch.save({"image": self.trainer.image_model.state_dict(), "text": self.trainer.text_model.state_dict(),
                            "epoch": -1, "train_loss": None, "val_metrics": {}}, self.checkpoint_dir / "epoch_init.pt")
            if self.config.get('eval_init'):   # validation metrics of the initialisation (logged as init_*, not used for selection)
                m0 = self.evaluator.eval_set(dataloader=val_loader, tasks=self.eval_tasks, eval_mapper=eval_mapper)
                mlflow.log_metrics({f"init_{k}": v for k, v in m0.items()})
                tqdm.write("\nInit | " + " | ".join(f"{k}: {v:.4f}" for k, v in m0.items()))
            epoch_pbar = tqdm(range(self.config['epochs']), desc="Training Pipeline", unit="epoch")
            current_lr = self.config.get('qlr', 1e-3)
            
            anneal = int(self.config.get('purity_anneal_epochs', 0) or 0)
            w0 = getattr(self.trainer.loss_fn, 'purity_weight', None)
            for epoch in epoch_pbar:
                start_time = time.time()
                if anneal and w0:   # purity penalty annealed linearly from purity_weight to 0 over purity_anneal_epochs
                    self.trainer.loss_fn.purity_weight = w0 * max(0.0, 1.0 - epoch / anneal)
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                loss = self.trainer.train_epoch(train_loader, eval_mapper)
                elapsed_time = time.time() - start_time

                grad_norm_dict = self.trainer.grad_norm_by_type()

                if self.scheduler is not None and not metric_needed:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)

                t_eval = time.time()
                metrics = self.evaluator.eval_set(
                    dataloader=val_loader,
                    tasks=self.eval_tasks,
                    eval_mapper=eval_mapper
                )
                self._log_perf(epoch, elapsed_time, len(train_loader.dataset), time.time() - t_eval)

                mlflow.log_metrics(metrics, step=epoch)
                for k, v in grad_norm_dict.items():
                    mlflow.log_metric(f"gradient_norm_{k}", v, step=epoch)
                metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                tqdm.write(f"\nEpoch {epoch:02d} | Loss: {loss:.4f} | Gradient Norms: {', '.join([f'{k}: {v:.4f}' for k, v in grad_norm_dict.items()])} | {metrics_str} | Time: {elapsed_time:.1f}s")

                if target_metric == 'mrr_mean' and 'i2t_mrr' in metrics:
                    metrics['mrr_mean'] = 0.5 * (metrics['i2t_mrr'] + metrics['t2i_mrr'])
                    mlflow.log_metric('mrr_mean', metrics['mrr_mean'], step=epoch)
                current_metric = metrics.get(target_metric, 0.0)
                if self.scheduler is not None and metric_needed:
                    self.scheduler.step(current_metric)
                    current_lr = self.scheduler.get_last_lr()[0]
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)
                    tqdm.write(f"lr after epoch {epoch}: {current_lr:.3g}")
                if current_metric > (best_metric_value + min_delta):
                    best_metric_value = current_metric
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False

                self._save_checkpoint(epoch, loss, metrics, is_best=is_best)
                mlflow.log_metric("train_loss", loss, step=epoch)
                ls_ = getattr(self.trainer.loss_fn, 'logit_scale', None)
                if ls_ is not None: mlflow.log_metric("logit_scale", float(ls_.exp().clamp(max=100)), step=epoch)
                every = self.config.get('checkpoint_every')
                if every and epoch % every == 0:   # per-epoch snapshots for training-dynamics analysis
                    torch.save({"image": self.trainer.image_model.state_dict(), "text": self.trainer.text_model.state_dict(),
                                "epoch": epoch, "train_loss": loss, "val_metrics": metrics}, self.checkpoint_dir / f"epoch_{epoch:03d}.pt")

                if patience_counter >= patience:
                    stopped_epoch = epoch
                    tqdm.write(
                        f"\n[EARLY STOPPING TRIGGERED] Metric '{target_metric}' failed to improve by > {min_delta} "
                        f"for {patience} consecutive epochs. Best Score: {best_metric_value:.4f} (at epoch {epoch - patience})."
                    )
                    mlflow.log_metric("early_stopped_epoch", stopped_epoch)
                    break
                
        log_phase("Experiment Run Concluded")

class ContrastiveTrainer:
    def __init__(self, image_model, text_model, optimizer, loss_fn, device):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.step_log = []   # (loss, seconds) per optimiser step, flushed to steps.csv each epoch

    @torch.no_grad()
    def grad_norm(self):
        grad_norm = 0.0

        for _, param in list(self.image_model.named_parameters()) + list(self.text_model.named_parameters()):
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5

    @torch.no_grad()
    def grad_norm_by_type(self) -> dict:
        verb_grad_sq = 0.0
        noun_grad_sq = 0.0
        other_grad_sq = 0.0

        sym2param = getattr(self.text_model, 'sym2param', {})
        param2sym = {idx: sym for sym, idx in sym2param.items()}
        if not sym2param:   # parameters keyed by name (ParameterDict): group by tensor order instead
            sq = defaultdict(float)
            for name, param in self.text_model.named_parameters():
                if param.grad is not None:
                    sq[f"order{param.dim()}" if name.startswith("params.") else "other"] += param.grad.norm(2).item() ** 2
            for k, v in getattr(self.text_model, 'last_row_grad_sq', {}).items():   # packed banks: last step's row grads
                sq[k] += v
            return {**{k: v ** 0.5 for k, v in sorted(sq.items())}, "total_text": sum(sq.values()) ** 0.5}

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
        # profiling (tools/tn/profile_train.py): `timing` dict gets cuda-synchronised seconds per phase, `max_steps` stops early
        timing, max_steps = getattr(self, 'timing', None), getattr(self, 'max_steps', None)
        t_ph = [time.time()]
        def tick(phase):
            if timing is None: return
            if torch.cuda.is_available(): torch.cuda.synchronize()
            now = time.time(); timing[phase] = timing.get(phase, 0.0) + now - t_ph[0]; t_ph[0] = now

        t_step = time.time()
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            tick('data')
            images, texts = batch_mapper(batch)
            images = images.to(self.device)

            image_emb = self.image_model(images)
            tick('fwd_image')
            text_emb = self.text_model(texts)
            tick('fwd_text')

            neg_emb = self.image_model(batch['neg_image'].to(self.device)) if getattr(self, 'hard_negatives', False) and 'neg_image' in batch else None
            swap_kw = {}
            if getattr(self, 'swap_negatives', False) and 'swap_caption' in batch:
                swap_kw = {'neg_text_emb': self.text_model(batch['swap_caption']), 'neg_text_valid': batch['swap_valid'].to(self.device)}
            if neg_emb is not None or swap_kw:
                loss = self.loss_fn(text_emb, image_emb, neg_emb, **swap_kw)
            else:
                loss = self.loss_fn(text_emb, image_emb)
            if getattr(self, 'teacher', None) is not None:
                loss = loss + self.teacher.loss(batch['idx'].to(self.device), self.loss_fn, text_emb, image_emb)
            tick('loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            tick('backward')

            # torch.nn.utils.clip_grad_norm_(self.text_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            tick('optimizer')
            l = loss.item()
            epoch_loss += l * images.shape[0]
            now = time.time(); self.step_log.append((l, now - t_step)); t_step = now
            t_ph[0] = time.time()
        
        return epoch_loss / len(dataloader.dataset)

def adaptive_optimizer(text_model, image_model, base_qlr: float = 1e-3, base_clr: float = 1e-5, 
                       arity_factor: float = 1.5, weight_decay: float = 0.01) -> torch.optim.AdamW:
    
    sym2param = getattr(text_model, 'sym2param', {})
    param2sym = {idx: sym for sym, idx in sym2param.items()}
    arity_param_groups = defaultdict(list)

    for name, param in text_model.named_parameters():
        if not param.requires_grad:
            continue

        if "params" in name:
            try:
                idx = int(name.split(".")[-1])
                sym_name = param2sym.get(idx, "")
            except ValueError:
                sym_name = ""
        else:
            sym_name = name

        cmplx_type = sym_name.split('__')[1].split('_')[0].split('@')
        op_arity = max(1, len(cmplx_type))
        arity_param_groups[op_arity].append(param)

    param_groups = []
    
    for arity in sorted(arity_param_groups.keys()):
        params = arity_param_groups[arity]
        group_lr = base_qlr * (arity_factor ** (arity - 1))
        
        param_groups.append({
            'params': params,
            'lr': group_lr,
            'name': f'q_text_arity_{arity}_lr_{group_lr:.1e}'
        })
        print(f"[Optimizer Config] Arity {arity} | {len(params)} params | LR: {group_lr:.6f}")

    # Add classical image tower group
    param_groups.append({
        'params': image_model.parameters(),
        'lr': base_clr,
        'name': 'image_tower'
    })
    print("[Optimizer Config] Classical Image Tower | {} params | LR: {:.6f}".format(len(list(image_model.parameters())), base_clr))
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

class MMEvaluator:
    def __init__(self, image_model, text_model, device):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.device = device
        self._sims_cache = None  # set by eval_set so several tasks share one encoding pass
        self.abs_scores = not getattr(text_model, 'real_scores', False)   # |<a|b>| unless the model's embeddings are real

    def _readout(self, s):
        return s.abs() if self.abs_scores else s.real

    def _score(self, a, b):
        # row-wise score of paired embeddings
        return self._readout(torch.sum(a.conj() * b, dim=1))

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
            pos_arr.append(self._score(m1_emb, pos_m2_emb))
            neg_arr.append(self._score(m1_emb, neg_m2_emb))
            m1_u = unit(m1_emb)
            pos_n_arr.append(self._score(m1_u, unit(pos_m2_emb)))
            neg_n_arr.append(self._score(m1_u, unit(neg_m2_emb)))
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
        # rank of the best positive = number of negatives scored strictly above it (no full argsort: 5k x 25k sorted twice
        # per val eval was most of its time). Same as the old argsort rank except at exact score ties (duplicate captions),
        # which argsort broke arbitrarily and which now count in the positive's favour.
        best_pos = scores.masked_fill(~mask, float('-inf')).max(dim=1, keepdim=True).values
        positions = ((scores > best_pos) & ~mask).sum(dim=1)
        
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
            all_img.append(img_emb)   # kept on the device: the 5k x 25k scores and metrics run there

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
            all_txt.append(txt_emb)

        similarity = self._readout(torch.cat(all_img, dim=0) @ torch.cat(all_txt, dim=0).conj().T)
        n_imgs, m_txts = similarity.shape

        mask = torch.zeros((n_imgs, m_txts), dtype=torch.bool, device=similarity.device)
        rows, cols = torch.tensor(match_coordinates, dtype=torch.long, device=similarity.device).T
        mask[rows, cols] = True

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

    @torch.no_grad()
    def evaluate_sugarcrepe_pp(self, dataloader: torch.utils.data.DataLoader) -> float:
        correct = p1 = p2 = total = 0
        for batch in dataloader:
            img_emb = self._encode_img(batch["image"])
            pos1_emb = self._encode_txt(batch["pos_caption1"])
            pos2_emb = self._encode_txt(batch["pos_caption2"])
            neg_emb = self._encode_txt(batch["neg_caption"])
            
            sim_pos1 = self._score(img_emb, pos1_emb)
            sim_pos2 = self._score(img_emb, pos2_emb)
            sim_neg  = self._score(img_emb, neg_emb)
            
            match = (sim_pos1 > sim_neg) & (sim_pos2 > sim_neg)
            correct += match.sum().item()
            p1 += (sim_pos1 > sim_neg).sum().item(); p2 += (sim_pos2 > sim_neg).sum().item()
            total += img_emb.size(0)
            
        return {"acc": correct / total, "acc_pos1": p1 / total, "acc_pos2": p2 / total}

    @torch.no_grad()
    def evaluate_winoground(self, dataloader) -> dict:
        text_corr = img_corr = group_corr = total = 0
        for batch in dataloader:
            i0, c0 = self._encode_img(batch["image_0"]), self._encode_txt(batch["caption_0"])
            i1, c1 = self._encode_img(batch["image_1"]), self._encode_txt(batch["caption_1"])
            
            s_i0_c0 = self._score(i0, c0)
            s_i0_c1 = self._score(i0, c1)
            s_i1_c0 = self._score(i1, c0)
            s_i1_c1 = self._score(i1, c1)

            t_match = (s_i0_c0 > s_i0_c1) & (s_i1_c1 > s_i1_c0)
            i_match = (s_i0_c0 > s_i1_c0) & (s_i1_c1 > s_i0_c1)
            g_match = t_match & i_match

            text_corr += t_match.sum().item()
            img_corr += i_match.sum().item()
            group_corr += g_match.sum().item()
            total += i0.size(0)
        
        return {"txt_score": text_corr/total, "img_score": img_corr/total, "grp_score": group_corr/total}

    @torch.no_grad()
    def cptp_diagnostic(self, dataloader, choice="text") -> dict:
        self.image_model.eval()
        self.text_model.eval()

        sum_m1_purity, sum_m1_entropy, sum_m1_eff_rank = 0.0, 0.0, 0.0
        sum_m2_purity, sum_m2_entropy, sum_m2_eff_rank = 0.0, 0.0, 0.0
        total_samples = 0
        metrics = {}

        def _to_density_matrix(emb: torch.Tensor) -> torch.Tensor:
            B = emb.size(0)
            if emb.dim() == 2:
                D = int(math.sqrt(emb.size(1)))
                return emb.view(B, D, D).to(torch.complex64)
            return emb.to(torch.complex64)

        def _compute_state_stats(rho: torch.Tensor):
            # 1. Purity P = Tr(rho^2) computed natively on GPU
            purity = torch.real(torch.einsum('bmn,bnm->b', rho, rho))

            # 2. CPU Offloading for Hermitian Eigenvalues to avoid cuSOLVER/CUDA driver crashes
            rho_cpu = rho.detach().cpu()
            evals = torch.linalg.eigvalsh(rho_cpu).clamp(min=1e-8)

            # 3. von Neumann Entropy S(rho) = -Tr(rho ln rho)
            entropy = -(evals * torch.log(evals)).sum(dim=-1)

            # 4. Effective Rank / Participation Ratio R_eff = 1 / Tr(rho^2)
            eff_rank = 1.0 / (evals ** 2).sum(dim=-1)

            return purity.sum().item(), entropy.sum().item(), eff_rank.sum().item()

        for batch in dataloader:
            if choice == "text":
                m1_emb = self._encode_img(batch["image"])
                pos_m2_emb = self._encode_txt(batch["pos_caption"])
                neg_m2_emb = self._encode_txt(batch["neg_caption"])
            elif choice == "image":
                m1_emb = self._encode_txt(batch["caption"])
                pos_m2_emb = self._encode_img(batch["pos_image"])
                neg_m2_emb = self._encode_img(batch["neg_image"])

            B = m1_emb.size(0)
            rho_m1 = _to_density_matrix(m1_emb)
            rho_m2_pos = _to_density_matrix(pos_m2_emb)
            rho_m2_neg = _to_density_matrix(neg_m2_emb)

            # Modality 1 statistics
            m1_pur, m1_ent, m1_rank = _compute_state_stats(rho_m1)
            sum_m1_purity += m1_pur
            sum_m1_entropy += m1_ent
            sum_m1_eff_rank += m1_rank

            # Modality 2 statistics (averaged across positive and negative pairs)
            m2_pos_pur, m2_pos_ent, m2_pos_rank = _compute_state_stats(rho_m2_pos)
            m2_neg_pur, m2_neg_ent, m2_neg_rank = _compute_state_stats(rho_m2_neg)

            sum_m2_purity += 0.5 * (m2_pos_pur + m2_neg_pur)
            sum_m2_entropy += 0.5 * (m2_pos_ent + m2_neg_ent)
            sum_m2_eff_rank += 0.5 * (m2_pos_rank + m2_neg_rank)

            total_samples += B

        if total_samples == 0:
            return {}

        # Map modality identifiers based on query choice
        m1_name = "img" if choice == "text" else "txt"
        m2_name = "txt" if choice == "text" else "img"

        mean_m1_pur = sum_m1_purity / total_samples
        mean_m2_pur = sum_m2_purity / total_samples

        # 1. Purities & Purity Imbalance Ratio
        metrics[f"diag_mean_{m1_name}_purity"] = mean_m1_pur
        metrics[f"diag_mean_{m2_name}_purity"] = mean_m2_pur
        metrics["diag_purity_imbalance"] = abs(mean_m1_pur - mean_m2_pur)

        # 2. von Neumann Entropies
        metrics[f"diag_mean_{m1_name}_entropy"] = sum_m1_entropy / total_samples
        metrics[f"diag_mean_{m2_name}_entropy"] = sum_m2_entropy / total_samples

        # 3. Effective Ranks / Participation Ratios
        metrics[f"diag_mean_{m1_name}_eff_rank"] = sum_m1_eff_rank / total_samples
        metrics[f"diag_mean_{m2_name}_eff_rank"] = sum_m2_eff_rank / total_samples

        # Legacy backward-compatibility key
        metrics["diag_mean_text_purity"] = metrics["diag_mean_txt_purity"]

        return metrics
    
    @torch.no_grad()
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

    def cptp_diagnostic_text_choice(self, dataloader) -> dict:
        return self.cptp_diagnostic(dataloader, choice="text")

    def cptp_diagnostic_image_choice(self, dataloader) -> dict:
        return self.cptp_diagnostic(dataloader, choice="image")
    
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