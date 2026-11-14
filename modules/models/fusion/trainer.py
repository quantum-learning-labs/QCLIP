import torch, yaml
import torch.nn.functional as F
from pathlib import Path
from typing import Callable
from torch.utils.data import DataLoader
from modules.factory import load_experiment

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

class Teacher:
    # Frozen post-selected teacher for score distillation (distill_from). Its circuits compile differently from the
    # student's, so its text and image embeddings are computed once for every training row and looked up by row index.
    def __init__(self, checkpoint_path, student_train_df, device, kd_weight=1.0, kd_tau=1.0):
        self.kd_weight = kd_weight
        self.kd_tau = kd_tau

        with open(Path(checkpoint_path).parent / 'config.yaml') as f:
            config = yaml.safe_load(f)
        exp = load_experiment(config, device, checkpoint_path)
        if len(exp.train) != len(student_train_df) or (exp.train['pos_image_id'].values != student_train_df['pos_image_id'].values).any():
            raise RuntimeError("Teacher and student training rows are not aligned (row count or image ids differ)")
        self.loss_fn = exp.loss_fn
        image_model, text_model = exp.image_model.to(device).eval(), exp.text_model.to(device).eval()

        loader = DataLoader(exp.data.get_dataset(exp.train, split='train'), batch_size=config['batch_size'],
                            shuffle=False, collate_fn=exp.data.collate_fn, num_workers=0)
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
