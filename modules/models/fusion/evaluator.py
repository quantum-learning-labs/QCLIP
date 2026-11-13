import time, torch

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

    def benchmark(self, data_engine, test_sets) -> tuple[dict, dict]:
        # every test set of the config: {set}_{metric} results and per-pair details
        self.image_model.eval()
        self.text_model.eval()
        results, details = {}, {}
        for split_name, split_info in test_sets.items():
            compiled_eval = data_engine.compile_text(split_name)
            if data_engine.config['model_type'] == 'vqc':   # circuit size metrics (quantum text model only)
                data_engine.describe_einsum(self.text_model, compiled_eval)
            eval_loader = data_engine.get_loader(compiled_eval, split=split_name)
            with torch.no_grad():
                start_task = time.time()
                out = getattr(self, split_info['task'])(eval_loader)
            task_metrics = out[0] if isinstance(out, tuple) else out
            if isinstance(out, tuple):
                details[split_name] = out[1]
            print(f" -> Evaluated '{split_name}' task in {time.time() - start_task:.2f}s")
            for key, val in task_metrics.items():
                results[f"{split_name}_{key}"] = val
            results[f"{split_name}_n"] = len(compiled_eval)
        return results, details
