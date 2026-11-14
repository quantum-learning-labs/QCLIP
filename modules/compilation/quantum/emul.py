import math, time, torch
import numpy as np
from tqdm import tqdm
from qiskit_aer import AerSimulator

class Emulator:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.shots = 64
        self.discard = self.config.get('text', {}).get('discard', False)
        self.meas_method = self.config.get('emulate', {}).get('measurement_method', 'compute_uncompute')
        self.save_counts = config['emulate'].get('save_counts', False)   # full counts of every circuit (first shot run)
        self.batch_size = config['emulate'].get('circuit_batch', 64)      # circuits per Aer job
        # emulate.dm_max_qubits N (hybrid): circuits with <= N active qubits run as exact density matrices
        # (cost independent of shots), the rest on the main backend (statevector trajectories)
        self.dm_max_qubits = config['emulate'].get('dm_max_qubits', 0)
        self.dm_backend = None
        # seed option of the backend: Aer's seed_simulator, or the IonQ simulator's sampler_seed
        self.seed_option = 'sampler_seed' if config['emulate'].get('provider') == 'ionq' else 'seed_simulator'
        if self.dm_max_qubits and config['emulate'].get('provider') == 'ionq':
            raise ValueError("emulate.dm_max_qubits is an Aer option: not available with the IonQ simulator")
        if self.dm_max_qubits:
            options = backend.options
            self.dm_backend = AerSimulator(method='density_matrix', device=options.device, precision=options.precision,
                                           noise_model=getattr(options, 'noise_model', None))

    def shot_estimation(self, nq_out, nq_ps=0, resolution=50):
        raw_shots = resolution * max(nq_ps, 1) * (2 ** nq_out)
        self.shots = min(int(math.ceil(raw_shots)), 1_000_000)

    @staticmethod
    def active_qubits(qc):
        return len({q for inst in qc.data for q in inst.qubits})

    def _run_counts(self, chunk, shots):
        idx_dm = [i for i, qc in enumerate(chunk) if self.dm_backend is not None and self.active_qubits(qc) <= self.dm_max_qubits]
        idx_sv = [i for i in range(len(chunk)) if i not in set(idx_dm)]
        out = [None] * len(chunk)
        for backend, idx in ((self.dm_backend, idx_dm), (self.backend, idx_sv)):
            if not idx:
                continue
            res = backend.run([chunk[i] for i in idx], shots=shots).result()
            for j, i in enumerate(idx):
                out[i] = res.get_counts(j)
        return out

    def run_circuit(self, qc_array, shots=None, batch_size=24):
        result_array = []
        chunks = [qc_array[i:i + batch_size] for i in range(0, len(qc_array), batch_size)]

        for chunk in tqdm(chunks, desc="Running Circuits"):
            all_counts = self._run_counts(chunk, shots)
            for i, qc in enumerate(chunk):
                counts = all_counts[i]
                if self.save_counts and self._run_idx == 0:
                    self.counts_log.append({'counts': dict(counts), 'out_qs': list(qc.metadata.get('output_qubits', [])),
                                            'num_clbits': qc.num_clbits, 'n_out': len(qc.metadata.get('output_qubits', []))})
                
                if self.meas_method == "compute_uncompute":
                    fidelity = self._eval_compute_uncompute(qc, counts)
                elif self.meas_method == "destructive_swap":
                    fidelity = self._eval_destructive_swap(qc, counts)
                else:
                    raise ValueError(f"Unknown measurement method: {self.meas_method}")
                result_array.append(fidelity)
        return torch.tensor(result_array, dtype=torch.float32)
    
    def _eval_compute_uncompute(self, qc, counts):
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0

        out_qs = set(qc.metadata.get("output_qubits", []))
        num_clbits = qc.num_clbits

        if self.discard:
            target_str = "0" * num_clbits
            self.accept_log.append(1.0)
            self.circuit_log.append((1.0, counts.get(target_str, 0) / total_shots, counts.get(target_str, 0) / total_shots))
            return counts.get(target_str, 0) / total_shots
        else:
            accepted_shots = 0
            out_zero_shots = 0   # output qubits all 0, whatever the post-selected qubits read (no post-selection)
            all_zero_shots = counts.get("0" * num_clbits, 0)
            for bstr, count in counts.items():
                rev_bstr = bstr[::-1]  
                if all(rev_bstr[q] == '0' for q in range(num_clbits) if q not in out_qs):
                    accepted_shots += count
                if all(rev_bstr[q] == '0' for q in out_qs):
                    out_zero_shots += count
            self.accept_log.append(accepted_shots / total_shots)
            self.circuit_log.append((accepted_shots / total_shots, all_zero_shots / total_shots, out_zero_shots / total_shots))
            if accepted_shots == 0:
                return 0.0
            return all_zero_shots / accepted_shots
        
    def _eval_destructive_swap(self, qc, counts):
        out_qs = qc.metadata.get("output_qubits", [])
        n_out = len(out_qs)

        accepted_shots = 0
        weighted_parity_sum = 0

        if self.discard:
            for bstr, count in counts.items():
                rev_bstr = bstr[::-1]
                k = sum(
                    int(rev_bstr[i]) * int(rev_bstr[n_out + i]) 
                    for i in range(n_out)
                )
                parity_sign = 1 if k % 2 == 0 else -1
                weighted_parity_sum += count * parity_sign
                accepted_shots += count
        else:
            n_txt = qc.num_clbits - n_out
            aux_qs = [q for q in range(n_txt) if q not in out_qs]

            parity_all = 0   # the same swap test over ALL shots = the traced (discard) readout of this circuit, no post-selection
            for bstr, count in counts.items():
                rev_bstr = bstr[::-1]
                k = sum(
                    int(rev_bstr[out_qs[i]]) * int(rev_bstr[n_txt + i])
                    for i in range(n_out)
                )
                parity_sign = 1 if k % 2 == 0 else -1
                parity_all += count * parity_sign
                if all(rev_bstr[q] == '0' for q in aux_qs):
                    weighted_parity_sum += count * parity_sign
                    accepted_shots += count
        total = max(sum(counts.values()), 1)
        self.accept_log.append(accepted_shots / total)  # post-selection acceptance (1 for discard)
        self.circuit_log.append((accepted_shots / total, weighted_parity_sum / total, (weighted_parity_sum if self.discard else parity_all) / total))
        if accepted_shots == 0:
            return 0.0

        fidelity = weighted_parity_sum / accepted_shots
        return max(0.0, float(fidelity))

    def run_experiment(self, pos_circs, neg_circs, shots=None, num_runs=5):
        shots = shots if shots is not None else self.shots
        torch.cuda.empty_cache()
        self.elapsed_log = []
        self.accept_log = []
        self.circuit_log = []   # per circuit (acceptance, accepted signal / all shots, unconditioned signal / all shots); order: per run, pos then neg

        all_accuracies = []
        all_pos_fidelities = []
        all_neg_fidelities = []
        per_pair_pos, per_pair_neg = [], []

        base_seed = int(np.random.randint(0, 2**31 - 1))

        self.counts_log = []
        for run_idx in range(num_runs):
            self._run_idx = run_idx
            self.backend.set_options(**{self.seed_option: base_seed + run_idx})
            if self.dm_backend is not None:
                self.dm_backend.set_options(seed_simulator=base_seed + run_idx)

            start_eval_time = time.time()
            pos_raw = self.run_circuit(pos_circs, shots=shots, batch_size=self.batch_size)
            neg_raw = self.run_circuit(neg_circs, shots=shots, batch_size=self.batch_size)
            elapsed = time.time() - start_eval_time
            self.elapsed_log.append(elapsed)

            correct = (pos_raw > neg_raw).float()
            accuracy = correct.mean().item()

            all_accuracies.append(accuracy)
            all_pos_fidelities.append(pos_raw.mean().item())
            all_neg_fidelities.append(neg_raw.mean().item())
            per_pair_pos.append(pos_raw.tolist())
            per_pair_neg.append(neg_raw.tolist())

            print(f" Trial {run_idx + 1} ({elapsed:.2f}s) | Accuracy: {accuracy * 100:.2f}% | Mean Pos Fid: {pos_raw.mean().item():.4f} | Mean Neg Fid: {neg_raw.mean().item():.4f}")

        accs = np.array(all_accuracies)
        pos_fids = np.array(all_pos_fidelities)
        neg_fids = np.array(all_neg_fidelities)

        n = len(accs)
        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs, ddof=1)) if n > 1 else 0.0
        acc_var = float(np.var(accs, ddof=1)) if n > 1 else 0.0
        acc_ci95 = float(1.96 * (acc_std / np.sqrt(n))) if n > 1 else 0.0

        pos_mean = float(np.mean(pos_fids))
        pos_std = float(np.std(pos_fids, ddof=1)) if n > 1 else 0.0
        neg_mean = float(np.mean(neg_fids))
        neg_std = float(np.std(neg_fids, ddof=1)) if n > 1 else 0.0


        return {
            "accuracy": acc_mean,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "accuracy_var": acc_var,
            "accuracy_ci95": acc_ci95,
            "pos_fidelity_mean": pos_mean,
            "pos_fidelity_std": pos_std,
            "neg_fidelity_mean": neg_mean,
            "neg_fidelity_std": neg_std,
            "num_runs": num_runs,
            "raw_accuracies": all_accuracies,
            "per_pair_pos_fidelities": per_pair_pos,
            "per_pair_neg_fidelities": per_pair_neg,
            "shots": shots,
            "accept_frac_mean": float(np.mean(self.accept_log)) if self.accept_log else None,
            "accept_frac_median": float(np.median(self.accept_log)) if self.accept_log else None,
            "per_circuit_counts": self.counts_log or None,   # emulate.save_counts: run 0, pos then neg circuits
            "elapsed_s_per_run": self.elapsed_log,
            "per_circuit_log": self.circuit_log,   # (acceptance, accepted signal/shots, unconditioned signal/shots); per run: pos then neg circuits
        }