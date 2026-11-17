import time, torch
import numpy as np
from tqdm import tqdm
from qiskit.quantum_info import Statevector

class Emulator:
    def __init__(self, config, backend, shots=1024, batch_size=64):
        self.config = config
        self.backend = backend
        self.shots = shots
        self.batch_size = batch_size

    def run_circuit(self, qc_pair, shots=None, batch_size=24):
        txt_circs, img_circs = qc_pair
        if len(txt_circs) != len(img_circs):
            raise ValueError("Text and image circuit lists must have the same length")

        result_array = []
        chunks = [
            (txt_circs[i:i + batch_size], img_circs[i:i + batch_size])
            for i in range(0, len(txt_circs), batch_size)
        ]

        for txt_chunk, img_chunk in tqdm(chunks, desc="Running Circuit Pairs"):
            for txt_qc, img_qc in zip(txt_chunk, img_chunk):
                txt_state = Statevector.from_instruction(txt_qc).data
                img_state = Statevector.from_instruction(img_qc).data
                if txt_state.shape != img_state.shape:
                    raise ValueError(
                        "Text and image circuits must use the same number of qubits "
                        "to compute a classical statevector overlap"
                    )
                txt_norm = np.linalg.norm(txt_state)
                img_norm = np.linalg.norm(img_state)
                if txt_norm == 0 or img_norm == 0:
                    result_array.append(0.0)
                    continue
                overlap = np.vdot(txt_state / txt_norm, img_state / img_norm)
                result_array.append(float(np.abs(overlap) ** 2))
        return torch.tensor(result_array, dtype=torch.float32)
    
    def run_experiment(self, pos_circs, neg_circs, shots=None, num_runs=5):
        shots = shots if shots is not None else self.shots
        torch.cuda.empty_cache()

        all_accuracies = []

        self.counts_log = []
        for run_idx in range(num_runs):
            self._run_idx = run_idx

            start_eval_time = time.time()
            pos_raw = self.run_circuit(pos_circs, shots=shots, batch_size=self.batch_size)
            neg_raw = self.run_circuit(neg_circs, shots=shots, batch_size=self.batch_size)
            elapsed = time.time() - start_eval_time

            correct = (pos_raw > neg_raw).float()
            accuracy = correct.mean().item()

            all_accuracies.append(accuracy)

            print(f" Trial {run_idx + 1} ({elapsed:.2f}s) | Accuracy: {accuracy * 100:.2f}% | Mean Pos Fid: {pos_raw.mean().item():.4f} | Mean Neg Fid: {neg_raw.mean().item():.4f}")
            
        return float(np.mean(np.array(all_accuracies)))