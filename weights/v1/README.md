# qclip v1 weights

The final post-selected CCG-VQC of the v1 release (`REPRODUCE.md`), one run per split seed k = 0–4 (training seed k).
Each `k<k>/` has the best-validation `best.pt`, the full resolved `config.yaml` it was trained with (seed, split record
and id, git commit) and `metrics.json` (its test results, `benchmark.py` output). Only text parameters are trained;
the image is the frozen CLIP ViT-B/32 vector, amplitude-encoded on 9 qubits.

## `svo-probes/post_selection/` (3 qubits per type, 5 layers, 19,485 text parameters; commit 7995aad)

| k | split id | best val | std | SwapFull | Swap95 |
|---|---|---|---|---|---|
| 0 | `8db5720edc4c2bc3` | 0.705 | 0.646 | 0.966 | 0.863 |
| 1 | `51a52f4736f82739` | 0.657 | 0.668 | 0.918 | 0.547 |
| 2 | `8c7eb29b39419d1a` | 0.697 | 0.676 | 0.912 | 0.800 |
| 3 | `85c52baf551cd51f` | 0.694 | 0.669 | 0.944 | 0.789 |
| 4 | `326f6c2f3f89ab7f` | 0.713 | 0.702 | 0.961 | 0.873 |
| mean ± SD | | | 0.673 ± 0.020 | 0.941 ± 0.024 | 0.775 ± 0.132 |

## `aro/post_selection/` (3 qubits per type, 3 layers, 59,130 text parameters; commit 311ee4c)

| k | split id | best val | Attribution | Relation |
|---|---|---|---|---|
| 0 | `0cc9e231ec874787` | 0.538 | 0.540 | 0.509 |
| 1 | `4fe905156665a616` | 0.561 | 0.583 | 0.530 |
| 2 | `32444ae892f1018d` | 0.557 | 0.556 | 0.524 |
| 3 | `fbd824cac093170b` | 0.540 | 0.530 | 0.530 |
| 4 | `af7239405c1f9d01` | 0.552 | 0.522 | 0.535 |
| mean ± SD | | | 0.547 ± 0.024 | 0.526 ± 0.010 |

Test accuracy counts a tie as ½ (`*_hard_neg_acc_tie_half`). Re-evaluate a run with

```bash
python scripts/benchmark.py -cp weights/v1/svo-probes/post_selection/k0/best.pt -o out.json
```

which reads the `config.yaml` next to the checkpoint and refuses data whose split differs from the recorded one.
