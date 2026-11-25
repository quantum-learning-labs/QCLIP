### svo-probes

| model | text params | std | swap_full | swap95 |
|---|---|---|---|---|
| Zero-shot CLIP ViT-B/32 | - | 0.833 ± 0.005 | 0.842 ± 0.015 | 0.589 ± 0.000 |
| Zero-shot OpenCLIP ViT-B/32 (LAION-2B) | - | 0.873 ± 0.009 | 0.722 ± 0.022 | 0.621 ± 0.000 |
| Zero-shot BLIP (ITM) | - | 0.945 ± 0.005 | 0.751 ± 0.039 | 0.632 ± 0.000 |
| Zero-shot BLIP (ITC) | - | 0.922 ± 0.007 | 0.760 ± 0.009 | 0.368 ± 0.000 |
| QBoW (5 layers) | 24,435 | 0.669 ± 0.015 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| MicroCLIP-16 (matched) | 19,504 | 0.733 ± 0.021 | 0.809 ± 0.033 | 0.528 ± 0.074 |
| CCG tensor network (bond 2) | 364,360 | 0.603 ± 0.027 | 0.683 ± 0.077 | 0.587 ± 0.121 |
| CCG-VQC, post-selection (3 qubits/type, 5 layers) | 19,485 | 0.673 ± 0.020 | 0.929 ± 0.036 | 0.760 ± 0.161 |

### aro

| model | text params | att | rel |
|---|---|---|---|
| Zero-shot CLIP ViT-B/32 | - | 0.629 ± 0.010 | 0.509 ± 0.006 |
| Zero-shot OpenCLIP ViT-B/32 (LAION-2B) | - | 0.584 ± 0.012 | 0.494 ± 0.010 |
| Zero-shot BLIP (ITM) | - | 0.872 ± 0.006 | 0.549 ± 0.003 |
| Zero-shot BLIP (ITC) | - | 0.777 ± 0.009 | 0.517 ± 0.006 |
| QBoW (3 layers) | 141,912 | 0.500 ± 0.001 | 0.500 ± 0.001 |
| MicroCLIP-24 (matched) | 59,134 | 0.498 ± 0.004 | 0.515 ± 0.009 |
| CCG tensor network (bond 2) | 288,720 | 0.611 ± 0.006 | 0.550 ± 0.006 |
| CCG-VQC, post-selection (3 qubits/type, 3 layers) | 59,130 | 0.543 ± 0.007 | 0.528 ± 0.007 |

