# QCLIP: compositional vision-language models as variational quantum circuits

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-emulation-6929C4.svg)](https://qiskit.github.io/qiskit-aer/)
[![lambeq](https://img.shields.io/badge/lambeq-CCG_parsing-green.svg)](https://cqcl.github.io/lambeq/)

This is the official implementation of **"Meaning Representations as Variational Quantum Circuits"** (LREC 2026).

*Tilen G. Limbäck-Stokin, Tanishka A. Birdavade, Kin Ian Lo, Mehrnoosh Sadrzadeh*, Quantum Learning Labs, University College London.

## Overview

Contrastive vision-language models such as CLIP read text as an unstructured sequence and often behave like
bags of words. **QuLIP** (Quantum Language-Image Pretraining) builds the text encoder from the sentence's grammar instead. A CCG parse becomes a
variational quantum circuit (VQC). Each word is a parameterised unitary, and each grammatical composition
contracts that word's wires with the wires of its arguments. Images are CLIP-embedded and mapped by a small
learned map to a quantum state on the same number of qubits. Text and image states are then aligned with a
contrastive (InfoNCE) loss on a quantum similarity. The models have 10k-100k parameters.

![QuLIP pipeline](assets/pipeline.png)
*Figure 1: The pipeline. Images are classically embedded and mapped to quantum states. Sentences are parsed with
CCG and compiled into circuits whose topology follows the parse. Alignment uses a quantum similarity.*

![CCG to VQC compilation](assets/circuit.png)
*Figure 2: The circuit for "Alice likes Bob". The grammar fixes the circuit topology.*

### Two readouts: post-selection and discard

A compiled sentence leaves non-sentence wires that must be removed. The code supports two ways to do this,
chosen by `text.discard`:

| | post-selection (`discard: false`) | discard / CPTP (`discard: true`) |
|---|---|---|
| non-sentence wires | projected onto Bell states (cups) and renormalised | traced out (partial trace) |
| text state | pure statevector | density matrix |
| default loss | Fubini-Study InfoNCE (`fs`) | Pauli-vector InfoNCE (`pauli`) |
| hardware cost | keeps only the ~2^-m of shots that pass the post-selection | uses every shot |

Post-selection gives the strongest models. On hardware, however, it throws away almost all shots as the number m
of traced qubits grows. Much of the recent work in this repo makes discard (CPTP) models competitive; see
[Results](#results).

## Repository layout

```
modules/
  symbolic/          CCG parsing (Bobcat via lambeq), grammar types, the diagram -> tensor-network functor
  compilation/
    quantum/         ansatz.py: tensor network -> circuit (curried ansatz, discard, sinks, mixers, Phi block)
                     gates.py: batched gates; emul.py: Qiskit/Aer circuit export and emulation
    classical/       tensor-network and neural compilers (classical baselines)
  models/
    text/            einsum_quantum.py (VQC text model), einsum_classical.py (TN), neural_model.py (MLP)
    vision/          CLIP features and their map to quantum states (quantum_map.py)
    fusion/          criteria.py (losses), engine.py (trainer, evaluator, run manager), distill.py (teacher)
  data_pipeline/     dataset retrieval, image embedding, datasets and collate functions, DataEngine
  utils/             factory (builds an experiment from a config), quantum/tensor ops, visualisation
scripts/             train.py, benchmark.py, emulate.py, retrieve.py, make_swap.py, run_hp.sh, run_baseline.sh
configs/             vqc.yaml, tn.yaml, mlp.yaml (templates); baseline/ (sweep bases); final/ (recommended configs)
analysis/            analysis scripts: cptp/ (discard vs post-selection), discard_gap/ (gap reports v1, v2)
notebooks/           train_{vqc,tn,mlp}.ipynb
assets/              figures
```

## Installation

You need Python 3.12 and a CUDA GPU. Training and emulation both run on the GPU; emulation uses Aer with cuStateVec.

```bash
git clone https://github.com/quantum-learning-labs/QCLIP.git
cd QCLIP
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt      # pinned environment, includes CLIP (from git), lambeq, qiskit-aer
```

Parsing uses the Bobcat CCG parser through lambeq. lambeq downloads the model to the directory you give it
(`PARSER_PATH` in `scripts/retrieve.py`). You only need the parser to preprocess a dataset. Training reads the
parsed diagrams from disk.

Run every script from the repository root with the root on the path: `PYTHONPATH=. python scripts/<x>.py ...`
(or `uv run python ...`).

## Data

The supported datasets are SVO-Probes (the main benchmark), ARO (attribution and relation) and MSCOCO. Each
dataset lives under `data/<name>/`:

```
data/svo-probes/processed/
  train.pkl  val.pkl  test.pkl     parsed captions with positive and negative images
  swap.pkl                         95 subject/object-swap pairs (text choice)
  swap_full.pkl  swap_animate.pkl  larger swap sets built with make_swap.py
  image_embeddings.pt              CLIP image features
```

- `scripts/retrieve.py` downloads a dataset, parses its captions and embeds its images with CLIP. Set
  `DATASET` and `PARSER_PATH` at the bottom of the script.
- `scripts/make_swap.py <in.pkl> <out.pkl> [--keep_all]` builds a subject/object-swap set from any split. It drops
  symmetric verbs and swaps that leave the sentence unchanged, and it never overwrites a file. With `--keep_all` it
  keeps every row and marks the real swaps, which is the input format for training with `swap_negatives`.

Set the dataset in the config's `dataset:` block. `configs/vqc.yaml` has SVO-Probes active and ARO parked as `_dataset`.

## Usage

### Train

```bash
PYTHONPATH=. python scripts/train.py --config configs/final/vqc_nodiscard.yaml --seed 0 --run_name nodiscard_s0
PYTHONPATH=. python scripts/train.py --config configs/vqc.yaml --seed 0 --run_name my_run \
    --set text.discard=true --set text.layers=3 --set qlr=0.003
```

- `--set key=value` overrides any config value (dotted keys for nested ones) and can be repeated.
  Values are parsed as YAML 1.1, so write `qlr=0.003`: `3e-3` parses as a string.
- `--init_from <ckpt.pt>` warm-starts the text and image weights from another checkpoint, for example a discard
  model from a post-selected one.
- `--epochs N` overrides the epoch count (for smoke tests).
- Checkpoints go to `checkpoints/<dataset>/<model_type>/<run_name>/`, with `best.pt` chosen on validation
  `hard_neg_acc` and the resolved `config.yaml`. Use a new `run_name` for every run: run directories are never reused.
- Every run is seeded with deterministic kernels and records its git commit. Metrics go to MLflow (a sqlite file,
  `mlflow_db` in the config): `mlflow ui --backend-store-uri sqlite:///mlf_dbs/<file>.db`.

### Evaluate

```bash
PYTHONPATH=. python scripts/benchmark.py -cp checkpoints/svo-probes/vqc/nodiscard_s0/best.pt
```

This evaluates on every `dataset.test` entry with the config saved next to the checkpoint. On SVO-Probes that
means `std` (image choice, ~1.8k pairs, the primary metric) and `swap` (text choice). It writes a JSON next to
the checkpoint (`-o` sets another path) and never overwrites.

### Emulate on a simulated quantum device

`scripts/emulate.py` exports the trained circuits to Qiskit and runs them on Aer with finite shots and,
optionally, hardware noise:

```bash
# noiseless shot sweep on the std set
PYTHONPATH=. python scripts/emulate.py -cp <ckpt>/best.pt --eval_set std --shots 64,256,1024
# noisy, using the calibration of a real IBM device
PYTHONPATH=. python scripts/emulate.py -cp <ckpt>/best.pt --eval_set std --shots 1024 --noise \
    --profile fake_kingston --method statevector \
    --aer batched_shots_gpu=false --aer shot_branching_enable=false
```

- `--profile` picks any `qiskit_ibm_runtime` fake backend (a calibration snapshot); the default is `fake_miami`.
- `--noise_profile honest` adds scheduled idle decoherence and coherent ZZ to the backend noise model, and its
  knobs can be changed with `--honest k=v`. The default, `backend`, is Aer's `NoiseModel.from_backend`.
- `--method` sets the Aer method. Density matrix is faster for small circuits. `--dm_max_qubits N` runs circuits
  with at most N active qubits as density matrices and the rest with `--method`.
- For noisy statevector runs, keep the two `--aer` flags shown above, or GPU memory grows until the job dies.
- `--num_runs` repeats the shot sampling only (shot noise), not training variance.

### Hyperparameter sweeps

```bash
bash scripts/run_hp.sh <tag> <method> <variant> [--emulate] <seed>... -- --set k=v ...
```

This trains `configs/baseline/vqc_<method>.yaml` (`nodiscard`, `discard`, `discard_hs`, `discard_uj`) with the
overrides after `--`, for each seed, then benchmarks `best.pt`. Checkpoints go to
`checkpoints/svo-probes/vqc/hp_<tag>_<method>_<variant>_s<seed>/` and results to `results/hpsearch/<tag>/`.
Select variants on validation `hard_neg_acc`, not on test.

## Configuration reference

The configs to start from are in `configs/final/`:

| config | model |
|---|---|
| `vqc_nodiscard.yaml` | post-selection, 3 text layers, qlr 3e-3 (best overall) |
| `vqc_discard.yaml` | discard from scratch, 3 text layers, Pauli loss |
| `vqc_discard_distill.yaml` | discard, warm-started from the post-selected model of the same seed and distilled from it (best discard); the run recipe is in the file header |

The main keys:

| key | meaning |
|---|---|
| `model_type` | `vqc` (quantum), `tn` (classical tensor network), `mlp` |
| `qlr`, `clr` | base learning rates for the circuit and the classical parameters |
| `embedding_qubits` | qubits of the sentence (output) wire and of the image state |
| `loss` | `fs` (Fubini-Study, post-selection), `pauli`, `uj`, `hs`, `bures` (discard); default by `text.discard` |
| `learn_temperature` | learnable logit scale |
| `purity_weight`, `spread_weight` | loss regularisers (purity: `pauli`; spread: `pauli`, `hs`, `fs`) |
| `hard_negatives` | add each caption's negative image to the softmax |
| `swap_negatives` | add subject/object-swapped captions as text negatives (train split from `make_swap.py --keep_all`) |
| `distill_from`, `distill_kd_weight`, `distill_tau`, `distill_state_weight` | distil from a frozen post-selected teacher checkpoint |
| `init_by_symbol` | warm start by symbol name (when the two circuits differ, e.g. extra mixer gates) |
| `text.layers` | ansatz depth per word, the most important knob |
| `text.curry` | curried ansatz (cups removed); `text.spider` must stay `false` |
| `text.discard` | discard (partial trace) instead of post-selection |
| `text.postselect_types` | with discard: still post-select wires of these atomic types (hybrid) |
| `text.mixer_layers`, `text.mixer_ansatz` | entangling layers on the sentence wires before the trace; `cr` is identity at angle 0 (safe for warm starts) |
| `text.sinks` | extra traced qubits coupled in by the mixer (needs `mixer_layers >= 1`) |
| `text.phi_block` | a fixed, parameter-free discard block whose channel reproduces post-selection's decisions (research use) |
| `vision.method`, `vision.layers` | image map: `mlp`, `pca` or `amp` (amplitude encoding) |
| `vision.env_qubits` | traced environment qubits, giving mixed image states (discard only) |
| `emulate.*` | defaults for `emulate.py` (set, shots, backend, `measurement_method`: `compute_uncompute` or `destructive_swap`) |

## Results

### Published results (LREC 2026)

| Model | Parameters | SVO-Swap | ARO Attribution | ARO Relation |
| :-- | :-- | :-- | :-- | :-- |
| QBoW | 100K | 50.00% | 50.00% | 50.00% |
| MicroCLIP | 100K | 68.42% | 50.85% | 51.05% |
| CLIP | 63M | 57.89% | 61.00% | 51.53% |
| **QuLIP** (CCG-VQC) | ~90K | **83.16%** | **71.19%** | **57.33%** |

### Current results on SVO-Probes

Test `std` hard-negative accuracy (image choice, ~1.8k pairs), mean ± std over 10 seeds:

| model | accuracy | config |
|---|---|---|
| post-selection, 3 text layers | 0.874 ± 0.011 | `configs/final/vqc_nodiscard.yaml` |
| discard, warm start + distillation | 0.850 ± 0.011 | `configs/final/vqc_discard_distill.yaml` |
| discard, warm start | 0.843 ± 0.010 | `vqc_discard.yaml` + `--init_from` |
| discard, from scratch | 0.813 ± 0.013 | `configs/final/vqc_discard.yaml` |

Findings so far:
- Text depth is the most important setting for both readouts (3 layers >> 2 >> 1).
- The discard gap sits on subject negatives; on verb and object negatives the two readouts are within 2 points.
  Other losses (Bures, exact fidelity, purity schedules) did not close it.
- Warm-starting from a post-selected model and distilling its scores closes most of the gap (10/10 paired wins
  over the warm start alone).
- In noisy emulation with realistic device calibrations, post-selection still wins at m=4 traced qubits.
  At m=8 the order reverses and plain discard is best: noise shrinks the margins and post-selection also
  loses most of its shots.

## Analysis scripts

`analysis/` holds the scripts behind the discard vs post-selection study: readout swaps (the same weights read by
both methods), gradient probes, margin bounds, noise and shot-cost summaries, branch decomposition and loss
checks. See [analysis/cptp/README.md](analysis/cptp/README.md) and
[analysis/discard_gap/README.md](analysis/discard_gap/README.md).

## Citation

```bibtex
@inproceedings{limbackstokin2026meaning,
  title={Meaning Representations as Variational Quantum Circuits},
  author={Limb\"{a}ck-Stokin, Tilen G. and Birdavade, Tanishka A. and Lo, Kin Ian and Sadrzadeh, Mehrnoosh},
  booktitle={LREC},
  year={2026},
  organization={Quantum Learning Labs, University College London}
}
```

## Contact

- Tilen G. Limbäck-Stokin: tilen.limback-stokin.21@ucl.ac.uk
- [Quantum Learning Labs, UCL](https://www.ucl.ac.uk/engineering/computer-science/research/research-groups-and-centres/programming-principles-logic-and-verification-group/quantum-learning-labs)

## Key references

**Quantum NLP**
- Kartsaklis et al. (2021). *lambeq: An Efficient High-Level Python Library for Quantum NLP.* [arXiv:2110.04236](https://arxiv.org/abs/2110.04236)
- Yeung & Kartsaklis (2021). *A CCG-Based Version of the DisCoCat Framework.* [arXiv:2105.07720](https://arxiv.org/abs/2105.07720)
- Lorenz et al. (2023). *QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer.* [arXiv:2102.12846](https://arxiv.org/abs/2102.12846)
- Coecke, Sadrzadeh & Clark (2010). *Mathematical Foundations for a Compositional Distributional Model of Meaning.* [arXiv:1003.4394](https://arxiv.org/abs/1003.4394)

**Vision-language models and compositionality**
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Yuksekgonul et al. (2023). *When and why Vision-Language Models behave like Bags-of-Words, and what to do about it?* [arXiv:2210.01936](https://arxiv.org/abs/2210.01936)
- Lo et al. (2025). *DisCoCLIP: A Distributional Compositional Tensor Network Encoder for Vision-Language Understanding.* [arXiv:2509.21287](https://arxiv.org/abs/2509.21287)
- Lewis et al. (2024). *Does CLIP Bind Concepts? Probing Compositionality in Large Image Models.* [arXiv:2212.10537](https://arxiv.org/abs/2212.10537)

**Quantum machine learning**
- Biamonte et al. (2017). *Quantum Machine Learning.* [arXiv:1611.09347](https://arxiv.org/abs/1611.09347)
- Abbas et al. (2021). *The Power of Quantum Neural Networks.* [arXiv:2011.00027](https://arxiv.org/abs/2011.00027)
- Cerezo et al. (2021). *Cost Function Dependent Barren Plateaus in Shallow Parametrized Quantum Circuits.* [arXiv:2001.00550](https://arxiv.org/abs/2001.00550)
- Sim, Johnson & Aspuru-Guzik (2019). *Expressibility and Entangling Capability of Parameterized Quantum Circuits.* [arXiv:1905.10876](https://arxiv.org/abs/1905.10876)

**Linguistics**
- Steedman & Baldridge (2011). *Combinatory Categorial Grammar.*
- Steedman (2000). *The Syntactic Process.* MIT Press.
