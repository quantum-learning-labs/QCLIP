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
  config.py          config loading, --set overrides, run ids, the command line shared by the scripts
  factory.py         builds an experiment from a config: models, loss, optimiser, data (load_experiment)
  common.py          device, seeding, pickle loading, import-by-path
  symbolic/          CCG parsing (Bobcat via lambeq), grammar types, the diagram -> tensor-network functor
  compilation/
    einsum.py        einsum string helpers, circuit size metrics
    quantum/         ansatz.py: tensor network -> circuit (curried ansatz, post-selection or discard); gates.py
                     circuits.py: Qiskit export and readout circuits; backends.py: device and noise models;
                     emul.py: shot-based emulation (Aer)
    classical/       tensor-network and neural compilers (classical baselines)
  models/
    text/            einsum_quantum.py (VQC text model), einsum_classical.py (TN), neural_model.py (MLP)
    vision/          CLIP features and their map to quantum states (quantum_map.py)
    fusion/          criteria.py (losses), trainer.py (trainer, distillation teacher), evaluator.py, run.py (runs, checkpoints)
  data_pipeline/     dataset retrieval (+ manifests/ of our curation), splits.py, image embedding, datasets and swap sets, DataEngine
scripts/             train.py, benchmark.py, emulate.py, retrieve.py, compare.py (paired comparison over splits)
configs/             vqc.yaml, tn.yaml, mlp.yaml (templates); data/ (one file per dataset); baseline/ (sweep bases); final/ (recommended configs)
notebooks/           train_vqc.ipynb, train_tn.ipynb: minimal training runs
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

Parsing uses the Bobcat CCG parser through lambeq. lambeq downloads the model to the directory in `BOBCAT_PATH`
(default `./bobcat`). You only need the parser to preprocess a dataset. Training reads the
parsed diagrams from disk.

Run every script from the repository root with the root on the path: `PYTHONPATH=. python scripts/<x>.py ...`
(or `uv run python ...`).

## Data

The supported datasets are SVO-Probes (the main benchmark), ARO (VG-Attribution and VG-Relation) and MSCOCO. A config
names its dataset (`dataset: "svo-probes"`), whose settings are in `configs/data/<name>.yaml`. Each dataset lives
under `data/<name>/`:

```
data/svo-probes/processed/
  all.pkl                the whole curated dataset: parsed captions with positive and negative images
  image_embeddings.pt    CLIP image features
```

- `python scripts/retrieve.py -cfg <config>` downloads the config's dataset, parses its captions and embeds its
  images with CLIP. Sources are listed in `modules/data_pipeline/retrieval.py` (`DATASET_SOURCES`). SVO-Probes comes
  from DeepMind's CSV and image URLs plus our curation in `modules/data_pipeline/manifests/svo-probes/` (the rows we
  keep and their captions rewritten as simple subject-verb-object sentences). ARO comes from the authors' release, with
  every image cropped to its row's box as in their evaluation.
- Splits are computed when the data is loaded, from the dataset's `split` setting (`modules/data_pipeline/splits.py`).
  Neither SVO-Probes nor ARO has an official training split, so both are stored whole (parsed once) and split by a
  seeded function that keeps groups together: every caption of a positive image (SVO-Probes, 60/20/20) or every row of a
  Visual Genome image (ARO, 70/15/15) lands in one split, chosen by a hash of (seed, group). The vocabulary is then
  closed: a val/test group with a word (word and wire types, one text-model tensor) that no training caption has moves
  to train, so every val/test word is trained. Datasets with an official split (MSCOCO) keep it.
- Every run records its split (`dataset.split_record` in the checkpoint's `config.yaml`, `split` in benchmark and
  emulation results): the spec, the sizes and a fingerprint `id`. Evaluating a checkpoint on a different split raises.
  For results over several splits, train each model on the same split seeds (`--set dataset.split.seed=k`) and compare
  with `scripts/compare.py <glob a> <glob b>`, which pairs runs by (split id, training seed) and refuses sets that
  differ.
- `--set dataset.split=legacy` gives the SVO-Probes split of all results before 2026-11-13 (`weights/` and the numbers
  in `configs/final/`). It was made by row, so captions and images are shared between its train, val and test sets.
- Subject/object-swap sets are derived when they are loaded: a test entry with `derive: swap` (`swap_full`, 1777
  pairs on the legacy test split) swaps the two nouns of every caption that has two, dropping symmetric verbs and swaps
  that leave the sentence unchanged. `swap_negatives: true` adds the swapped captions to the training set as extra
  negatives.

## Usage

### Train

```bash
PYTHONPATH=. python scripts/train.py -cfg configs/final/vqc_nodiscard.yaml --seed 0 --set run_name=nodiscard_s0
PYTHONPATH=. python scripts/train.py -cfg configs/vqc.yaml --seed 0 --set run_name=my_run \
    --set text.discard=true --set text.layers=3 --set qlr=0.003
```

All scripts take the same arguments: `-cfg` (config), `-cp` (checkpoint), `-o` (output), `--seed` and `--set`.
- `--set key=value` overrides any config value (dotted keys for nested ones) and can be repeated, e.g.
  `--set epochs=2` for a smoke test. Values are parsed as YAML 1.1, so write `qlr=0.003`: `3e-3` parses as a string.
- `-cp <ckpt.pt>` warm-starts the text and image weights from another checkpoint (config key `init_from`), for
  example a discard model from a post-selected one.
- Checkpoints go to `checkpoints/<dataset>/<model_type>/<run_name>/`, with `best.pt` chosen on validation
  `hard_neg_acc` and the resolved `config.yaml`. Use a new `run_name` for every run: run directories are never reused.
- Every run is seeded with deterministic kernels and records its git commit. Metrics go to MLflow (a sqlite file,
  `mlflow_db` in the config): `mlflow ui --backend-store-uri sqlite:///mlf_dbs/<file>.db`.

### Evaluate

```bash
PYTHONPATH=. python scripts/benchmark.py -cp checkpoints/svo-probes/vqc/nodiscard_s0/best.pt
```

This evaluates on every `dataset.test` entry with the config saved next to the checkpoint. On SVO-Probes that
means `std` (image choice, the primary metric) and `swap_full` (text choice). It writes a JSON next to
the checkpoint (`-o` sets another path) and never overwrites.

### Emulate on a simulated quantum device

`scripts/emulate.py` exports the trained circuits to Qiskit and runs them on Aer with finite shots and,
optionally, hardware noise:

```bash
# noiseless shot sweep on the std set
PYTHONPATH=. python scripts/emulate.py -cp <ckpt>/best.pt --set emulate.set=std --set "emulate.shots=[64,256,1024]"
# noisy, using the calibration of a real IBM device
PYTHONPATH=. python scripts/emulate.py -cp <ckpt>/best.pt --set emulate.shots=1024 --set emulate.set=std \
    --set emulate.noise=true --set emulate.profile=fake_kingston --set emulate.backend=statevector \
    --set emulate.aer_options.batched_shots_gpu=false --set emulate.aer_options.shot_branching_enable=false
# IonQ's cloud simulator with the Forte noise model (API key from $IONQ_API_KEY or ~/.ionq_key; never a QPU)
PYTHONPATH=. python scripts/emulate.py -cp <ckpt>/best.pt --set emulate.shots=1024 --set emulate.set=std --set emulate.limit=40 \
    --set emulate.noise=true --set emulate.provider=ionq --set emulate.profile=forte-1 --set emulate.circuit_batch=40
# the same with an IQM device (Garnet, 20 qubits, native r and cz gates)
PYTHONPATH=. python scripts/emulate.py -cp <ckpt>/best.pt --set emulate.shots=1024 --set emulate.set=std \
    --set emulate.noise=true --set emulate.provider=iqm --set emulate.profile=garnet
```

Emulation settings are `emulate.*` config keys, changed with `--set` as in `train.py`:
- `shots`: shot budget(s) per circuit (circuits are compiled once and run at each); by default it follows
  `resolution`. `limit: N` emulates only the first N pairs.
- `provider` and `profile`: the device to emulate; never hardware. `ibm` (default): any `qiskit_ibm_runtime` fake
  backend (a calibration snapshot, run locally on Aer), default `fake_miami`. `iqm`: an IQM fake backend from
  `iqm-client[qiskit]` (`garnet` (default), `apollo`, `aphrodite`, `adonis`) with IQM's own noise model, on Aer.
  `ionq`: IonQ's cloud simulator with one of IonQ's noise models (`forte-1` (default), `aria-1`, ...; `noise: false` is
  the ideal simulator). It needs an IonQ API key in `$IONQ_API_KEY` or in the file `emulate.ionq_key_file` (default
  `~/.ionq_key`, keep it `chmod 600`). The code only ever opens IonQ's simulator; seeds go to its `sampler_seed`.
- `noise_profile: honest` (IBM profiles only) adds scheduled idle decoherence and coherent ZZ to the backend noise model; its
  knobs are `emulate.honest.<knob>`. The default, `backend`, is Aer's `NoiseModel.from_backend`.
- `backend` is the Aer method. Density matrix is faster for small circuits. `dm_max_qubits: N` runs circuits
  with at most N active qubits as density matrices and the rest with `backend`.
- `aer_options.<option>` sets any Aer option. For noisy statevector runs keep the two shown above, or GPU memory
  grows until the job dies.
- `num_runs` repeats the shot sampling only (shot noise), not training variance; `circuit_batch` circuits go to Aer
  per call; `save_counts: true` stores every circuit's counts.

## Configuration reference

The configs to start from are in `configs/final/`:

| config | model |
|---|---|
| `vqc_nodiscard.yaml` | post-selection, 3 text layers, qlr 0.003 (best overall) |
| `vqc_discard.yaml` | discard from scratch, 3 text layers, Pauli loss |
| `vqc_discard_distill.yaml` | discard, warm-started from the post-selected model of the same seed and distilled from it (best discard); the run recipe is in the file header |

The main keys:

| key | meaning |
|---|---|
| `model_type` | `vqc` (quantum), `tn` (classical tensor network), `mlp` |
| `qlr`, `clr` | base learning rates for the circuit and the classical parameters |
| `embedding_qubits` | qubits of the sentence (output) wire and of the image state |
| `loss` | `fs` (Fubini-Study, post-selection), `pauli`, `uj`, `hs` (discard); default by `text.discard` |
| `purity_weight` | purity regulariser of `pauli` (default 0.1; every final recipe uses 0) |
| `swap_negatives` | add subject/object-swapped captions to the training set as text negatives |
| `distill_from`, `distill_kd_weight`, `distill_tau` | distil from a frozen post-selected teacher checkpoint |
| `text.ansatz` | word-box circuit (default `custom_v5`); add one by subclassing `BaseAnsatz` (define `ansatz()`) and registering it in `ANSATZE` |
| `text.layers` | ansatz depth per word, the most important knob |
| `text.curry` | curried ansatz (cups removed); `text.spider` must stay `false` |
| `text.discard` | discard (partial trace) instead of post-selection |
| `text.root` | `word` (default): the root word's open wires become the output, so a root word has its own parameters; `head`: one shared head box per root type maps the open wires to the output |
| `vision.method`, `vision.layers` | image map: `mlp`, `pca` or `amp` (amplitude encoding) |
| `emulate.*` | settings of `emulate.py` (set, backend, noise, profile, `measurement_method`: `compute_uncompute` or `destructive_swap`, ...) |

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
| discard, warm start | 0.843 ± 0.010 | `vqc_discard.yaml` + `-cp` (post-selected checkpoint) |
| discard, from scratch | 0.813 ± 0.013 | `configs/final/vqc_discard.yaml` |

Findings so far:
- Text depth is the most important setting for both readouts (3 layers >> 2 >> 1).
- The discard gap sits on subject negatives; on verb and object negatives the two readouts are within 2 points.
  Other losses (Bures angle, exact fidelity, purity schedules), extra traced qubits and mixers did not close it
  (that code was removed; it is on the research branches).
- Warm-starting from a post-selected model and distilling its scores closes most of the gap (10/10 paired wins
  over the warm start alone).
- In noisy emulation with realistic device calibrations, post-selection still wins at m=4 traced qubits.
  At m=8 the order reverses and plain discard is best: noise shrinks the margins and post-selection also
  loses most of its shots.

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
