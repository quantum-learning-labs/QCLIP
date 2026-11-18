## Circuits, scaling and experimental setup

Every number below is taken from the code, configs, data files or job logs named next to it. Items marked **unknown** or **assumed** could not be traced to a measurement or a source.

### Circuits

- **Model family.** The qclip compositional variational quantum circuit: each sentence's CCG diagram becomes a circuit with one parametrised box per word. The ansatz is `custom_v5` (`modules/compilation/quantum/ansatz.py`): a Hadamard on every wire, then per layer Rz·Ry·Rz on every qubit followed by a ring of CX gates (qubit i controls i+1 mod N). Cups are removed by currying (`text.curry: true`). The readout is post-selected (`text.discard: false`).
- **Wire sizes.** Noun wires have n qubits (`text.n`), sentence and preposition wires one qubit (`text.s = text.p = 1`), and the output register has e qubits (`embedding_qubits`). Config: `hp_sc4i_nodiscard_e3_n*_tl*_qlr3e-3_s0/config.yaml`, model code commit `39974f7`.
- **Image side.** The CLIP ViT-B/32 image embedding passes through a 2-layer MLP image tower (`vision.method: mlp`), which prepares the image state on the e output qubits.
- **One decision.** An image choice on the SVO-Probes standard test set: does the caption match the positive or the negative image? Each decision is priced on two compute–uncompute (CU) circuits, one per image (`cu_pos`, `cu_neg`). Each circuit is the text circuit, then the inverse image-state preparation on the output register, then measurement of all qubits. The decision margin is m = U₀/P₀ per image, where P₀ is the post-selection acceptance and U₀ the post-selected overlap; the decision is d = m_pos − m_neg (`analysis/bench/calib_noise.py`). The O-readout variant (`O`) is computed but not used for pricing.
- **Sentences per size.** The modal ("typical") qubit width of the std test sentences at that size (`inf_engines.load`, `frames(typical=...)`). The first 3–5 such sentences are transpiled, 2 networks per size get greedy and random-greedy path searches and 1 gets cotengra, 40 decisions are used for the GPU energies, and 300 for the noise calibration.
- **Circuits for each size.** Untrained checkpoints from `analysis/bench/make_init_ckpt.py`: random-initialised parameters, used for structure only. They take the config of `hp_sc4i_nodiscard_e3_n9_tl2` with `text.n`, `text.layers` and `embedding_qubits` overridden. The lines use e = 3 with n ∈ {4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 32, 40}, giving typical widths of 16, 19, 22, 25, 28, 31, 34, 40, 49, 64, 79, 100 and 124 qubits.
- **Depth.** The log-depth rule is L = round(log₂ qubits): 16–22 q at 4 layers, 25–40 q at 5, 49–79 q at 6 and 100–124 q at 7. Fixed 2, 4 and 8 layers are shown as references.

### Scaling and point status

- **Measured.** GPU energy runs: statevector engines up to 31 q at 2 layers; exact tensor networks where the largest intermediate is ≤ 2²⁹ entries (`--tn_width_max 29`).
- **Upper bound.** Classical energy from the best contraction path found, as FLOP / η. A better path can only lower it.
- **Interpolated.** Depths 5–7 that have not been computed yet. Gate counts, durations and best-path log₁₀ FLOP are interpolated linearly in L between the 4- and 8-layer circuits of the same size. Each interpolated point is replaced by the real circuit when its job lands.
- **Statevector line.** The best of cuStateVec, the lazy torch statevector and Aer, measured at 2 layers to 31 q. It is scaled to other depths by the logical two-qubit gate count, and beyond 31 q extrapolated with an exponential fit (log₁₀ J per gate linear in qubits, over the 3 largest measured sizes).

### Classical side

- **Hardware.** NVIDIA L4 for every GPU energy point (`gpu_engines*.csv`), on the UCL CS cluster, with one process alone on the GPU. A run aborts if the idle power exceeds 100 W.
- **Energy measurement.** The NVML total-energy counter of the job's GPU, matched by UUID because NVML ignores `CUDA_VISIBLE_DEVICES` (fix 698a2be). Each run repeats until ≥ 10 s and excludes a warm-up run (`analysis/bench/inf_engines.py`). The figure uses gross energy per decision (including idle power); net = gross − idle × time. The band spans net to gross.
- **Host share.** 20 W × run time, one busy CPU core (**assumed**).
- **Engines.**
  - Aer GPU statevector (qiskit-aer-gpu-cu11 0.17.2, double precision).
  - cuStateVec through Aer (custatevec-cu11 1.6.0, single precision, default fusion), for the statevector and the density matrix.
  - Lazy torch statevector (torch 2.9.0).
  - Exact tensor network: opt_einsum 3.4.0 on torch complex64. The measured runs use the greedy path.
  - Tensor-network sampler: N₀ post-selected amplitude contractions.
  - MPS (torch, χ = 2–256): counted only when its margin error is ≤ 10% relative to exact and its decisions agree with exact on ≥ 95% of pairs.
- **Paths** (`analysis/bench/tw_bounds.py`).
  - Searches: opt_einsum greedy and random-greedy-128; cotengra 0.7.5 `HyperOptimizer` with greedy trials, a 600 s budget with subtree reconfiguration, and a hard stop at 1.5 × budget.
  - Energy: best-path FLOP / η, with η = 1.86 × 10¹⁰ FLOP/J (median of the compute-bound L4 runs).
  - Time: 1.31 × 10¹² FLOP/s.
  - A better path lowers only the compute part of a measured run.
- **One-off costs.** Path-search and transpile wall time × 20 W, shared by K = 1833/25 = 73.3 decisions per circuit shape (std test set).

### QPU side

- **Transpilation.** Qiskit 2.5.0 at `optimization_level=1` (`analysis/bench/qpu_esp.py`, `qpu_iqm.py`).
  - IBM: FakeKingston (qiskit-ibm-runtime 0.47.0), seeds 0–4, keeping the fewest two-qubit gates per circuit.
  - IonQ: basis {rz, ry, rx, rxx}, all-to-all, seed 0.
  - IQM: iqm-client 35.0.1 `IQMFakeAphrodite` (54-qubit square lattice, prx + cz), seeds 0–4.
- **Routing overhead.** Routed two-qubit gates divided by logical two-qubit gates, as a median per depth. Kingston: 3.94 (2 layers), 4.15 (4), 3.59 (5), 3.69 (8). IQM: 1.58 (4), 1.39 (5). Every size is priced at logical × median overhead; the spread of individual circuits enters the uncertainty.
- **Shots.** N = N₀ / F² with N₀ = 179 noiseless shots per decision for a 10% margin error. N₀ comes from the shot study (CPTP report §23.1). That accounting divides the per-shot variance by shots × acceptance, so 1/p₀ is included. The derivation of 179 is **not re-derived here**.
- **Fidelity.** ln F = a − ε·n₂q.
  - IBM: a = −0.187, ε = 3.31 × 10⁻³ per routed two-qubit gate. F is the least-squares slope of noisy on exact margins, from 54 density-matrix emulations on FakeKingston (6–13 q, 300 std pairs, 2²⁰ shots, CU readout).
  - IonQ: ε₂ = 4 × 10⁻³ and ε₁ = 2 × 10⁻⁴ (99.6% / 99.98%, **assumed**), with Kingston's a.
  - IQM: Kingston's fit scaled by the median-error ratios: ε = 3.31 × 10⁻³ × 5 × 10⁻³ / 1.8 × 10⁻³ and a = −0.187 × 3 × 10⁻² / 9.52 × 10⁻³. This is a model, not a calibration.
  - Like-for-like calibrations on the same 27 post-selected models (IQM fake Garnet, IonQ cloud simulator with noise model forte-1) are running and replace these constants when they land.
- **Energy per decision.**
  - IBM: [N (t_circ + 250 µs) + 2·2/190 s] × 25 kW.
  - IonQ: [N (n₁q·110 µs + n₂q·900 µs + 1.5 ms) + 2·10 s] × 5 kW.
  - IQM: N (1.92 ms + t_circ) × 24 kW.
  - Each plus transpile time × 20 W / K.
- **Sources.**
  - IQM: 24 kW typical mean total ([IQM Radiance spec](https://iqm.tech/products/iqm-radiance/)); 30 kW peak ([Mansfield et al. 2025](https://arxiv.org/abs/2509.12949)); median errors and CLOPS 2600 at QV 32 ([arXiv:2408.12433](https://arxiv.org/abs/2408.12433)); 1.92 ms = 100·10·100·5 / 2600 / 10⁵ shots.
  - IBM: 25 kW, 250 µs and 2 s job overhead are public figures (**assumed**); 26 kW System Two refrigerator ([arXiv:2304.14344](https://arxiv.org/abs/2304.14344)).
  - IonQ: timings from [Knitter et al. 2026](https://arxiv.org/abs/2605.02798). The 5 kW value is carried over from the earlier page and was **not re-verified** in that paper.

### Uncertainty

- **Method.** 2000 Monte Carlo draws; the bands are the 16th–84th percentiles; the central line is the current estimate (`efig_calc.py`, seed 12345). Classical and statevector bands are one-sided: the central estimate (best path found, gross energy) is their upper edge.
- **Workload.** The QPU routing overhead is drawn from the overheads of individual circuits at that depth. On the classical side one typical-width network per size was searched, so the per-sentence spread of contraction cost is **not covered**.
- **GPU energy.** U(net/gross, 1) × measured (median net/gross 0.47). Run-to-run repeats on an L4 are queued and will be added.
- **Classical upper bounds.** One-sided: log₁₀ FLOP − U(0, 1) × g, with g = 0.135 × log₁₀ FLOP, a single fit over 92 networks of the gain of the 600 s cotengra search over random-greedy-128 (residual 0.78 decades). It is used as the scale of a further improvement a better search could find. The band vanishes where the cost is set by the fixed per-run overhead (≥ 1 J of kernel launches) rather than by the path.
- **Interpolated depths.** Log-normal with σ = 0.35 decades, measured on the real 5-layer circuits.
- **Statevector extrapolation.** Student-t prediction interval.
- **QPU noise fit.** Bootstrap of the 27 post-selected Kingston calibration runs, applied to all backends until their own calibrations land.
- **QPU ranges.** Power, uniform: IBM 25–30 kW, IonQ 3.5–6.5 kW (**assumed**), IQM 24–30 kW. Time per shot: IBM repetition delay 100–500 µs (**assumed**), IonQ ×0.8–1.2 (**assumed**), IQM 0.96–1.92 ms (**assumed** lower end).
- **Not covered.** Better classical algorithms than those tried; hardware roadmaps; error mitigation or correction overheads; queueing.

### Software, hardware and commands

- **Software.** Cluster container `ubuntu2404`: Python 3.12.13, torch 2.9.0, qiskit 2.5.0, qiskit-aer-gpu-cu11 0.17.2, qiskit-ibm-runtime 0.47.0, qiskit-ionq 1.1.1, iqm-client 35.0.1, cuquantum-python-cu11 24.3.0.post0, custatevec-cu11 1.6.0.post1, opt_einsum 3.4.0, cotengra 0.7.5, numpy 2.5.1, nvidia-ml-py 13.610.43.
- **Hardware.** UCL CS SGE cluster: CPU nodes for structure, transpiles and path search; NVIDIA L4 for GPU energies; NVIDIA L40S or similar for the noise emulations.
- **Code.** Branch `energy-fig` (from `shotcost` 0f3ecf3), on quantum-learning-labs/QCLIP:
  - `analysis/bench/qpu_iqm.py`;
  - `analysis/bench/shotcost_noise.py` (iqm_garnet profile, `--dump`);
  - `analysis/bench/ionq_calib.py`;
  - `analysis/bench/efig/` (`efig_calc.py`, `efig_plot.py`, `efig_page.html`, `efig_ingest.sh`, job scripts in `job_scripts/`).
- **Reproduce the figure.**
  1. Run the jobs in `analysis/bench/efig/job_scripts/`: `job_cpu_l{5,6,7}.sh`, `job_iqm_l4.sh`, `job_calcounts.sh`, `job_gpu_rep.sh`, `mk_iqm.sh`, `job_ionq_dump.sh`.
  2. Copy the CSVs into `adv_data/`: `efig_ingest.sh` does this.
  3. Aggregate and compute: `python3 adv_agg.py && python3 efig_calc.py`.
  4. Draw: `python efig_plot.py efig.json efig_log2 log2`.
