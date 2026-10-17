# CPTP (discard) vs post-selected training: analysis scripts (2026-10-17)
Report: the Claude Docs page "CPTP vs Post-selected Training in QCLIP: Theory and Experiments".
- readout_swap.py (+ readout_swap_fns.py): read the same weights by post-selection and by partial trace;
  takes checkpoint dirs (best.pt) or epoch_*.pt files (RS_TRAIN_N env). readout_summ.py: tables (E1, E2).
- dyn_summ.py: per-epoch dynamics from readout_swap over epoch_*.pt (E5). Train with --set checkpoint_every=1.
- bound_check.py: margin budget |Delta| <= sqrt(2 (P - 1/d)(1 - F)) on per-pair data (E3).
- grad_probe.py: gradient magnitude vs number of non-sentence wires (E4).
- emu_summ.py / noise_summ.py: emulation shot sweeps and noiseless / baseline-noise / honest-noise tables (E6).
- probe_backend.py: fake-backend calibration numbers; aer_gpu_diag.py: Aer GPU method check.
- hp_summ_local.py: stdlib-only summary of results/hpsearch JSONs.
Emulation: emulate.py --shots --eval_set --noise --noise_profile {backend,honest} --method; always on GPU.
