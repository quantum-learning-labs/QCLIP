# Discard vs non-discard gap analysis (2026-10-16/17)
- REPORT_v1.md: first pass (gap concentrated on verb negatives, purity, generalisation).
- REPORT_v2.md: post-selection vs partial trace on the same weights (readout swap), distinguishability,
  gradient scaling with the number of non-sentence wires, margins, effective dimension.
- gap_analysis.py / gap_summ.py: v1 per-pair scores (several similarity functions, negative types).
- readout_swap.py (+ readout_swap_fns.py) / readout_summ.py: v2 readout swap, run from the qclip code dir
  on a GPU: `PYTHONPATH=. python analysis/discard_gap/readout_swap.py out.pkl <ckpt_dir>...`
- grad_probe.py: gradient statistics vs m; expects readout_swap_fns.py at ~/qclip_hpc/gap/ (cluster layout).
Raw outputs: qclip_snapshots/gap/ on tails and ~/qclip_hpc/gap/ on the cluster.
