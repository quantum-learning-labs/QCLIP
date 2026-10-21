# Why discard validates worse than non-discard (2026-10-16)

Data: the 10 paired baseline seeds (base_{nodiscard,discard}_s0-9, best.pt), val set (1829 pairs) and the
first 2000 train pairs, image-choice task. Script: qclip_snapshots/tools/gap_analysis.py, raw per-pair
results: qclip_snapshots/gap/base_*.pkl. Val accuracy here: non-discard 0.849±0.007, discard 0.802±0.014
(gap 0.047±0.017).

## Findings
1. The gap is mostly on verb negatives (~80% of it). Val accuracy by what differs in the negative image:
   verb 0.863 vs 0.779 (-8.4 pts, 855 pairs), object 0.916 vs 0.896 (-2.0, 542), subject 0.734 vs 0.728
   (-0.6, 432). (855/1829) x 8.4 pts = 3.9 of the 4.7 pts.
2. Structural cause for (1): in the curried circuits, verb information reaches the sentence wires only through
   the verb's 1-qubit `s` wire, which feeds the preposition box (e.g. "A girl sits on a bench":
   sits: n->s, on: s,n,n->out,n). That `s` wire is then post-selected (non-discard) or traced out (discard).
   Post-selection projects what it holds onto the output wires; tracing it out throws away whatever is still
   entangled with it. Subject and object also have direct paths into the preposition box, so they lose less.
   90% of val sentences trace out 4 non-sentence wires (including the verb's s wire) against 3 kept.
3. Discard text states are very mixed: median purity 0.36 (1/8 = 0.125 is fully mixed). Images are pure in both
   modes (the image circuit has only the 3 embedding qubits). Purer sentences score higher (0.82 in the purest
   quartile vs 0.79 in the most mixed); longer sentences trace more qubits and are more mixed (purity 0.60 at
   3 words -> 0.21-0.31 at 7-10).
4. Not the similarity function: Tr(rho sigma), fidelity, Pauli cosine and normalised HS give *identical*
   rankings (images are pure, so all are monotone in <psi|rho|psi>); trace distance is 0.3 pts lower.
   The choice of discard loss also doesn't close it (Pauli / UJ / HS test std 0.774 / 0.774 / 0.787).
5. Mostly generalisation, not fit: train accuracy 0.952 vs 0.935 (gap only 1.7 pts), val gap 4.7 pts.
   The gap appears early (val at epoch 5: 0.789 vs 0.749; epoch 10: 0.832 vs 0.779) and persists; discard
   peaks later (epoch 25 vs 17). Gradient norms per word type are similar (no vanishing gradients).
6. The discard image tower spreads images less: median overlap between the two images of a pair 0.474 vs
   0.432, and both methods lose most on the closest image pairs (0.71 vs 0.75).
   Fitting pure images to mixed text (best match = top eigenvector of rho) may pull images together.
7. Per-pair errors are shared: correctness correlates 0.71 across methods; only 5 pairs are reliably right
   for non-discard and wrong for discard (2 the other way). Discard is uniformly a bit worse, not broken on a subset.

## Leads to test (cheap, each fits stage 1/2 of the search)
- Post-select the verb's `s` wire and trace only the rest (hybrid), or route the verb into the out wires.
  Prediction: closes most of the verb-negative gap.
- Give the image tower environment qubits (mixed image states) so both sides have the same kind of state.
- Purity: Pauli purity_weight (stage 1 pw0 / pw0.5 runs), more text layers (tl3) on the box that absorbs `s`.
- Check how verb accuracy depends on the preposition/verb box depth.
