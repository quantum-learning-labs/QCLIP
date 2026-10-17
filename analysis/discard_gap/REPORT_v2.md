# Discard vs non-discard: post-selection, partial trace, and what it does to training (v2, 2026-10-17)

Supersedes v1 (`qclip_snapshots/gap/REPORT.md`, which stays as it was). New measurements:
`readout_swap.py` (both readouts of the same trained weights) and `grad_probe.py` (gradients vs number of
non-sentence wires), run as cluster GPU jobs on the 10 paired baseline seeds (2 text layers) plus the
stage-1 runs with 1 and 3 text layers (2-3 seeds each, so treat L1/L3 rows as indicative).
Accuracy = image-choice hard-negative accuracy on val (1829 pairs) or the first 2000 train pairs.

## 1. What the two models actually compute

Both models use exactly the same circuit and parameters. Every sentence circuit is a unitary U on the 3
sentence wires plus m other wires (m = 4 for 90% of sentences), all started in |0>:

    U|0...0> = |Psi> = sum_k sqrt(p_k) |psi_k> (x) |k>        (k runs over the 2^m states of the other wires)

* **Non-discard (post-selection)** keeps one branch: the sentence is |psi_0>, obtained with probability p_0.
  The map from the circuit to the sentence is a single Kraus operator K_0 = (I (x) <0|) U, followed by
  renormalisation. This is a trace-*decreasing* CP map made trace-preserving by dividing by p_0, which is a
  non-linear operation on states.
* **Discard (partial trace)** keeps all branches: rho = sum_k p_k |psi_k><psi_k|. This is the CPTP channel
  whose Stinespring dilation is U itself. The post-selected state is literally one term of this mixture,
  with weight p_0.

So "discard" is not a different model class bolted on. It is the same unitary read out differently:
* post-selection conditions on one measurement outcome of the other wires;
* the partial trace averages over all outcomes.

## 2. Expressivity: unitaries + post-selection vs CPTP maps

Neither set contains the other:

| | post-selection (non-discard) | partial trace (discard) |
|---|---|---|
| outputs | pure states only (14 real dims for 3 qubits) | mixed states (63 real dims) |
| allowed maps | any linear map K up to scale (post-selection is as powerful as PostBQP=PP) | CPTP channels only: convex mixtures, no filtering |
| distinguishability | can *increase* it (filtering removes shared branches) | can only decrease it (data-processing inequality: D(Phi rho, Phi sigma) <= D(rho, sigma)) |
| cost on hardware | needs ~1/p_0 shots per sample, p_0 ~ 2^-m | none; every shot counts |
| compositional semantics | DisCoCat cups/caps are post-selected Bell effects, so composition = linear-map application | composition = channel composition, contractive |

**Seen in practice:**
* *Data processing holds exactly.* We took val sentence pairs that differ in only one of subject, verb or
  object and share the same circuit shape. Across them, the trace distance after tracing never exceeded the
  distance between the full 7-qubit states (0 of ~250k pairs).
* *Post-selection amplifies.* In up to 30% of pairs (typically ~15%, depending on the model and the word)
  the post-selected states are *further apart* than the full states. That is impossible for any CPTP map.
* *Success probability decays as 2^-m.* At random angles, median p_0 is 0.50 (m=1), 0.25 (m=2), 0.065
  (m=4), 0.013 (m=6) and 0.0085 (m=7), i.e. about 2^-m. In the trained models it is 0.03-0.14 for m=4
  sentences (and lower with more layers), so emulated or hardware post-selection needs about 7-40x more shots.
* *The trace throws away most of what the circuit separates.* Mean distance between sentences that differ in
  one word, 2 text layers, discard-trained:

  | word that differs | full state | after partial trace | post-selected branch |
  |---|---|---|---|
  | subject | 0.61 | 0.36 | 0.48 |
  | verb | 0.79 | 0.48 | 0.59 |
  | object | 0.83 | 0.61 | 0.77 |

  The circuit does tell the sentences apart. A large part of the difference sits in correlations with the
  discarded wires, where the partial trace cannot read it.

## 3. Each model only works with its own readout (readout swap)

Same weights, read out both ways (val accuracy, mean ± sd over seeds):

| trained as | text layers | read by post-selection | read by partial trace |
|---|---|---|---|
| non-discard | 1 | **0.745** ± 0.004 | 0.687 ± 0.010 |
| non-discard | 2 | **0.849** ± 0.007 | 0.732 ± 0.037 |
| non-discard | 3 | **0.887** ± 0.003 | 0.719 ± 0.005 |
| discard | 1 | 0.728 ± 0.017 | **0.736** ± 0.011 |
| discard | 2 | 0.587 ± 0.057 | **0.802** ± 0.014 |
| discard | 3 | 0.603 ± 0.030 | **0.827** ± 0.008 |

(Bold = the readout the model was trained with; these reproduce the logged validation scores exactly.)

* Non-discard training concentrates the useful information in branch 0 and lets the other branches be
  arbitrary. Tracing those back in costs 12-17 points.
* Discard training spreads information over all branches; branch 0 alone is near chance on subject pairs
  (0.55). The post-selected branch is only about 8% of the mixture (median p_0 = 0.077).
* **At 1 text layer the two are equivalent.** The discard model learns almost pure sentence states (purity
  0.97), so every branch looks the same and both readouts agree. Test accuracy in stage 1 matches as well:
  0.711 for both methods.
* The gap only opens with depth:
  * deeper circuits entangle the sentence wires with the other wires;
  * discard purity falls 0.97 -> 0.39 -> 0.21 (L1 -> L2 -> L3);
  * post-selection turns that extra entanglement into extra expressive power, while the trace averages it away.

## 4. Training dynamics

* **Smaller, more compressed scores under the trace.**
  * Spread of the image-text scores on val: sd 0.117 (discard L2) vs 0.277 (non-discard L2).
  * Median margin (positive minus negative score): 0.077 vs 0.304.
  * Discard margins *shrink* with depth (0.109 -> 0.077 -> 0.053), while non-discard margins grow
    (0.219 -> 0.304 -> 0.383).
  * With a fixed temperature, this means flatter logits and weaker contrast per step. Stage 1 also shows
    that rescaling alone doesn't fix it: discard temperature 0.03 left val unchanged (0.802). The
    compression is lost information, not just scale.
* **Gradients (`grad_probe`, untrained circuits, cost = overlap with a random target).** The partial-trace
  gradients are 3-35x smaller than the post-selected ones, and the ratio grows with m and with depth:

  | 3 layers, uniform random angles | m=2 | m=4 | m=7 |
  |---|---|---|---|
  | mean squared gradient, trace | 3.7e-4 | 1.8e-4 | 1.6e-4 |
  | mean squared gradient, post-selection | 1.6e-3 | 2.0e-3 | 5.6e-3 |

  The trace cost is the p_k-weighted average over branches, so branch gradients partly cancel. The
  renormalised post-selected cost keeps O(1) gradients in simulation. The price is paid in shots on hardware
  (1/p_0 ~ 2^m), which is the post-selection version of a barren plateau: exponentially rare samples rather
  than exponentially small gradients. At 7 qubits neither effect is dramatic yet, but both trends are visible.
* **The losses differ in more than the readout.**
  * FS (non-discard) and HS include a regulariser that pushes sentence states apart (lambda_reg 0.1).
  * Pauli (discard) instead has a purity term.
  * HS is the best discard loss on test (0.787), and purity weight 0 improved discard val by +2.3 in
    stage 1.
  * Part of the gap is therefore the objective, not the physics. Adding the spreading term to the Pauli
    loss is a cheap test.

## 5. Generalisation

* Contractive (Lipschitz <= 1) maps are usually *good* for generalisation, but here the margins shrink just
  as much, so the margin-normalised picture is worse:
  * val pairs within 0.02 of a tie: 12.6% (discard L2) and 20.4% (L3) vs 3.6% and 2.8% for non-discard;
  * train-val drop: discard 0.935 -> 0.802 (13.3 pts) vs non-discard 0.952 -> 0.849 (10.3 pts) at L2.
* Effective dimension (participation ratio of the Bloch vectors, val):
  * Text: 15.8 (discard) vs 18.0 (non-discard) at L2, and 21.8 vs 28.7 at L3.
  * Images: discard 15.1 -> 13.8 from L2 to L3, while non-discard images grow 16.0 -> 26.1.
  * The discard image tower stops using extra dimensions at depth 3. This is consistent with v1's finding
    that discard images sit closer together: pure images fitted to increasingly mixed text.
* By negative type the bottleneck moves with depth:
  * at L2 discard loses mostly on verbs (0.779 vs 0.863);
  * at L3 the verb gap is gone (0.873 vs 0.869), but discard is far behind on subjects (0.681 vs 0.898,
    2-3 seeds).

## 6. Test of the main lead: hybrid readout and mixed images (dv1, seeds 0-4, 2 text layers)

Hybrid = post-select only the verb's `s` wire and trace out the rest (1 post-selected wire instead of 4).
Mixed images = extra image-circuit wires that are traced out.

| variant | val (selection) | test std | test swap (norm) |
|---|---|---|---|
| non-discard (baseline) | 0.847 ± 0.008 | 0.832 ± 0.009 | 0.756 ± 0.131 |
| discard (baseline) | 0.793 ± 0.013 | 0.761 ± 0.024 | 0.794 ± 0.126 |
| discard + hybrid `s` | **0.839** ± 0.022 | **0.817** ± 0.014 | 0.766 ± 0.159 |
| discard + hybrid `s` + 1 image env qubit | 0.835 ± 0.013 | **0.822** ± 0.009 | 0.766 ± 0.099 |
| discard + 1 image env qubit | 0.811 ± 0.009 | 0.792 ± 0.010 | 0.718 ± 0.052 |
| discard + 2 image env qubits (3 seeds) | 0.798 ± 0.019 | 0.766 ± 0.026 | 0.698 ± 0.179 |

* Post-selecting the single wire that carries the verb into the sentence closes most of the test gap:
  7.1 -> 1.5 points against non-discard on these 5 seeds. That matches sections 2-3: the loss comes from
  averaging over the branches of that wire.
* One mixed image qubit helps a little (+3 test points); two do not.
* At 3 text layers the hybrid adds less (discard 0.816 -> 0.825 test, 2 seeds), consistent with the verb gap
  already being closed at that depth (section 5).

## 7. What to try (status)

1. Hybrid readout: done (section 6), closes most of the gap at 2 layers. Next: post-select other single
   wire types (e.g. the preposition's subject input) to target the subject gap at 3 layers.
2. Mixed images: done (section 6); 1 env qubit +3 points, 2 no gain.
3. Pauli loss without the purity term (pw0 helped val), plus the FS/HS spreading regulariser. Not run yet;
   needs a small loss change.
4. Fewer traced wires per sentence (e.g. a smaller determiner/preposition footprint), since purity and
   accuracy fall with m.
5. If hardware is the target, report shot cost: post-selection needs ~1/p_0 (7-40x at m=4) more shots.
