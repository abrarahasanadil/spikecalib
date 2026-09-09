"""Calibrators, the two ECE estimators, and seeded tie-breaking.

Owns correctness rule 2: **both equal-width (Guo et al.) and equal-mass ECE are
computed and both are reported.** With discrete spike counts the confidence distribution
is lumpy; equal-width binning leaves bins empty and concentrates mass in a few others,
which biases the estimate optimistically at low ``T`` -- precisely the regime the
hypothesis is about. Disagreement between the two estimators is itself a finding and is
reported, never resolved by picking the more convenient one.

Owns the tie-breaking decision: **argmax ties are broken uniformly at random under the
experiment seed, never by lowest index.** The library default biases every tie toward
class 0, and that bias would grow as ``T`` falls, contaminating the exact trend under
test. The tie rate is a reported quantity, not a diagnostic -- the hypothesis predicts it
rises as ``T`` falls, and that rise is direct evidence for the discreteness mechanism, so
it belongs in the results table.

Owns the calibrator-fitting half of correctness rule 4: **fitting uses CPU RNG and
scikit-learn only**, so fitted calibrators are reproducible bit-for-bit regardless of MPS
float variation in the trained weights.

Confidence is the softmax over the integer per-class spike counts. Both calibrators
operate on that same quantity, so the comparison isolates the calibrator, not the
readout.

TODO: softmax_confidence(counts) -> confidence from integer counts in [0, T].
TODO: break_ties(counts, rng) -> argmax with uniform random tie-breaking under the
      experiment seed; return the tie rate alongside the predictions.
TODO: fit_temperature(calib_conf, calib_correct) -> single-scalar baseline
      (Sun & Bohte, AOT-SNN).
TODO: fit_isotonic_top_label(...) -> PRIMARY formulation: maps max-confidence to binary
      correctness, fitted on the calibration split, matching what ECE measures. This is
      the headline-table number.
TODO: fit_isotonic_ovr(...) -> SECONDARY formulation: one-vs-rest with renormalisation,
      reported for comparability with Guo et al.
TODO: cross-validated isotonic for the small datasets -- DVS-Gesture has ~1,342 samples
      and isotonic overfits small calibration sets.
TODO: ece_equal_width(conf, correct, n_bins) and ece_equal_mass(conf, correct, n_bins);
      both are always returned together so neither can be reported alone.
"""

# TODO: implement.
