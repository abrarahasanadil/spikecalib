"""Training loop for the directly-trained spiking networks.

Owns correctness rule 4: **every experiment is seeded and reproducible, and every seed
is recorded alongside its result.** The caveat is stated rather than left for a reader to
discover: MPS floating-point operations are not bitwise deterministic, so training on
Apple silicon varies slightly at a fixed seed. That variation is the reason the study
reports over 3 seeds instead of one, and the reason splits and calibrator fitting are
kept on CPU RNG and scikit-learn where exact reproducibility is achievable.

This module trains; it does not calibrate. The trained network is fixed before any
calibrator is fitted -- the study is strictly post-hoc, which is what distinguishes it
from Bayesian treatments of SNN overconfidence (Skatchkovsky & Simeone).

TODO: set_seed(seed) -> seed Python, NumPy and torch; record the seed in the run record.
TODO: train(model, loaders, cfg) -> trained weights plus a run record carrying seed,
      dataset, timesteps, and the split sizes actually used.
TODO: predict_counts(model, loader) -> integer spike counts in [0, T] for the calib and
      test splits; these are the logits everything downstream operates on.
"""

# TODO: implement.
