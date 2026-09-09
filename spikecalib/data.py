"""Dataset loading and the train / validation / calibration / test split.

Owns correctness rule 1: **the calibration split is disjoint from both the test split
and the validation split, asserted on index sets.** The assertion is on the index sets
themselves, not inferred from the construction that produced them. A post-hoc calibrator
fitted on data that overlaps the evaluation set reports its own training error as a
result, which is the most damaging thing that can silently go wrong in this study.

Also owns correctness rule 4 as it applies to splitting: **splits use CPU RNG only.**
MPS float ops are not bitwise deterministic, so anything that must be reproducible
bit-for-bit is kept off the accelerator. Split index sets are reproducible exactly;
trained weights are not, which is why the study reports over 3 seeds.

Per ``NOTES.md``, split sizes are per-dataset configuration, never a global constant:
DVS-Gesture has only ~1,342 samples, so a fixed 500-per-class calibration split is
arithmetically impossible there. Calibration-set size is returned alongside the splits
because it is reported next to every result.

TODO: load_dataset(name, timesteps) -> event frame dataset via the SpikingJelly class
      named in the fetch-script registry.
TODO: make_splits(n, sizes, seed) -> train / val / calib / test index sets, CPU RNG only.
TODO: assert_disjoint(*index_sets) -> hard failure on any intersection; call it from
      every path that produces splits, not only from tests.
TODO: CIFAR10-DVS has no official train/test split -- define one, seed it, save it to
      disk, and document it in NOTES.md when that dataset is brought into the study.
"""

# TODO: implement.
