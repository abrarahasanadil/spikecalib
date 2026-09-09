"""Experiment orchestration: the T x seed x dataset sweep and the results table.

Runs ``T`` in {2, 4, 6, 8, 10} with 3 seeds on every dataset in the study -- 15 runs per
dataset per calibration arm. The sweep is what tests the hypothesis: the isotonic
advantage over temperature scaling is predicted to track the samples-per-distinct-level
ratio ``n_cal / n_distinct``, for which ``T`` is a proxy. If the advantage is flat in
``T``, or tracks ``T`` but not the ratio when ``n_cal`` is varied at fixed ``T``, the
capacity mechanism is wrong and the study reports that.

Owns the reporting contract that makes results readable rather than merely correct.
Every row of the results table carries, alongside the calibration metrics:

- **both** ECE estimators (correctness rule 2) -- never one without the other;
- the **tie rate**, a descriptive measure of how discrete the output is (it is not
  itself the mechanism -- neither calibrator breaks ties; see NOTES.md);
- the number of **distinct confidence levels** and the resulting
  **samples-per-level ratio** ``n_cal / n_distinct``, which is the quantity the
  hypothesis names;
- the **realised isotonic block count** after monotonicity pooling, because the
  synthetic pre-test showed ``n_distinct`` is a very loose bound on the capacity
  actually used (4-22 blocks while ``n_distinct`` spanned 34-4,197);
- the **uncalibrated ECE**, so the isotonic advantage can be compared at matched
  miscalibration severity -- otherwise a trend across ``T`` cannot be separated from
  a trend in how much error there was to remove;
- the **calibration-set size**, so a reader can judge whether an isotonic win on a small
  dataset is real or is small-sample overfitting;
- the **seed**, per correctness rule 4.

Owns the multi-dataset reporting rule from ``NOTES.md``: a trend that holds on one
dataset and not another is a finding to report, not a failure to hide. Every dataset that
is run appears in the output.

Calls into ``data.assert_disjoint`` before any calibrator is fitted, so correctness
rule 1 is enforced on the live experiment path and not only in tests.

TODO: run_sweep(dataset, timesteps_list, seeds, cfg) -> tidy results frame.
TODO: write results to results/ as CSV; results/*.csv is untracked by design, so the
      committed artifact is the code that regenerates it.
"""

# TODO: implement.
