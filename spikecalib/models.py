"""Spiking network definitions and the SynOps instrumentation they carry.

Owns the heterogeneous-geometry hazard from ``NOTES.md``: **model sizing is
configuration, never hardcoded.** The three datasets differ in both spatial size and
class count (N-MNIST 34x34 / 10, DVS-Gesture 128x128 / 11, CIFAR10-DVS 128x128 / 10).
Input dimensions, downsampling, and classifier width all derive from the dataset
registry. A hardcoded 34x34 assumption surfaces as a shape error at best and a silently
wrong model at worst once the second dataset is added.

Also owns the counting side of correctness rule 3: the model exposes the accumulate and
multiply-accumulate counts that the SynOps energy proxy is computed from. The
Horowitz (ISSCC 2014) constants and the "design-comparison proxy, never measured power"
framing live with the reporting code in ``figures.py``.

The readout is the per-class spike count over ``T`` timesteps, which ``NOTES.md`` fixes
as the logit for both calibrators. Networks are directly trained as SNNs, not converted
from ANNs.

TODO: build_model(spatial, n_classes, timesteps, **cfg) -> directly-trained SNN whose
      forward returns integer per-class spike counts in [0, T].
TODO: synops counters (AC and MAC separately) attached as hooks, not folded into the
      forward return, so instrumentation cannot alter the computation.
"""

# TODO: implement.
