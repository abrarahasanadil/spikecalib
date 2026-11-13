"""SpikeCalib: post-hoc confidence calibration for event-driven spiking neural networks.

Code for the paper "Counting Spikes Caps Confidence": the softmax over output spike counts
is capped, and this package measures what the cap does to calibration at low timestep counts
``T`` on N-MNIST and N-Caltech101. The study began by testing whether isotonic regression's
advantage over temperature scaling grows as ``T`` falls; that hypothesis failed, as
``NOTES.md`` at the repository root records.

Package-level responsibility: this module fixes the version and the public surface. It
holds no experiment logic, so that importing the package never has side effects on RNG
state -- a package import must not perturb the seeded, reproducible pipeline described in
correctness rule 4.

"""

__version__ = "1.0.0"

