"""SpikeCalib: post-hoc confidence calibration for event-driven spiking neural networks.

A multi-dataset benchmark testing whether the advantage of isotonic regression over
temperature scaling grows as the number of simulation timesteps ``T`` falls. The design
document for the study is ``NOTES.md`` at the repository root; it is authoritative, and
this package is expected to follow it.

Package-level responsibility: this module fixes the version and the public surface. It
holds no experiment logic, so that importing the package never has side effects on RNG
state -- a package import must not perturb the seeded, reproducible pipeline described in
correctness rule 4.

TODO: export the stable public API once the submodules exist.
"""

__version__ = "0.1.0"

# TODO: __all__ = ["data", "models", "train", "calibration", "sweep", "figures"]
