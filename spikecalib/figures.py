"""Figures and tables for the paper.

Owns correctness rule 3: **energy is the SynOps proxy** computed with 45 nm constants
``E_AC = 0.9 pJ`` and ``E_MAC = 4.6 pJ``, attributed to Horowitz, ISSCC 2014. ``[LIT]``
for the paper; **``[UNVERIFIED]`` for the values** -- the figure number and the exact
figures have not been checked (SOURCES.md, "To verify" item 1), so captions must not cite
a figure number. Every appearance
of an energy number -- in an axis label, a caption, a table header, or body text --
describes it as a **design-comparison proxy**. It is never described as measured power,
and no claim about a physical device is derived from it. That wording is enforced here
rather than left to whoever writes the caption later.

The headline figure is the isotonic-minus-temperature calibration gap against ``T``,
which is the plot the hypothesis lives or dies on. The mechanism panel plots that same
gap against the samples-per-distinct-level ratio ``n_cal / n_distinct``, because that
ratio -- not ``T`` -- is what the capacity argument predicts it tracks. Tie rate against
``T`` is a third, descriptive panel showing the output really is discrete; it is not the
mechanism, since neither calibrator breaks ties.

Where the two ECE estimators disagree (correctness rule 2), the figure shows both rather
than choosing one.

TODO: plot_gap_vs_timesteps(results) -> headline figure, one panel per dataset.
TODO: plot_gap_vs_samples_per_level(results) -> mechanism panel; the ratio is moved
      independently of T by varying calibration-set size.
TODO: plot_tie_rate_vs_timesteps(results) -> descriptive discreteness panel.
TODO: plot_energy_proxy(results) -> SynOps proxy, axis label and caption both stating
      "design-comparison proxy, not measured power".
TODO: headline_table(results) -> top-label isotonic primary, one-vs-rest secondary, with
      both ECEs, tie rate, calibration-set size and seed on every row.
"""

# TODO: implement.
