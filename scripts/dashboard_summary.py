"""The headline numbers and the prediction ledger for the dashboard, built from the artifacts.

``scripts/export_dashboard_data.py`` calls :func:`build_summary` and :func:`build_ledger`
and writes what they return into ``docs/data.js``. The page renders those fields and types
no number of its own.

**The ledger re-derives every verdict.** Predictions 6a to 15 carry verdict blocks written
by the scripts that tested them, and those are used as they stand. Predictions 1 to 5 were
recorded before verdict blocks existed; their outcomes are in ``NOTES.md``. For each of
them the registered condition is applied here to the artifact rows, and the result is
checked against the verdict ``NOTES.md`` records (``RECORDED``). If the two ever disagree
the export stops: the page must not show a verdict that the record does not.
"""

from __future__ import annotations

import csv
import io
import statistics as st

# The verdicts NOTES.md records for predictions 1 to 5 and their parts, under the headings
# named. These are the cross-check, not the source: the source is the computation below.
RECORDED = {
    "1": False,    # "Outcome, recorded 2026-09-10 after the sweep": "the prediction as stated is wrong"
    "2.1": True,   # "Outcome of prediction 2": "Prediction 1 held"
    "2.2": True,   # "Outcome of prediction 2": "Prediction 2 held"
    "2.L": False,  # "The connection to prior work: tested, and it does not hold"
    "3.1": True,   # "Outcome of prediction 3": "The stated falsification condition is not met, so the prediction stands"
    "3.2": True,   # "Part 2 of the prediction held cleanly"
    "3.3": False,  # "Part 3 was wrong"
    "4.1": False,  # "Outcome of prediction 4": "Part 1 is falsified, and that is the result"
    "4.2": True,   # "Part 2 held"
    "4.3": False,  # "Part 3 also missed"
    "5.1": True,   # "Outcome of prediction 5", 1: "prediction HELD"
    "5.2": False,  # "Outcome of prediction 5", 2: "prediction FALSIFIED"
    "5.3": True,   # "Outcome of prediction 5", 3: "prediction HELD"
    "5.4": True,   # "Outcome of prediction 5", 4: "prediction HELD"
}

# Thresholds as registered in NOTES.md, copied from the prediction text.
P3_DEFICIT_LIMIT = {6: 0.0066, 10: 0.0045}
P4_SHARE_LIMIT = 0.5
P5_GAIN_LIMIT = 0.05
P5_RESIDUAL_LIMIT = 0.08
P2_FACTOR = 2.0
P4_C10_LIMIT = 0.05
P4_C101_GAP = 0.15


def _mean(rows, key):
    return st.mean(r[key] for r in rows)


def _sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def sweep_rows(csv_text):
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    for r in rows:
        for k in ("timesteps", "seed"):
            r[k] = int(r[k])
    return rows


def build_summary(readout, conditional, csv_text, caltech):
    """The four hero tiles. Every value is a mean over the three seeds."""
    sweep = sweep_rows(csv_text)

    def ece(arm, col):
        return st.mean(float(r[col]) for r in sweep if r["arm"] == arm and r["timesteps"] == 2)

    share = {ds: _mean(_sel(conditional, dataset=ds, timesteps=2), "share_explained_theoretical")
             for ds in ("nmnist", "ncaltech101")}
    return {
        "share_explained_t2": {"c10": share["nmnist"], "c101": share["ncaltech101"]},
        "ece_t2": {
            "uncalibrated": {"ew": ece("uncalibrated", "ece_equal_width"), "em": ece("uncalibrated", "ece_equal_mass")},
            "temperature": {"ew": ece("temperature", "ece_equal_width"), "em": ece("temperature", "ece_equal_mass")},
        },
        "auroc_t2": {
            "count": _mean(_sel(readout, readout="count", timesteps=2), "auroc"),
            "accumulator": _mean(_sel(readout, readout="accumulator", timesteps=2), "auroc"),
        },
        "ncaltech_count_t2_accuracy": _mean(_sel(caltech, readout="count", timesteps=2), "test_accuracy"),
    }


def _f(x, d=4, sign=False):
    s = f"{abs(x):.{d}f}" if sign else f"{x:.{d}f}"
    return (("+" if x >= 0 else "−") + s) if sign else s


def early_predictions(readout, conditional, arch, csv_text):
    """Predictions 1 to 5 and their parts, each re-derived from its registered condition.

    Each row carries ``values``, the numbers its measurement is stated in, so that the paper's
    macros and the dashboard read the same computation.
    """
    sweep = sweep_rows(csv_text)
    runs = {(r["timesteps"], r["seed"]): r for r in sweep if r["arm"] == "isotonic-top-label"}
    ew_iso = sorted(k for k, r in runs.items() if float(r["iso_minus_temp_ci_high"]) < 0)
    ew_res = [k for k, r in runs.items() if r["iso_minus_temp_resolvable"] == "True"]
    em_res = [k for k, r in runs.items() if r["iso_minus_temp_em_resolvable"] == "True"]
    rows = []

    def add(pid, recorded, statement, measured, held, **values):
        rows.append({"id": pid, "recorded": recorded, "statement": statement, "measured": measured,
                     "held": bool(held), "values": values})

    add("1", "2026-09-10",
        "N-MNIST: temperature scaling matches or beats top-label isotonic regression at every T, with no "
        "isotonic advantage resolvable under equal-width ECE",
        f"isotonic resolvably better in {len(ew_iso)} of {len(runs)} runs under equal-width "
        f"(T = {ew_iso[0][0]}, seed {ew_iso[0][1]}); {len(em_res)} resolve under equal-mass, "
        f"{len(set(ew_res) & set(em_res))} under both" if ew_iso else "none",
        not ew_iso, runs=len(runs), ew_iso=len(ew_iso), iso_t=ew_iso[0][0] if ew_iso else None,
        iso_seed=ew_iso[0][1] if ew_iso else None, em_res=len(em_res), both=len(set(ew_res) & set(em_res)))

    def m(name, T, key):
        return _mean(_sel(readout, readout=name, timesteps=T), key)
    count, wc, mem, acc_ece2 = (m(n, 2, "ece_equal_width") for n in ("count", "weighted-count", "membrane", "accumulator"))
    add("2.1", "2026-09-11",
        "N-MNIST, T = 2: a membrane readout removes the cap; its uncalibrated ECE is not within a factor of "
        "two of count's",
        f"ECE (equal-width) {_f(mem)} against {_f(count)}, accuracy {_f(m('membrane', 2, 'test_accuracy'))} "
        f"against {_f(m('count', 2, 'test_accuracy'))}",
        mem * P2_FACTOR < count, mem=mem, count=count,
        acc_mem=m("membrane", 2, "test_accuracy"), acc_count=m("count", 2, "test_accuracy"),
        max_mem=m("membrane", 2, "max_confidence_observed"))
    add("2.2", "2026-09-11",
        "N-MNIST, T = 2: weighted count, which shares the cap, stays near count rather than membrane",
        f"ECE (equal-width) {_f(wc)} against count's {_f(count)}",
        abs(wc - count) < abs(wc - mem), wc=wc, count=count)
    mem_gaps = {T: m("membrane", T, "mean_conf_minus_acc") for T in (2, 4, 6, 8, 10)}
    mem2 = _sel(readout, readout="membrane", timesteps=2)
    add("2.L", "2026-09-11",
        "Connection stated in advance: removing the cap restores the overconfidence the literature reports",
        "membrane's mean confidence is below its accuracy at every T, by "
        f"{_f(-mem_gaps[2])} at T = 2, in {sum(r['mean_conf_minus_acc'] < 0 for r in mem2)} of {len(mem2)} seeds",
        any(v > 0 for v in mem_gaps.values()), gap2=mem_gaps[2],
        under2=sum(r["mean_conf_minus_acc"] < 0 for r in mem2), seeds=len(mem2),
        max_gap=max(mem_gaps.values()))

    deficit = {T: m("count", T, "auroc") - m("accumulator", T, "auroc") for T in (6, 10)}

    def by_seed(name, T):
        return {r["seed"]: r for r in _sel(readout, readout=name, timesteps=T)}
    wins = sum(by_seed("accumulator", T)[s]["auroc"] >= by_seed("count", T)[s]["auroc"]
               for T in (6, 10) for s in by_seed("count", T))
    add("3.1", "2026-09-11",
        "N-MNIST: removing the leak leaves an AUROC deficit against count of at most half the leaky "
        f"membrane's ({P3_DEFICIT_LIMIT[6]} at T = 6, {P3_DEFICIT_LIMIT[10]} at T = 10), and keeps "
        "membrane's T = 2 calibration",
        f"deficit {_f(deficit[6])} and {_f(deficit[10])}; T = 2 ECE {_f(acc_ece2)}. Its headline, "
        f"match or beat count, was not met: the accumulator wins {wins} of 6 seeds",
        all(deficit[T] <= P3_DEFICIT_LIMIT[T] for T in (6, 10)) and acc_ece2 <= mem,
        d6=deficit[6], d10=deficit[10], ece=acc_ece2, wins=wins)
    beats = sum(by_seed("accumulator", T)[s]["auroc"] > by_seed("membrane", T)[s]["auroc"]
                for T in (6, 10) for s in by_seed("membrane", T))
    add("3.2", "2026-09-11", "N-MNIST: the accumulator ranks better than the leaky membrane at T = 6 and 10",
        f"higher AUROC in {beats} of 6 seeds", beats == 6, beats=beats)
    au_acc2, au_mem2 = m("accumulator", 2, "auroc"), m("membrane", 2, "auroc")
    add("3.3", "2026-09-11", "N-MNIST, T = 2: accumulator and membrane are close",
        f"close on ranking (AUROC {_f(au_acc2)} against {_f(au_mem2)}) but not on calibration "
        f"(ECE {_f(acc_ece2)} against {_f(mem)})",
        mem <= P2_FACTOR * acc_ece2, au_acc=au_acc2, au_mem=au_mem2, ece_acc=acc_ece2, ece_mem=mem)

    c101, c10 = _sel(conditional, dataset="ncaltech101", timesteps=6), _sel(conditional, dataset="nmnist", timesteps=6)
    share = _mean(c101, "share_explained_conditional")
    add("4.1", "2026-09-11",
        "N-Caltech101, T = 6: a ceiling conditioned on measured competitor activity explains more than "
        "half of the underconfidence",
        f"explains {100 * share:.0f}%: the conditional ceiling, {_f(_mean(c101, 'conditional_ceiling_mean'))}, "
        f"sits above accuracy, {_f(_mean(c101, 'test_accuracy'))}",
        share >= P4_SHARE_LIMIT, share=share, cond=_mean(c101, "conditional_ceiling_mean"),
        acc=_mean(c101, "test_accuracy"))
    gap10 = _mean(c10, "ceiling_gap_theo_minus_cond")
    add("4.2", "2026-09-11",
        f"N-MNIST, T = 6: the conditional and theoretical ceilings lie within {P4_C10_LIMIT} of each other",
        f"they differ by {_f(gap10)}", gap10 < P4_C10_LIMIT, gap=gap10)
    gap101 = _mean(c101, "ceiling_gap_theo_minus_cond")
    add("4.3", "2026-09-11",
        f"N-Caltech101, T = 6: the two ceilings differ by more than {P4_C101_GAP}",
        f"they differ by {_f(gap101)}", gap101 > P4_C101_GAP, gap=gap101)

    count_arch = _sel(arch, readout="count")
    worst = max(r["ceiling_signed_excess"] for r in count_arch)
    add("5.1", "2026-09-11",
        "Matched architectures, both datasets: no count readout exceeds its derived ceiling",
        f"worst excess {worst:.1e} over {len(count_arch)} runs", worst <= 1e-9, worst=worst, runs=len(count_arch))
    gains = {}
    for b in (2, 4):
        cnt = _sel(arch, dataset="ncaltech101", n_blocks=b, readout="count", timesteps=2)
        acc = _sel(arch, dataset="ncaltech101", n_blocks=b, readout="accumulator", timesteps=2)
        gains[b] = _mean(acc, "test_accuracy") - _mean(cnt, "test_accuracy")
    add("5.2", "2026-09-11",
        f"N-Caltech101, T = 2: removing the cap gains more than {P5_GAIN_LIMIT} accuracy at both depths",
        f"gain {_f(gains[4], sign=True)} with four blocks, {_f(gains[2], sign=True)} with two",
        all(g > P5_GAIN_LIMIT for g in gains.values()), gain4=gains[4], gain2=gains[2])
    res = {b: _mean(_sel(arch, dataset="ncaltech101", n_blocks=b, readout="count", timesteps=6), "residual")
           for b in (2, 4)}
    add("5.3", "2026-09-11",
        f"N-Caltech101, T = 6: the underconfidence stays above {P5_RESIDUAL_LIMIT} at both depths",
        f"{_f(res[4])} with four blocks, {_f(res[2])} with two",
        all(v > P5_RESIDUAL_LIMIT for v in res.values()), res4=res[4], res2=res[2])
    shares = {b: _mean(_sel(arch, dataset="ncaltech101", n_blocks=b, readout="count", timesteps=6),
                       "share_explained_conditional") for b in (2, 4)}
    add("5.4", "2026-09-11",
        "N-Caltech101, T = 6: the conditional ceiling explains none of it at either depth",
        f"{100 * shares[4]:.0f}% with four blocks, {100 * shares[2]:.0f}% with two",
        all(v == 0 for v in shares.values()), share4=shares[4], share2=shares[2])

    if [r["id"] for r in rows] != list(RECORDED):
        raise SystemExit("the earlier predictions and RECORDED list different parts")
    for p in rows:
        if p["held"] != RECORDED[p["id"]]:
            raise SystemExit(f"prediction {p['id']}: recomputed verdict {p['held']} disagrees with NOTES.md")
        p["group"] = "early"
    return rows


def paper_predictions(scaled, resolution, caltech_cal, cal_size, fusion):
    """Predictions 6a to 15, from the verdict blocks their scripts wrote."""
    P = scaled["predictions"]
    p9, p10, p11 = resolution["prediction_9"], caltech_cal["prediction_10"], cal_size["prediction_11"]
    below = sum(c["count_below_accumulator"] for c in p9["comparisons"])
    at100 = [x for x in cal_size["summary"] if x["n_cal"] == 100]
    F = fusion["predictions"]
    lows = ", ".join(_f(ci[0], sign=True) for ci in F["12"]["delta_ci"])
    means = F["14"]["mean_delta_by_t"]
    rows = [
        ("6a", "2026-10-13", "N-MNIST, T = 2: learned-scale count has a gap below 0.05 in size",
         f"gap {_f(P['6a']['mean_signed_gap'], sign=True)}", P["6a"]["held"]),
        ("6b", "2026-10-13", "N-Caltech101, T = 2: learned-scale count gap is above zero",
         f"gap {_f(P['6b']['mean_signed_gap'], sign=True)}", P["6b"]["held"]),
        ("7", "2026-10-13", f"N-Caltech101, T = 2: learned-scale accuracy reaches {P['7']['threshold_registered']:.4f}, "
                            "half the gap to the accumulator",
         f"{P['7']['mean_accuracy']:.4f}, {100 * P['7']['share_of_gap_recovered']:.0f}% of the gap", P["7"]["held"]),
        ("8", "2026-10-13", "N-Caltech101, T = 6: learned-scale gap is above −0.05",
         f"gap {_f(P['8']['mean_signed_gap'], sign=True)}, overshooting into overconfidence", P["8"]["held"]),
        ("9", "2026-10-13", f"N-MNIST, T = 2: count resolution below the accumulator's in all {len(p9['comparisons'])} comparisons",
         f"below in {below}", p9["held"]),
        ("10", "2026-10-13", f"N-Caltech101: calibrators tie in at least 7 of {p10['runs']} count runs",
         f"tie in {p10['ties']}", p10["held"]),
        ("11", "2026-10-13", "N-MNIST, T = 2, 100 calibration samples: temperature better in at least 10 of "
                             f"{cal_size['draws']} draws for two seeds; ties at 1,000 and 5,000",
         f"at most {max(x['temperature_wins'] for x in at100)} of {cal_size['draws']}", p11["held"]),
        ("12", "2026-10-14", "Fusion pilot, count, T = 2: δ above zero with its interval excluding zero in every seed",
         f"lower ends {lows}", F["12"]["held"]),
        ("13", "2026-10-14", "Fusion pilot, count, T = 2: ½P(S)(a₁ − a₂) inside the on-S interval in every seed",
         f"inside in {sum(lo <= p <= hi for (lo, hi), p in zip(F['13']['delta_on_s_ci'], F['13']['predicted']))} "
         f"of {len(F['13']['predicted'])}", 
         F["13"]["held"]),
        ("14", "2026-10-14", "Fusion pilot, count: mean δ falls strictly over T = 2, 4, 6, lower again at 8 and 10",
         "means " + ", ".join(_f(means[k], 5, sign=True) for k in ("2", "4", "6", "8", "10")), F["14"]["held"]),
        ("15", "2026-10-14", f"Fusion pilot, accumulator: no interval on δ excludes zero in {F['15']['runs']} runs",
         f"{F['15']['intervals_excluding_zero']} excludes zero, against calibration", F["15"]["held"]),
    ]
    return [{"id": i, "recorded": d, "statement": s, "measured": m, "held": bool(h), "group": "paper"}
            for i, d, s, m, h in rows]


def count_cells(readout, caltech, arch, blocks_of):
    """Distinct count-readout configurations, each counted once: (dataset, blocks, T, seed).

    The matched-architecture grid reuses twelve models from the other two experiments, so
    the three row counts must not be added.
    """
    cells = set()
    for ds, rows in (("nmnist", readout), ("ncaltech101", caltech)):
        for r in rows:
            if r["readout"] == "count":
                cells.add((ds, blocks_of(ds, r["timesteps"], r["seed"]), r["timesteps"], r["seed"]))
    for r in arch:
        if r["readout"] == "count":
            cells.add((r["dataset"], r["n_blocks"], r["timesteps"], r["seed"]))
    return len(cells)


def build_ledger(readout, conditional, arch, csv_text, scaled, resolution, caltech_cal, cal_size, fusion):
    early = early_predictions(readout, conditional, arch, csv_text)
    paper = paper_predictions(scaled, resolution, caltech_cal, cal_size, fusion)
    tally = lambda rows: {"total": len(rows), "held": sum(r["held"] for r in rows),  # noqa: E731
                          "falsified": sum(not r["held"] for r in rows)}
    return {"rows": early + paper, "early": tally(early), "paper": tally(paper), "all": tally(early + paper)}
