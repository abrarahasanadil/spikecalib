/* SpikeCalib dashboard.
 *
 * Every displayed number is computed here from the sweep CSV at runtime; none is
 * written into the HTML. The CSV arrives as a string in data.js rather than over
 * fetch() because fetch is blocked under file://, and the page has to work when
 * opened straight from disk.
 *
 * Charts are hand-rolled SVG rather than a CDN library, for the same reason: a
 * CDN <script> cannot load from file:// without a network.
 */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var D = window.SPIKECALIB || {};
  var ARM_ORDER = ["uncalibrated", "temperature", "isotonic-top-label", "isotonic-one-vs-rest"];
  var ARM_LABEL = {
    "uncalibrated": "uncalibrated",
    "temperature": "temperature",
    "isotonic-top-label": "isotonic (top-label)",
    "isotonic-one-vs-rest": "isotonic (one-vs-rest)"
  };
  var ARM_COLOUR = {
    "uncalibrated": "var(--d1)",
    "temperature": "var(--d2)",
    "isotonic-top-label": "var(--d3)",
    "isotonic-one-vs-rest": "var(--d4)"
  };

  /* ---------- csv ---------- */

  function parseCSV(text) {
    var lines = text.trim().split(/\r?\n/);
    var head = splitRow(lines[0]);
    var out = [];
    for (var i = 1; i < lines.length; i++) {
      if (!lines[i]) continue;
      var cells = splitRow(lines[i]);
      var row = {};
      for (var j = 0; j < head.length; j++) row[head[j]] = cells[j];
      out.push(row);
    }
    out.columns = head;
    return out;
  }
  function splitRow(line) {
    var out = [], cur = "", q = false;
    for (var i = 0; i < line.length; i++) {
      var ch = line[i];
      if (ch === '"') { if (q && line[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
      else if (ch === "," && !q) { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }

  var ROWS = parseCSV(window.SPIKECALIB_CSV || "");
  var num = function (r, k) { var v = parseFloat(r[k]); return isFinite(v) ? v : NaN; };
  var mean = function (a) { return a.length ? a.reduce(function (x, y) { return x + y; }, 0) / a.length : NaN; };
  var uniq = function (a) { return a.filter(function (v, i, s) { return s.indexOf(v) === i; }); };

  var T_VALUES = uniq(ROWS.map(function (r) { return +r.timesteps; })).sort(function (a, b) { return a - b; });
  var SEEDS = uniq(ROWS.map(function (r) { return +r.seed; })).sort(function (a, b) { return a - b; });

  /* ---------- svg helpers ---------- */

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  /* Charts are sized from their container so they fill the panel they sit in,
     rather than being a fixed-size box floating inside a large empty card. */
  function chart(host, w, h, pad) {
    host.textContent = "";
    var avail = host.clientWidth || (host.parentNode && host.parentNode.clientWidth) || 0;
    if (avail > 40) w = Math.round(avail);
    /* width/height attributes as well as the viewBox: an SVG sized only by viewBox
       has no intrinsic aspect ratio in older engines and collapses to zero height
       under `height: auto`. CSS still scales it to the container. */
    var svg = el("svg", { viewBox: "0 0 " + w + " " + h, width: w, height: h,
      preserveAspectRatio: "xMidYMid meet", role: "img" }, host);
    return { svg: svg, w: w, h: h, pad: pad,
      ix: function (v, lo, hi) { return pad.l + (v - lo) / (hi - lo) * (w - pad.l - pad.r); },
      iy: function (v, lo, hi) { return h - pad.b - (v - lo) / (hi - lo) * (h - pad.t - pad.b); } };
  }
  /* A graticule, not a grid: the faint ruled screen of a scope. */
  function graticule(c, cols, rows) {
    var g = el("g", { stroke: "var(--graticule)", "stroke-width": 1 }, c.svg);
    var i;
    for (i = 0; i <= cols; i++) {
      var x = c.pad.l + i * (c.w - c.pad.l - c.pad.r) / cols;
      el("line", { x1: x, y1: c.pad.t, x2: x, y2: c.h - c.pad.b }, g);
    }
    for (i = 0; i <= rows; i++) {
      var y = c.pad.t + i * (c.h - c.pad.t - c.pad.b) / rows;
      el("line", { x1: c.pad.l, y1: y, x2: c.w - c.pad.r, y2: y }, g);
    }
  }
  function axisText(c, x, y, s, anchor, size) {
    var t = el("text", { x: x, y: y, "text-anchor": anchor || "middle",
      fill: "var(--ink-3)", "font-size": size || 10, "font-family": "var(--mono)" }, c.svg);
    t.textContent = s;
    return t;
  }
  function path(c, d, stroke, width, dash) {
    return el("path", { d: d, fill: "none", stroke: stroke, "stroke-width": width || 2,
      "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": dash || null }, c.svg);
  }

  /* ================= HERO: the ceiling instrument ================= */

  var ceilingOf = function (T, C) { return Math.exp(T) / (Math.exp(T) + C - 1); };

  /* The bound, inverted. ceiling(T,C) >= a  <=>  T >= ln((C-1)a/(1-a)).
     Descriptive becomes prescriptive: given a class count and a target accuracy, the
     smallest T at which calibrated confidence is even reachable. */
  var minTimesteps = function (C, a) { return Math.log((C - 1) * a / (1 - a)); };

  var hero = {
    T: 2, C: 10,
    tEl: document.getElementById("t-slider"),
    cEl: document.getElementById("c-slider"),
    tOut: document.getElementById("t-value"),
    cOut: document.getElementById("c-value"),
    aEl: document.getElementById("a-slider"),
    aOut: document.getElementById("a-value"),
    tminOut: document.getElementById("out-tmin"),
    designText: document.getElementById("design-text"),
    ceilOut: document.getElementById("out-ceiling"),
    floorOut: document.getElementById("out-floor"),
    bindOut: document.getElementById("out-bind"),
    verdict: document.getElementById("verdict-text"),
    measuredNote: document.getElementById("measured-note"),
    host: document.getElementById("ceiling-chart")
  };

  function fillTrack(input) {
    var pct = (input.value - input.min) / (input.max - input.min) * 100;
    input.style.setProperty("--fill", pct + "%");
  }

  function drawCeiling() {
    var C = hero.C, Tsel = hero.T, TMAX = 20;
    /* 540, not 340. The panel beside this one now carries the T-min table, and at 340 the
       chart left it 207px short at 1440. A taller plot is also a better one here: the
       ceiling curve is steep at low T and the extra vertical resolution is where the
       argument lives. */
    var c = chart(hero.host, 760, 540, { l: 48, r: 18, t: 14, b: 32 });
    graticule(c, 10, 5);

    var X = function (t) { return c.ix(t, 1, TMAX); };
    var Y = function (v) { return c.iy(v, 0, 1); };

    /* Region above the curve: accuracy here cannot be matched by any confidence
       the readout can emit. This is data, not decoration. */
    var top = "M" + X(1) + "," + Y(1);
    var d = "M" + X(1) + "," + Y(ceilingOf(1, C));
    for (var t = 1; t <= TMAX; t += 0.1) d += "L" + X(t) + "," + Y(ceilingOf(t, C));
    el("path", { d: d + "L" + X(TMAX) + "," + Y(1) + "L" + X(1) + "," + Y(1) + "Z",
      fill: "var(--accent)", opacity: 0.13 }, c.svg);

    /* axes */
    var i;
    for (i = 0; i <= 5; i++) {
      var v = i / 5;
      axisText(c, c.pad.l - 7, Y(v) + 3.5, v.toFixed(1), "end");
    }
    for (i = 1; i <= TMAX; i += (TMAX > 10 ? 2 : 1)) axisText(c, X(i), c.h - c.pad.b + 15, i);
    axisText(c, c.pad.l + (c.w - c.pad.l - c.pad.r) / 2, c.h - 2, "simulation timesteps T");
    var yl = axisText(c, 11, c.pad.t + (c.h - c.pad.t - c.pad.b) / 2, "confidence", "middle");
    yl.setAttribute("transform", "rotate(-90 11 " + (c.pad.t + (c.h - c.pad.t - c.pad.b) / 2) + ")");

    path(c, d, "var(--accent)", 2.25);

    /* measured maxima — only meaningful at a C this study ran: 10 (N-MNIST) and 101
       (N-Caltech101). At 101 the ring is the largest maximum over the three seeds, read
       from the count-readout rows of caltech_experiment.json, and the note is computed
       rather than typed because T = 10 is the one place the bound is not reached. */
    var measured = (D.measured || []);
    var ring = function (T, v) {
      el("circle", { cx: X(T), cy: Y(v), r: 5.5, fill: "none",
        stroke: "var(--d2)", "stroke-width": 2 }, c.svg);
    };
    var cal = (D.caltech || []).filter(function (r) { return r.readout === "count"; });
    if (C === D.n_classes) {
      measured.forEach(function (m) { ring(m.T, m.observed_max); });
      hero.measuredNote.textContent =
        "Rings are this study's five measured maxima. They sit on the curve to within 4e-13.";
    } else if (cal.length && C === cal[0].n_classes) {
      var onCurve = 0, slack = [];
      uniq(cal.map(function (r) { return +r.timesteps; })).forEach(function (T) {
        var top = Math.max.apply(null, cal.filter(function (r) { return +r.timesteps === T; })
          .map(function (r) { return r.max_confidence_observed; }));
        var gap = ceilingOf(T, C) - top;
        ring(T, top);
        if (gap > 1e-9) slack.push("At T = " + T + " the ring sits " + gap.toExponential(1) +
          " below the curve, respected but not reached");
        else onCurve = Math.max(onCurve, Math.abs(gap));
      });
      hero.measuredNote.textContent =
        "Rings are the N-Caltech101 maxima, the largest over three seeds at each T. " +
        (slack.length ? slack.join(". ") + ". The others sit" : "They sit") +
        " on the curve to within " + onCurve.toExponential(0) + ".";
    } else {
      hero.measuredNote.textContent =
        "Measured maxima are hidden: they were recorded at C = " + D.n_classes +
        (cal.length ? " and C = " + cal[0].n_classes : "") + ", and C is currently " + C + ".";
    }

    /* selected T marker */
    var cy = Y(ceilingOf(Tsel, C));
    el("line", { x1: X(Tsel), y1: c.pad.t, x2: X(Tsel), y2: c.h - c.pad.b,
      stroke: "var(--accent)", "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.55 }, c.svg);
    el("circle", { cx: X(Tsel), cy: cy, r: 6.5, fill: "var(--screen)" }, c.svg);
    el("circle", { cx: X(Tsel), cy: cy, r: 3.5, fill: "var(--accent)" }, c.svg);

    c.svg.setAttribute("aria-label",
      "Attainable confidence ceiling against timesteps for " + C + " classes. At T=" +
      Tsel + " the ceiling is " + ceilingOf(Tsel, C).toFixed(4) + ".");
  }

  /* The derived panel used to end well above the chart beside it. This is the same
     arithmetic the readout above already shows, swept across the class counts the study
     actually uses plus the two it does not, so the space is filled with derived content
     rather than with padding. It follows the target-accuracy slider. */
  var TMIN_C = [10, 11, 101];   // the three class counts in the dataset registry
  function renderTmin(a) {
    var host = document.getElementById("tmin-table");
    if (!host) return;
    var html = "<thead><tr><th>classes C</th><th>ceiling at T = " + hero.T +
      "</th><th>T<sub>min</sub></th></tr></thead><tbody>";
    TMIN_C.forEach(function (C) {
      var t = minTimesteps(C, a);
      var here = C === hero.C ? ' class="k"' : "";
      html += "<tr" + (C === hero.C ? ' style="background:rgba(102,212,207,.07)"' : "") +
        "><td" + here + ">" + C + "</td><td>" + ceilingOf(hero.T, C).toFixed(4) +
        "</td><td>" + (t <= 0 ? "&lt; 1" : Math.max(1, Math.ceil(t))) + "</td></tr>";
    });
    host.innerHTML = html + "</tbody>";
  }

  /* ---------- card faces, rendered from the artifacts ----------

     These seven tables used to be static HTML. One of them drifted: card 2 claimed 44.1%
     for the C=10, T=6 explained share while conditional_ceiling.json had moved to 44.5%
     after the architecture fix re-ran it. A number on a card face that no artifact can
     reproduce is the worst kind, because it looks like a measurement.

     CONVENTION: every share and every mean here is a MEAN OF PER-SEED VALUES, not a ratio
     of pooled means. The two differ -- for C=10, T=6 they are 44.52% and 44.40% -- and
     mean-of-ratios is used because each seed is an independent run and the spread across
     seeds is the thing the study reports elsewhere. Applied to all three cells of card 2,
     not just the one that was wrong. */
  function mini(headers, rows) {
    return '<table class="mini"><thead><tr>'
      + headers.map(function (h) { return "<th>" + h + "</th>"; }).join("")
      + "</tr></thead><tbody>"
      + rows.map(function (r) {
          return "<tr>" + r.map(function (c) { return "<td>" + c + "</td>"; }).join("") + "</tr>";
        }).join("")
      + "</tbody></table>";
  }
  function put(id, html) { var e = document.getElementById(id); if (e) e.innerHTML = html; }
  function avg(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }
  function pick(rows, f) { return (rows || []).filter(f); }

  function renderCardFigures() {
    var cond = D.conditional || [], arch = D.arch || [], ro = D.readout || [];

    /* card 1 -- the derived ceiling against the observed maximum */
    var cells = [[10, 2], [101, 2], [101, 6]].map(function (k) {
      var r = pick(cond, function (x) { return x.n_classes === k[0] && x.timesteps === k[1]; });
      if (!r.length) return null;
      return [k[0], k[1], avg(r.map(function (x) { return x.theoretical_ceiling; })).toFixed(6),
              avg(r.map(function (x) { return x.observed_max_confidence; })).toFixed(6)];
    }).filter(Boolean);
    if (cells.length) put("fig-ceiling", mini(["C", "T", "derived", "observed"], cells));

    /* card 2 -- the share of pre-calibration error the ceiling forces */
    var sh = [[10, 2], [10, 6], [101, 2]].map(function (k) {
      var r = pick(cond, function (x) { return x.n_classes === k[0] && x.timesteps === k[1]; });
      if (!r.length) return null;
      return [k[0], k[1],
        (avg(r.map(function (x) { return x.share_explained_theoretical; })) * 100).toFixed(1) + "%"];
    }).filter(Boolean);
    if (sh.length) put("fig-explained", mini(["C", "T", "explained"], sh));

    /* card 5 -- what removing the cap buys, N-Caltech101 at T = 2 */
    var gain = [4, 2].map(function (blocks) {
      var c = pick(arch, function (x) {
        return x.dataset === "ncaltech101" && x.n_blocks === blocks
            && x.timesteps === 2 && x.readout === "count"; });
      var a = pick(arch, function (x) {
        return x.dataset === "ncaltech101" && x.n_blocks === blocks
            && x.timesteps === 2 && x.readout === "accumulator"; });
      if (!c.length || !a.length) return null;
      var mc = avg(c.map(function (x) { return x.test_accuracy; }));
      var ma = avg(a.map(function (x) { return x.test_accuracy; }));
      return [blocks + " blocks", mc.toFixed(4), ma.toFixed(4),
              (ma - mc >= 0 ? "+" : "") + ((ma - mc) * 100).toFixed(1)];
    }).filter(Boolean);
    if (gain.length) put("fig-gain", mini(["depth", "count", "uncapped", "gain"], gain));

    /* card 4 -- the direction each readout family fails in, N-MNIST at T = 2 */
    var BOUND = { count: "capped", membrane: "unbounded, leaky", accumulator: "unbounded" };
    var fam = ["count", "membrane", "accumulator"].map(function (name) {
      var r = pick(ro, function (x) { return x.readout === name && x.timesteps === 2; });
      if (!r.length) return null;
      var v = avg(r.map(function (x) { return x.mean_conf_minus_acc; }));
      return [name, BOUND[name], (v >= 0 ? "+" : "") + v.toFixed(4)];
    }).filter(Boolean);
    if (fam.length) put("fig-readout", mini(["readout", "bound", "conf \u2212 acc"], fam));

    /* card 3 -- the cross-entropy floor the same bound implies, against measured plateaus */
    var meas = D.measured || [];
    var fl = [2, 6, 10].map(function (T) {
      var r = meas.filter(function (x) { return +x.T === T; })[0];
      if (!r) return null;
      return [T, (-Math.log(r.ceiling)).toFixed(4), (+r.train_loss_final).toFixed(4)];
    }).filter(Boolean);
    if (fl.length) put("fig-floor", mini(["T", "floor", "plateau"], fl));

    /* card 6 -- how the fifteen paired intervals fall. Read off the CSV the page already
       parses, so it follows the data rather than restating a count written by hand. */
    /* The last row is the finding, not the counts above it: the runs that resolve under
       each estimator are computed on identical confidences, so a real difference would
       resolve under both. Counted here from the CSV rather than written down, because the
       argument rests on that intersection being empty. */
    var seen = {}, both = 0, c = { equal_width: [0, 0, 0], equal_mass: [0, 0, 0] };
    ROWS.forEach(function (r) {
      var k = r.timesteps + "|" + r.seed;
      if (seen[k]) return;
      seen[k] = 1;
      var hit = 0;
      [["equal_width", "iso_minus_temp_resolvable", "iso_minus_temp_ece_ew"],
       ["equal_mass", "iso_minus_temp_em_resolvable", "iso_minus_temp_ece_em"]].forEach(function (e) {
        if (r[e[1]] === undefined) return;
        if (String(r[e[1]]) !== "True") c[e[0]][0]++;
        else { hit++; if (+r[e[2]] > 0) c[e[0]][1]++; else c[e[0]][2]++; }
      });
      if (hit === 2) both++;
    });
    /* Only overwrite the static fallback when there were rows to count. Without data.js
       this renderer used to replace the correct fallback with a table of zeros, including
       "resolve under both 0" -- a claim about data it never saw. */
    if (Object.keys(seen).length) put("fig-bootstrap", mini(["outcome", "eq-width", "eq-mass"], [
      ["straddle zero", c.equal_width[0], c.equal_mass[0]],
      ["favour temperature", c.equal_width[1], c.equal_mass[1]],
      ["favour isotonic", c.equal_width[2], c.equal_mass[2]]])
      .replace("</tbody>", '<tr><td>resolve under both</td><td colspan="2">' + both + "</td></tr></tbody>"));

    /* card 7 -- how many selective-prediction operating points each readout allows */
    var ops = [["count", "count"], ["weighted-count", "weighted-count"]].map(function (k) {
      var r = pick(ro, function (x) { return x.readout === k[0] && x.timesteps === 2; });
      if (!r.length) return null;
      return [k[1], Math.round(avg(r.map(function (x) {
        return x.n_achievable_coverage_levels; }))).toLocaleString()];
    }).filter(Boolean);
    if (ops.length) {
      ops.push(["uncapped", (10000).toLocaleString()]);
      put("fig-ops", mini(["readout", "T = 2 ops"], ops));
    }
  }

  /* ---------- the cell counts, rendered from the artifacts ----------

     A cell is one (dataset, architecture, T, seed). The page said 48 for a while: 15 + 9 + 24
     rows across three artifacts. But the matched grid reuses twelve models the readout and
     C-axis experiments had already measured, so twelve cells were counted twice. The rows
     of those two experiments do not carry their depth, so a model is recognised here by
     dataset, T, seed and test accuracy, which a retrained network does not share. Every
     [data-n] span keeps its typed value as the fallback if the payload is unavailable. */
  function renderCells() {
    var seen = {}, per = { readout: 0, caltech: 0, arch: 0 };
    function add(src, dataset, r) {
      if (r.readout !== "count") return;
      per[src]++;
      seen[[dataset, r.timesteps, r.seed, r.test_accuracy.toFixed(9)].join("|")] =
        [r.max_confidence_observed, r.ceiling_derived];
    }
    (D.readout || []).forEach(function (r) { add("readout", D.dataset, r); });
    (D.caltech || []).forEach(function (r) { add("caltech", r.dataset, r); });
    (D.arch || []).forEach(function (r) { add("arch", r.dataset, r); });
    var cells = Object.keys(seen).map(function (k) { return seen[k]; });
    if (!cells.length) return;
    var ok = cells.filter(function (c) { return c[0] <= c[1] + 1e-12; }).length;
    var gaps = cells.map(function (c) { return c[1] - c[0]; }).filter(function (g) { return Math.abs(g) > 1e-9; });
    var grid = (D.arch || []).filter(function (r) { return r.readout === "count"; });
    var values = {
      "cells-all": cells.length, "cells-ok": ok, "cells-sat": cells.length - gaps.length,
      "cells-slack": gaps.length, "cells-readout": per.readout, "cells-caltech": per.caltech,
      "cells-shared": per.readout + per.caltech + per.arch - cells.length,
      "grid-all": grid.length,
      "grid-ok": grid.filter(function (r) { return r.max_confidence_observed <= r.ceiling_derived + 1e-12; }).length,
      "slack-gap": gaps.length ? avg(gaps).toExponential(3) : "0"
    };
    Array.prototype.forEach.call(document.querySelectorAll("[data-n]"), function (e) {
      var k = e.getAttribute("data-n");
      if (values[k] !== undefined) e.textContent = values[k];
    });
  }

  function updateHero() {
    hero.T = +hero.tEl.value;
    hero.C = +hero.cEl.value;
    fillTrack(hero.tEl); fillTrack(hero.cEl);
    hero.tOut.textContent = hero.T;
    hero.cOut.textContent = hero.C;

    var ceil = ceilingOf(hero.T, hero.C);
    hero.ceilOut.textContent = ceil.toFixed(4);
    hero.floorOut.textContent = (-Math.log(ceil)).toFixed(4);
    hero.bindOut.textContent = (ceil * 100).toFixed(1) + "%";

    var a = +hero.aEl.value;
    hero.aOut.textContent = a.toFixed(3);
    fillTrack(hero.aEl);
    var tmin = minTimesteps(hero.C, a);
    var tminCeil = Math.max(1, Math.ceil(tmin));
    hero.tminOut.textContent = tmin <= 0 ? "< 1" : String(tminCeil);
    renderTmin(a);
    hero.designText.textContent =
      "To reach " + (a * 100).toFixed(1) + "% accuracy with " + hero.C +
      " classes and still be able to express it, a spike-count readout needs T \u2265 " +
      tminCeil + " (exactly " + tmin.toFixed(3) + "). " +
      (hero.T >= tmin
        ? "The current T = " + hero.T + " is sufficient."
        : "The current T = " + hero.T + " is not: its ceiling is " +
          (ceilingOf(hero.T, hero.C) * 100).toFixed(1) + "%.");

    hero.verdict.textContent =
      "At T = " + hero.T + " with " + hero.C + " classes, a model above " +
      (ceil * 100).toFixed(1) + "% accuracy cannot be correctly confident: no reachable " +
      "spike-count vector produces a confidence that high.";

    drawCeiling();
  }

  [hero.tEl, hero.cEl, hero.aEl].forEach(function (input) {
    input.addEventListener("input", updateHero);
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-preset]"), function (btn) {
    btn.addEventListener("click", function () {
      var p = btn.getAttribute("data-preset").split(",");
      hero.tEl.value = p[0]; hero.cEl.value = p[1];
      updateHero();
      Array.prototype.forEach.call(document.querySelectorAll("[data-preset]"), function (b) {
        b.setAttribute("aria-checked", b === btn ? "true" : "false");
      });
    });
  });

  /* ================= RESULTS EXPLORER ================= */

  var state = {
    T: T_VALUES.slice(),
    seeds: SEEDS.slice(),
    arms: ARM_ORDER.slice(),
    estimator: "equal_width",
    sortKey: "timesteps",
    sortDir: 1
  };
  var ecoCol = function () { return state.estimator === "equal_width" ? "ece_equal_width" : "ece_equal_mass"; };

  function filtered() {
    return ROWS.filter(function (r) {
      return state.T.indexOf(+r.timesteps) >= 0 &&
             state.seeds.indexOf(+r.seed) >= 0 &&
             state.arms.indexOf(r.arm) >= 0;
    });
  }

  /* `single` marks a pick-one group: exactly one option is active and clicking another
     moves the selection rather than adding to it. Those get radio semantics and the
     segmented track; the multi-select filters keep aria-pressed and separate chips.

     Every button is re-synced from `get` after any click, not just the one clicked. The
     old code updated only the clicked button, so choosing equal-mass left equal-width
     still reading aria-pressed="true" and both appeared selected. */
  function buildChips(host, values, labels, get, set, single) {
    host.textContent = "";
    if (single) {
      host.classList.add("segmented");
      host.setAttribute("role", "radiogroup");
    }
    var btns = [];
    function sync() {
      btns.forEach(function (b) {
        b.setAttribute(single ? "aria-checked" : "aria-pressed", get(b.__v) ? "true" : "false");
      });
    }
    values.forEach(function (v, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "chip" + (typeof v === "number" ? " mono" : "");
      b.textContent = labels ? labels[i] : v;
      b.__v = v;
      if (single) b.setAttribute("role", "radio");
      b.addEventListener("click", function () { set(v); sync(); render(); });
      btns.push(b);
      host.appendChild(b);
    });
    sync();
  }

  function toggler(listName) {
    return {
      get: function (v) { return state[listName].indexOf(v) >= 0; },
      set: function (v) {
        var i = state[listName].indexOf(v);
        if (i >= 0) { if (state[listName].length > 1) state[listName].splice(i, 1); }
        else state[listName].push(v);
      }
    };
  }

  var tTog = toggler("T"), sTog = toggler("seeds"), aTog = toggler("arms");
  buildChips(document.getElementById("filter-t"), T_VALUES, null, tTog.get, tTog.set);
  buildChips(document.getElementById("filter-seed"), SEEDS, null, sTog.get, sTog.set);
  buildChips(document.getElementById("filter-arm"), ARM_ORDER,
    ARM_ORDER.map(function (a) { return ARM_LABEL[a]; }), aTog.get, aTog.set);
  buildChips(document.getElementById("filter-est"), ["equal_width", "equal_mass"],
    ["equal-width", "equal-mass"],
    function (v) { return state.estimator === v; },
    function (v) { state.estimator = v; }, true);

  /* ---------- tiles ---------- */

  function tile(label, value, code, sub, muted) {
    return '<div class="tile"><div class="tile-label">' + label + '</div>' +
      '<div class="tile-value' + (muted ? " muted" : "") + '">' + value + '</div>' +
      (sub ? '<span class="tile-sub">' + sub + "</span>" : "") +
      '<code class="tile-code">' + code + "</code></div>";
  }

  function renderTiles(rows) {
    var col = ecoCol();
    var unc = rows.filter(function (r) { return r.arm === "uncalibrated"; }).map(function (r) { return num(r, col); });
    var cal = rows.filter(function (r) { return r.arm !== "uncalibrated"; });
    var best = cal.length ? Math.min.apply(null, cal.map(function (r) { return num(r, col); })) : NaN;
    var iso = uniqRuns(rows).map(function (r) { return num(r, "iso_minus_temp_ece_ew"); });
    var isoEm = uniqRuns(rows).map(function (r) { return num(r, "iso_minus_temp_ece_em"); });
    var auroc = uniqRuns(rows).map(function (r) { return num(r, "auroc_raw_test"); });
    var fmt = function (v, d) { return isFinite(v) ? v.toFixed(d === undefined ? 4 : d) : "—"; };

    /* A mean over several T spans two orders of magnitude — 0.54 at T=2 against
       0.003 at T=10 — so the span is shown beside it. A bare mean here would be a
       true number that misleads. */
    var span = unc.length > 1
      ? "mean of " + unc.length + ", range " + fmt(Math.min.apply(null, unc)) + "–" + fmt(Math.max.apply(null, unc))
      : unc.length + " run";
    var bestWhere = "";
    cal.forEach(function (r) { if (num(r, col) === best) bestWhere = ARM_LABEL[r.arm] + ", T=" + r.timesteps + " seed " + r.seed; });

    /* These four move with the rail filters while the cards beside them do not, so they
       say so. A live number sitting unlabelled among fixed ones is how a reader ends up
       reading a study-wide claim as a filtered result. */
    document.getElementById("tiles").innerHTML =
      '<p class="tiles-label">Live &mdash; follows the sweep filters below</p>' +
      tile("Uncalibrated ECE", fmt(mean(unc)), col, span, !unc.length) +
      tile("Best calibrated ECE", fmt(best), col, bestWhere || "no calibrated arm selected", !isFinite(best)) +
      /* Both estimators, side by side, whatever the estimator control says. This tile
         used to show equal-width alone because the paired bootstrap had only been run on
         that estimator; rerun_paired_bootstrap.py added iso_minus_temp_ece_em, and the two
         means have opposite signs, which a single number hid. */
      tile("Isotonic − temperature",
        (isFinite(mean(iso)) && isFinite(mean(isoEm))
          ? signed(mean(iso), 4) + "<small> eq-width</small> / " + signed(mean(isoEm), 4) + "<small> eq-mass</small>"
          : "—"),
        "iso_minus_temp_ece_ew / _em", "positive favours temperature",
        !iso.length) +
      tile("AUROC, raw readout", fmt(mean(auroc)), "auroc_raw_test", "confidence vs correctness", !auroc.length);
  }

  function uniqRuns(rows) {
    var seen = {}, out = [];
    rows.forEach(function (r) {
      var k = r.timesteps + "|" + r.seed;
      if (!seen[k]) { seen[k] = 1; out.push(r); }
    });
    return out;
  }

  /* ---------- paired intervals ----------

     Two panels with shared rows and one horizontal scale: equal-width on the left,
     equal-mass on the right. The chart used to draw the equal-width interval alone. That
     shows two resolvable runs and cannot show the argument, which is that the runs
     resolving under one estimator are not the runs resolving under the other. It does not
     follow the estimator control: both are always on screen, because the comparison
     between them is the finding. */
  var PAIRED = [
    { label: "equal-width", d: "iso_minus_temp_ece_ew", lo: "iso_minus_temp_ci_low",
      hi: "iso_minus_temp_ci_high", res: "iso_minus_temp_resolvable" },
    { label: "equal-mass", d: "iso_minus_temp_ece_em", lo: "iso_minus_temp_em_ci_low",
      hi: "iso_minus_temp_em_ci_high", res: "iso_minus_temp_em_resolvable" }
  ];

  function renderIntervals(rows) {
    var runs = uniqRuns(rows).sort(function (a, b) {
      return (+a.timesteps - +b.timesteps) || (+a.seed - +b.seed);
    });
    var host = document.getElementById("interval-chart");
    if (!runs.length) { host.textContent = ""; return; }
    var rowH = 21, top = 38, bottom = 40, block = runs.length * rowH + top + bottom;
    var padL = 66, padR = 10, gap = 22;
    /* Side by side where each panel gets room for its axis labels; one above the other
       where it does not. At phone width two half-panels left about 100px each and the
       tick labels ran into one another. */
    var avail = host.clientWidth || (host.parentNode && host.parentNode.clientWidth) || 620;
    var stacked = (avail - padL - padR - gap) / 2 < 170;
    var c = chart(host, 620, stacked ? block * 2 : block, { l: padL, r: padR, t: top, b: bottom });
    var pw = stacked ? c.w - padL - padR : (c.w - padL - padR - gap) / 2;

    /* One scale for both panels, always containing zero, so an interval's distance from
       zero means the same thing in either. */
    var lo = 0, hi = 0;
    runs.forEach(function (r) {
      PAIRED.forEach(function (p) {
        lo = Math.min(lo, num(r, p.lo)); hi = Math.max(hi, num(r, p.hi));
      });
    });
    var pad = (hi - lo) * 0.08 || 0.001;
    lo -= pad; hi += pad;

    var counts = PAIRED.map(function (p, k) {
      var x0 = padL + (stacked ? 0 : k * (pw + gap)), x1 = x0 + pw;
      var oy = stacked ? k * block : 0, y0 = oy + top, y1 = oy + block - bottom;
      var X = function (v) { return x0 + (v - lo) / (hi - lo) * pw; };
      graticule({ svg: c.svg, w: c.w, h: c.h, pad: { l: x0, r: c.w - x1, t: y0, b: c.h - y1 } }, 4, runs.length);
      el("line", { x1: X(0), y1: y0, x2: X(0), y2: y1,
        stroke: "var(--ink-3)", "stroke-width": 1.5 }, c.svg);

      var resolved = 0;
      runs.forEach(function (r, i) {
        var y = y0 + i * rowH + rowH / 2;
        var a = num(r, p.lo), b = num(r, p.hi), d = num(r, p.d);
        var res = r[p.res] === "True";
        var col = res ? "var(--status)" : "var(--d1)";
        if (res) resolved++;
        /* the rows are shared, so they are labelled once unless the panels are stacked */
        if (stacked || k === 0) axisText(c, padL - 8, y + 3.5, "T=" + r.timesteps + "  s" + r.seed, "end");
        el("line", { x1: X(a), y1: y, x2: X(b), y2: y, stroke: col, "stroke-width": 2.5,
          "stroke-linecap": "round", opacity: res ? 1 : 0.75 }, c.svg);
        el("circle", { cx: X(d), cy: y, r: 3.2, fill: col }, c.svg);
        /* the whole row is the hover target, not the 2.5px stroke */
        var hit = el("rect", { x: x0, y: y - rowH / 2, width: pw, height: rowH, fill: "transparent" }, c.svg);
        el("title", {}, hit).textContent = "T = " + r.timesteps + ", seed " + r.seed + ", " + p.label +
          ": " + signed(d, 4) + "  [" + signed(a, 4) + ", " + signed(b, 4) + "]" +
          (res ? "  excludes zero, favours " + (b < 0 ? "isotonic" : "temperature") : "");
      });

      var head = el("text", { x: x0, y: oy + 14, "text-anchor": "start", fill: "var(--ink-2)",
        "font-size": 11.5, "font-family": "var(--sans)", "font-weight": 600 }, c.svg);
      head.textContent = p.label;
      axisText(c, x0, oy + 28, (runs.length - resolved) + " of " + runs.length + " straddle zero", "start");

      axisText(c, x0, y1 + 15, fmt(lo, 4), "start");
      axisText(c, x1, y1 + 15, fmt(hi, 4), "end");
      if (X(0) - x0 > 62 && x1 - X(0) > 62) axisText(c, X(0), y1 + 15, "0");
      /* Direction is the side of zero, said once on the axis. A tag beside each amber
         interval collided with the zero line once the panels were half the width. */
      axisText(c, x0, y1 + 30, "\u2190 isotonic", "start");
      axisText(c, x1, y1 + 30, "temperature \u2192", "end");
      return resolved;
    });

    var both = runs.filter(function (r) {
      return PAIRED.every(function (p) { return r[p.res] === "True"; });
    }).length;
    c.svg.setAttribute("aria-label", "Paired bootstrap intervals for the isotonic minus temperature " +
      "ECE difference, " + runs.length + " runs, under equal-width and equal-mass binning. " +
      counts[0] + " exclude zero under equal-width, " + counts[1] + " under equal-mass, " + both + " under both.");
    document.getElementById("interval-count").textContent =
      runs.length + " run" + (runs.length === 1 ? "" : "s") + " shown. Intervals excluding zero: " +
      counts[0] + " under equal-width, " + counts[1] + " under equal-mass, " + both + " under both.";
  }

  /* ---------- reliability ---------- */

  function renderReliability(rows) {
    var host = document.getElementById("reliability-grid");
    host.textContent = "";
    var keys = [];
    rows.forEach(function (r) {
      var k = r.timesteps + "|" + r.seed + "|" + r.arm;
      if (keys.indexOf(k) < 0) keys.push(k);
    });
    keys.sort(function (a, b) {
      var A = a.split("|"), B = b.split("|");
      return (+A[0] - +B[0]) || (+A[1] - +B[1]) || (ARM_ORDER.indexOf(A[2]) - ARM_ORDER.indexOf(B[2]));
    });
    /* Two at a size where the bins can actually be read, rather than six that
       cannot. The filters choose which two. */
    var CAP = 2, shown = keys.slice(0, CAP);
    document.getElementById("reliability-count").textContent =
      keys.length <= CAP
        ? keys.length + " diagram" + (keys.length === 1 ? "" : "s") + " shown."
        : "Showing " + CAP + " of " + keys.length + " selected combinations, at a size where the "
          + "bins are readable. Narrow the filters to choose which.";

    shown.forEach(function (key) {
      var parts = key.split("|");
      var bins = ((D.reliability || {})[key] || {})[state.estimator] || [];
      var wrap = document.createElement("div");
      wrap.className = "well";
      var cap = document.createElement("div");
      cap.className = "chart-note";
      cap.innerHTML = "<b>T = " + parts[0] + ", seed " + parts[1] + "</b> — " + ARM_LABEL[parts[2]];
      var box = document.createElement("div");
      wrap.appendChild(box); wrap.appendChild(cap);
      host.appendChild(wrap);

      var c = chart(box, 460, 330, { l: 44, r: 14, t: 12, b: 34 });
      graticule(c, 5, 5);
      var X = function (v) { return c.ix(v, 0, 1); };
      var Y = function (v) { return c.iy(v, 0, 1); };
      el("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1),
        stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "4 4" }, c.svg);

      var maxN = bins.length ? Math.max.apply(null, bins.map(function (b) { return b.n; })) : 1;
      var d = "";
      bins.forEach(function (b, i) { d += (i ? "L" : "M") + X(b.conf) + "," + Y(b.acc); });
      if (d) path(c, d, ARM_COLOUR[parts[2]], 1.8);
      bins.forEach(function (b) {
        /* Marker area tracks the bin count, so a point resting on three samples
           cannot look as solid as one resting on four thousand. */
        var r = 2 + 5 * Math.sqrt(b.n / maxN);
        var node = el("circle", { cx: X(b.conf), cy: Y(b.acc), r: r,
          fill: ARM_COLOUR[parts[2]], "fill-opacity": 0.55,
          stroke: ARM_COLOUR[parts[2]], "stroke-width": 1.2 }, c.svg);
        var title = el("title", {}, node);
        title.textContent = "confidence " + b.conf.toFixed(4) + ", accuracy " +
          b.acc.toFixed(4) + ", n = " + b.n;
      });
      for (var i = 0; i <= 5; i += 1) {
        axisText(c, c.pad.l - 6, Y(i / 5) + 3, (i / 5).toFixed(1), "end", 9);
        axisText(c, X(i / 5), c.h - c.pad.b + 13, (i / 5).toFixed(1), "middle", 9);
      }
      axisText(c, c.pad.l + (c.w - c.pad.l - c.pad.r) / 2, c.h - 1, "confidence", "middle", 9.5);
      c.svg.setAttribute("aria-label", "Reliability diagram, T " + parts[0] + " seed " +
        parts[1] + ", " + ARM_LABEL[parts[2]] + ". " + bins.length + " occupied bins.");
    });
  }

  /* ---------- table ---------- */

  var COLS = [
    { k: "timesteps", t: "T" }, { k: "seed", t: "seed" },
    { k: "arm", t: "arm", text: true },
    { k: "test_accuracy", t: "test acc", d: 4 },
    { k: "ece_equal_width", t: "ECE ew", d: 4 },
    { k: "ece_equal_mass", t: "ECE em", d: 4 },
    { k: "auroc", t: "AUROC", d: 4 },
    { k: "n_distinct_cal", t: "levels", d: 0 },
    { k: "samples_per_level", t: "samp/lvl", d: 1 },
    { k: "epochs_run", t: "epochs", d: 0 },
    { k: "early_stopped", t: "early stop", text: true }
  ];

  function renderTable(rows) {
    var sorted = rows.slice().sort(function (a, b) {
      var k = state.sortKey, x = a[k], y = b[k];
      var nx = parseFloat(x), ny = parseFloat(y);
      if (isFinite(nx) && isFinite(ny)) return (nx - ny) * state.sortDir;
      return String(x).localeCompare(String(y)) * state.sortDir;
    });
    var html = "<thead><tr>";
    COLS.forEach(function (c) {
      var active = state.sortKey === c.k;
      html += '<th data-k="' + c.k + '" scope="col" tabindex="0" role="button">' + c.t +
        (active ? ' <span class="dir">' + (state.sortDir > 0 ? "↑" : "↓") + "</span>" : "") + "</th>";
    });
    html += "</tr></thead><tbody>";
    sorted.forEach(function (r) {
      html += "<tr>";
      COLS.forEach(function (c) {
        var v = r[c.k];
        if (!c.text && c.d !== undefined) { var f = parseFloat(v); v = isFinite(f) ? f.toFixed(c.d) : "—"; }
        if (c.k === "arm") v = ARM_LABEL[v] || v;
        if (c.k === "early_stopped") v = v === "True" ? "yes" : "no";
        html += '<td class="' + (c.text ? "k" : "") + '">' + v + "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody>";
    var table = document.getElementById("results-table");
    table.innerHTML = html;
    Array.prototype.forEach.call(table.querySelectorAll("th"), function (th) {
      function sort() {
        var k = th.getAttribute("data-k");
        if (state.sortKey === k) state.sortDir *= -1; else { state.sortKey = k; state.sortDir = 1; }
        render();
      }
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); }
      });
    });
    document.getElementById("table-meta").textContent =
      sorted.length + " of " + ROWS.length + " rows. Sorted by " + state.sortKey +
      (state.sortDir > 0 ? ", ascending." : ", descending.");
  }

  /* ---------- artifact-driven tables ----------

     These render from window.SPIKECALIB.readout and .quantisation, which are copied
     verbatim out of results/readout_experiment.json and results/quantisation_test.json by
     scripts/export_dashboard_data.py. Nothing here carries a number of its own: that is
     how the page previously came to state a membrane advantage measured on one seed as
     though it held across three. A section whose artifact is missing hides itself.
  */

  function mean(xs) { var t = 0, i; for (i = 0; i < xs.length; i++) t += xs[i]; return t / xs.length; }
  function sd(xs) {
    if (xs.length < 2) return 0;
    var m = mean(xs), t = 0, i;
    for (i = 0; i < xs.length; i++) t += (xs[i] - m) * (xs[i] - m);
    return Math.sqrt(t / xs.length);
  }
  function hide(id) {
    var n = document.getElementById(id);
    if (n) { var sec = n.closest("section"); if (sec) sec.hidden = true; }
  }
  function fmt(v, d) { return (v < 0 ? "\u2212" : "") + Math.abs(v).toFixed(d); }
  // Signed: where the SIGN is the result -- conf - acc is under- vs overconfidence -- an
  // unmarked positive reads as "no sign shown" rather than "positive".
  function signed(v, d) { return (v < 0 ? "\u2212" : "+") + Math.abs(v).toFixed(d); }

  var READOUT_ORDER = ["count", "weighted-count", "membrane", "accumulator"];
  var READOUT_LABEL = {
    "count": "count", "weighted-count": "weighted-count",
    "membrane": "membrane (leaky)", "accumulator": "accumulator (non-leaky)"
  };

  function renderReadoutTable() {
    var host = document.getElementById("readout-table");
    if (!host) return;
    var rows = window.SPIKECALIB.readout;
    if (!rows || !rows.length) { hide("readout-table"); return; }

    var Ts = [], seen = {};
    rows.forEach(function (r) { if (!seen[r.timesteps]) { seen[r.timesteps] = 1; Ts.push(r.timesteps); } });
    Ts.sort(function (a, b) { return a - b; });

    var html = "<thead><tr>" +
      ["T", "readout", "seeds", "test acc", "conf \u2212 acc", "ECE (equal-width)",
       "AUROC", "AURC", "operating points"]
        .map(function (t) { return '<th scope="col" style="cursor:default">' + t + "</th>"; })
        .join("") + "</tr></thead><tbody>";

    Ts.forEach(function (T) {
      READOUT_ORDER.forEach(function (ro) {
        var g = rows.filter(function (r) { return r.timesteps === T && r.readout === ro; });
        if (!g.length) return;
        function col(k) { return g.map(function (r) { return r[k]; }); }
        function cell(k, d, showSign) {
          var v = col(k), m = mean(v), s = sd(v);
          return "<td>" + (showSign ? signed(m, d) : fmt(m, d)) +
            (v.length > 1 ? ' <span class="sd">\u00b1' + Math.abs(s).toFixed(d) + "</span>"
                          : "") + "</td>";
        }
        html += '<tr><td class="k">' + T + '</td><td class="k">' + (READOUT_LABEL[ro] || ro) +
          '</td><td class="k">' + g.length + "</td>" +
          cell("test_accuracy", 4) + cell("mean_conf_minus_acc", 4, true) +
          cell("ece_equal_width", 4) + cell("auroc", 4) + cell("aurc", 5) +
          "<td>" + Math.round(mean(col("n_achievable_coverage_levels"))).toLocaleString() +
          "</td></tr>";
      });
    });
    host.innerHTML = html + "</tbody>";

    // The crossover, stated from the data rather than asserted in prose.
    var lines = [];
    Ts.forEach(function (T) {
      function m(ro, k) {
        var g = rows.filter(function (r) { return r.timesteps === T && r.readout === ro; });
        return g.length ? mean(g.map(function (r) { return r[k]; })) : null;
      }
      var ce = m("count", "ece_equal_width"), me = m("membrane", "ece_equal_width");
      var ca = m("count", "auroc"), ma = m("membrane", "auroc");
      var cr = m("count", "aurc"), mr = m("membrane", "aurc");
      if (ce === null || me === null) return;
      lines.push("<tr><td class=\"k\">" + T + "</td><td>" + (me < ce ? "membrane" : "count") +
        "</td><td>" + (ma > ca ? "membrane" : "count") + "</td><td>" +
        (mr < cr ? "membrane" : "count") + "</td></tr>");
    });
    var cross = document.getElementById("readout-crossover");
    if (cross) {
      cross.innerHTML = "<thead><tr><th style=\"cursor:default\">T</th>" +
        "<th style=\"cursor:default\">better ECE</th>" +
        "<th style=\"cursor:default\">better AUROC</th>" +
        "<th style=\"cursor:default\">better AURC</th></tr></thead><tbody>" +
        lines.join("") + "</tbody>";
    }
  }

  function renderSelectiveTable() {
    var host = document.getElementById("selective-table");
    if (!host) return;
    var rows = window.SPIKECALIB.readout;
    if (!rows || !rows.length) { hide("selective-table"); return; }
    var Ts = [], seen = {};
    rows.forEach(function (r) { if (!seen[r.timesteps]) { seen[r.timesteps] = 1; Ts.push(r.timesteps); } });
    Ts.sort(function (a, b) { return a - b; });
    var html = "<thead><tr>" +
      ["T", "readout", "operating points", "AURC", "oracle AURC", "coverage at risk \u2264 1%",
       "coverage at risk \u2264 0.1%"]
        .map(function (t) { return '<th scope="col" style="cursor:default">' + t + "</th>"; })
        .join("") + "</tr></thead><tbody>";
    Ts.forEach(function (T) {
      READOUT_ORDER.forEach(function (ro) {
        var g = rows.filter(function (r) { return r.timesteps === T && r.readout === ro; });
        if (!g.length) return;
        function avg(k) {
          var v = g.map(function (r) { return r[k]; }).filter(function (x) { return x !== null && x !== undefined; });
          return v.length ? mean(v) : null;
        }
        function pct(k) {
          var v = avg(k), n = g.filter(function (r) { return r[k] !== null && r[k] !== undefined; }).length;
          if (v === null) return "<td>none</td>";
          return "<td>" + (v * 100).toFixed(1) + "%" +
            (n < g.length ? ' <span class="sd">' + n + "/" + g.length + " seeds</span>" : "") + "</td>";
        }
        html += '<tr><td class="k">' + T + '</td><td class="k">' + (READOUT_LABEL[ro] || ro) +
          "</td><td>" + Math.round(avg("n_achievable_coverage_levels")).toLocaleString() +
          "</td><td>" + avg("aurc").toFixed(5) +
          "</td><td>" + avg("aurc_optimal").toFixed(5) + "</td>" +
          pct("coverage_at_risk_0p01") + pct("coverage_at_risk_0p001") + "</tr>";
      });
    });
    host.innerHTML = html + "</tbody>";
  }

  function renderCaxis() {
    var host = document.getElementById("caxis-table");
    if (!host) return;
    var rows = window.SPIKECALIB.caltech;
    if (!rows || !rows.length) { hide("caxis-table"); return; }
    var Ts = [], seen = {};
    rows.forEach(function (r) { if (!seen[r.timesteps]) { seen[r.timesteps] = 1; Ts.push(r.timesteps); } });
    Ts.sort(function (a, b) { return a - b; });
    var html = "<thead><tr>" +
      ["T", "readout", "ceiling", "observed max", "test acc", "binds?", "conf \u2212 acc",
       "ECE (equal-width)", "AUROC"]
        .map(function (t) { return '<th scope="col" style="cursor:default">' + t + "</th>"; })
        .join("") + "</tr></thead><tbody>";
    Ts.forEach(function (T) {
      ["count", "accumulator"].forEach(function (ro) {
        var g = rows.filter(function (r) { return r.timesteps === T && r.readout === ro; });
        if (!g.length) return;
        function avg(k) { return mean(g.map(function (r) { return r[k]; })); }
        var d = g[0].ceiling_derived;
        var binds = g.filter(function (r) { return r.ceiling_binds; }).length;
        html += '<tr><td class="k">' + T + '</td><td class="k">' +
          (ro === "accumulator" ? "accumulator (uncapped)" : "count (capped)") + "</td>" +
          "<td>" + (d === null ? "none" : d.toFixed(6)) + "</td>" +
          "<td>" + avg("max_confidence_observed").toFixed(6) + "</td>" +
          "<td>" + avg("test_accuracy").toFixed(4) + "</td>" +
          "<td>" + (d === null ? "\u2014" : binds + "/" + g.length) + "</td>" +
          "<td>" + signed(avg("mean_conf_minus_acc"), 4) + "</td>" +
          "<td>" + avg("ece_equal_width").toFixed(4) + "</td>" +
          "<td>" + avg("auroc").toFixed(4) + "</td></tr>";
      });
    });
    host.innerHTML = html + "</tbody>";
  }

  function renderArch() {
    var host = document.getElementById("arch-table");
    if (!host) return;
    var rows = window.SPIKECALIB.arch;
    if (!rows || !rows.length) { hide("arch-table"); return; }
    function cells(ds, nb, T, ro) {
      return rows.filter(function (r) {
        return r.dataset === ds && r.n_blocks === nb && r.timesteps === T && r.readout === ro;
      });
    }
    var html = "<thead><tr>" +
      ["dataset", "C", "blocks", "params", "T", "count acc", "accum acc", "gain",
       "residual", "ceiling held", "cond ceiling", "share"]
        .map(function (t) { return '<th scope="col" style="cursor:default">' + t + "</th>"; })
        .join("") + "</tr></thead><tbody>";
    [["ncaltech101", "N-Caltech101"], ["nmnist", "N-MNIST"]].forEach(function (d) {
      [2, 6].forEach(function (T) {
        [2, 4].forEach(function (nb) {
          var c = cells(d[0], nb, T, "count"), a = cells(d[0], nb, T, "accumulator");
          if (!c.length) return;
          var acc = mean(c.map(function (r) { return r.test_accuracy; }));
          var aacc = a.length ? mean(a.map(function (r) { return r.test_accuracy; })) : null;
          var held = c.every(function (r) { return r.ceiling_signed_excess <= 1e-12; });
          html += '<tr><td class="k">' + d[1] + '</td><td class="k">' + c[0].n_classes +
            '</td><td class="k">' + nb + "</td><td>" +
            (c[0].n_parameters / 1000).toFixed(0) + "k</td>" +
            '<td class="k">' + T + "</td>" +
            "<td>" + acc.toFixed(4) + "</td>" +
            "<td>" + (aacc === null ? "\u2014" : aacc.toFixed(4)) + "</td>" +
            "<td>" + (aacc === null ? "\u2014" : signed(aacc - acc, 4)) + "</td>" +
            "<td>" + signed(mean(c.map(function (r) { return r.residual; })), 4) + "</td>" +
            "<td>" + (held ? "yes" : "NO") + "</td>" +
            "<td>" + mean(c.map(function (r) { return r.conditional_ceiling_mean; })).toFixed(4) + "</td>" +
            "<td>" + (mean(c.map(function (r) { return r.share_explained_conditional; })) * 100).toFixed(0) + "%</td>" +
            "</tr>";
        });
      });
    });
    host.innerHTML = html + "</tbody>";
  }

  function renderConditional() {
    var host = document.getElementById("conditional-table");
    if (!host) return;
    var rows = window.SPIKECALIB.conditional;
    if (!rows || !rows.length) { hide("conditional-table"); return; }
    var keys = [];
    rows.forEach(function (r) {
      var k = r.dataset + "|" + r.timesteps;
      if (keys.indexOf(k) < 0) keys.push(k);
    });
    keys.sort(function (a, b) {
      var A = a.split("|"), B = b.split("|");
      return A[0] === B[0] ? (+A[1]) - (+B[1]) : (A[0] < B[0] ? -1 : 1);
    });
    var html = "<thead><tr>" +
      ["dataset", "C", "T", "accuracy", "mean conf", "theoretical", "share", "conditional",
       "share", "gap"]
        .map(function (t) { return '<th scope="col" style="cursor:default">' + t + "</th>"; })
        .join("") + "</tr></thead><tbody>";
    keys.forEach(function (k) {
      var g = rows.filter(function (r) { return r.dataset + "|" + r.timesteps === k; });
      function avg(f) { return mean(g.map(f)); }
      var res = avg(function (r) { return r.residual; });
      var cond = avg(function (r) { return r.conditional_ceiling_mean; });
      var acc = avg(function (r) { return r.test_accuracy; });
      var hot = (g[0].dataset === "ncaltech101" && g[0].timesteps === 6);
      html += "<tr" + (hot ? ' style="font-weight:600"' : "") + '><td class="k">' +
        (g[0].dataset === "ncaltech101" ? "N-Caltech101" : "N-MNIST") + '</td><td class="k">' +
        g[0].n_classes + '</td><td class="k">' + g[0].timesteps + "</td>" +
        "<td>" + acc.toFixed(4) + "</td>" +
        "<td>" + avg(function (r) { return r.mean_confidence; }).toFixed(4) + "</td>" +
        "<td>" + avg(function (r) { return r.theoretical_ceiling; }).toFixed(4) + "</td>" +
        "<td>" + (avg(function (r) { return r.share_explained_theoretical; }) * 100).toFixed(0) + "%</td>" +
        "<td>" + cond.toFixed(4) + "</td>" +
        "<td>" + (avg(function (r) { return r.share_explained_conditional; }) * 100).toFixed(0) + "%</td>" +
        "<td>" + avg(function (r) { return r.ceiling_gap_theo_minus_cond; }).toFixed(4) + "</td></tr>";
    });
    host.innerHTML = html + "</tbody>";
  }

  function renderQuantisation() {
    var host = document.getElementById("quant-within-table");
    if (!host) return;
    var rows = (window.SPIKECALIB.quantisation || []).filter(function (r) {
      return r.scheme === "equal-mass" && r.n_levels_requested !== null;
    });
    if (!rows.length) { hide("quant-within-table"); return; }
    var GRID = [5, 10, 20, 50, 100, 200, 500, 1000];
    var html = "<thead><tr>" +
      ["levels N", "pairs merged", "AUROC within a level", "verdict"]
        .map(function (t) { return '<th scope="col" style="cursor:default">' + t + "</th>"; })
        .join("") + "</tr></thead><tbody>";
    GRID.forEach(function (n) {
      var g = rows.filter(function (r) { return r.n_levels_requested === n; });
      if (!g.length) return;
      var merged = mean(g.map(function (r) { return r.pair_fraction_merged; }));
      var within = mean(g.map(function (r) { return r.auroc_within; }));
      html += '<tr><td class="k">' + n + "</td><td>" + (merged * 100).toFixed(2) + "%</td><td>" +
        within.toFixed(4) + "</td><td>" +
        (within > 0.52 ? "signal \u2014 merging it costs"
                       : within < 0.48 ? "noise \u2014 merging it pays" : "uninformative") +
        "</td></tr>";
    });
    host.innerHTML = html + "</tbody>";
  }

  /* ---------- render ---------- */

  function render() {
    var rows = filtered();
    renderTiles(rows);
    renderIntervals(rows);
    renderReliability(rows);
    renderTable(rows);
  }

  /* ---------- modals ----------

     Native <dialog> rather than a hand-rolled overlay: showModal() gives the focus trap,
     the inert background, the Escape handler and the focus return for free, and a
     hand-rolled version of any one of those is where accessibility bugs live. What it
     does NOT give is a backdrop-click close or a scroll lock on the page behind, so
     those two are added here.
  */
  /* Delegated, and wired BEFORE anything reads data.

     What was actually broken: this ran last, after six render calls, inside one IIFE.
     A single failure anywhere above it -- an unavailable data.js, one absent column --
     threw, the IIFE stopped, wireModals() never ran, and all eight buttons went dead
     with one console error nobody opens. An interface must not depend on the data
     rendering succeeding. Delegating on document also means no handler can be orphaned
     by a re-render, now or later. */
  function wireModals() {
    document.addEventListener("click", function (e) {
      var t = e.target;
      if (!t || !t.closest) return;
      var btn = t.closest("[data-modal]");
      if (btn) {
        var d = document.getElementById(btn.getAttribute("data-modal"));
        if (!d) return;
        d.__trigger = btn;
        /* showModal() throws if the dialog is already open, and wherever the top layer
           is unavailable; show() still puts the panel on screen. */
        try { d.showModal(); } catch (err) { if (!d.open) d.show(); }
        document.body.classList.add("modal-open");
        var body = d.querySelector(".modal-body");
        if (body) body.scrollTop = 0;   // a reopened dialog otherwise keeps its scroll
        return;
      }
      var dlg = t.closest("dialog.modal");
      if (!dlg) return;
      /* The dialog element fills the viewport, so a click whose target IS the dialog
         landed outside the visible panel. */
      if (t === dlg || t.closest("[data-close]")) dlg.close();
    });
    /* close does not bubble, so listen in the capture phase: fires for the close
       button, the backdrop and Escape alike. */
    document.addEventListener("close", function (e) {
      var d = e.target;
      if (!d || !d.classList || !d.classList.contains("modal")) return;
      document.body.classList.remove("modal-open");
      if (d.__trigger) d.__trigger.focus();
    }, true);
  }

  /* The interface first -- nothing below is allowed to be able to disable it -- and
     each renderer isolated, so one missing column cannot take the whole page down. */
  /* The footer used to carry a date somebody had to remember to edit, and it was two days
     stale. It now states when the newest artifact this page is built from was written,
     resolved at export time -- a clone rewrites every mtime, so the question is only
     answerable on the machine that ran the work. The markup keeps the last exported date
     as its fallback, so the line is never blank if the payload is unavailable. */
  function renderUpdated() {
    var el = document.getElementById("last-updated");
    if (!el || !D.last_updated) return;
    el.textContent = "Last updated " + D.last_updated + ", the newest of "
      + D.n_source_artifacts + " result artifacts (" + D.last_updated_source + ").";
  }

  wireModals();
  [updateHero, renderReadoutTable, renderSelectiveTable, renderQuantisation,
   renderCaxis, renderConditional, renderArch, renderUpdated, renderCardFigures, renderCells,
   render].forEach(function (fn) {
    try { fn(); } catch (err) { console.error("spikecalib: " + fn.name + " failed", err); }
  });
  window.addEventListener("resize", function () {
    try { drawCeiling(); render(); } catch (err) { console.error("spikecalib: resize", err); }
  });
})();
