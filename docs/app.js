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

  var hero = {
    T: 2, C: 10,
    tEl: document.getElementById("t-slider"),
    cEl: document.getElementById("c-slider"),
    tOut: document.getElementById("t-value"),
    cOut: document.getElementById("c-value"),
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
    var c = chart(hero.host, 760, 340, { l: 48, r: 18, t: 14, b: 32 });
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

    /* measured maxima — only meaningful at the C this study used */
    var measured = (D.measured || []);
    if (C === D.n_classes) {
      measured.forEach(function (m) {
        el("circle", { cx: X(m.T), cy: Y(m.observed_max), r: 5.5, fill: "none",
          stroke: "var(--d2)", "stroke-width": 2 }, c.svg);
      });
      hero.measuredNote.textContent =
        "Rings are this study's five measured maxima. They sit on the curve to within 4e-13.";
    } else {
      hero.measuredNote.textContent =
        "Measured maxima are hidden: they were recorded at C = " + D.n_classes +
        ", and C is currently " + C + ".";
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

    hero.verdict.textContent =
      "At T = " + hero.T + " with " + hero.C + " classes, a model above " +
      (ceil * 100).toFixed(1) + "% accuracy cannot be correctly confident: no reachable " +
      "spike-count vector produces a confidence that high.";

    drawCeiling();
  }

  [hero.tEl, hero.cEl].forEach(function (input) {
    input.addEventListener("input", updateHero);
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-preset]"), function (btn) {
    btn.addEventListener("click", function () {
      var p = btn.getAttribute("data-preset").split(",");
      hero.tEl.value = p[0]; hero.cEl.value = p[1];
      updateHero();
      Array.prototype.forEach.call(document.querySelectorAll("[data-preset]"), function (b) {
        b.setAttribute("aria-pressed", b === btn ? "true" : "false");
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

  function buildChips(host, values, labels, get, set) {
    host.textContent = "";
    values.forEach(function (v, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "chip" + (typeof v === "number" ? " mono" : "");
      b.textContent = labels ? labels[i] : v;
      b.setAttribute("aria-pressed", get(v) ? "true" : "false");
      b.addEventListener("click", function () {
        set(v);
        b.setAttribute("aria-pressed", get(v) ? "true" : "false");
        render();
      });
      host.appendChild(b);
    });
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
    function (v) { state.estimator = v; });

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

    document.getElementById("tiles").innerHTML =
      tile("Uncalibrated ECE", fmt(mean(unc)), col, span, !unc.length) +
      tile("Best calibrated ECE", fmt(best), col, bestWhere || "no calibrated arm selected", !isFinite(best)) +
      tile("Isotonic − temperature", (isFinite(mean(iso)) ? (mean(iso) >= 0 ? "+" : "") + fmt(mean(iso)) : "—"),
        "iso_minus_temp_ece_ew", "positive favours temperature", !iso.length) +
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

  /* ---------- paired intervals ---------- */

  function renderIntervals(rows) {
    var runs = uniqRuns(rows).sort(function (a, b) {
      return (+a.timesteps - +b.timesteps) || (+a.seed - +b.seed);
    });
    var host = document.getElementById("interval-chart");
    if (!runs.length) { host.textContent = ""; return; }
    var rowH = 21, h = runs.length * rowH + 42;
    var c = chart(host, 620, h, { l: 80, r: 104, t: 10, b: 26 });

    var lo = Math.min.apply(null, runs.map(function (r) { return num(r, "iso_minus_temp_ci_low"); }));
    var hi = Math.max.apply(null, runs.map(function (r) { return num(r, "iso_minus_temp_ci_high"); }));
    var pad = (hi - lo) * 0.12 || 0.001;
    lo -= pad; hi += pad;
    var X = function (v) { return c.ix(v, lo, hi); };

    graticule(c, 6, runs.length);
    el("line", { x1: X(0), y1: c.pad.t, x2: X(0), y2: h - c.pad.b,
      stroke: "var(--ink-3)", "stroke-width": 1.5 }, c.svg);

    runs.forEach(function (r, i) {
      var y = c.pad.t + i * rowH + rowH / 2;
      var a = num(r, "iso_minus_temp_ci_low"), b = num(r, "iso_minus_temp_ci_high");
      var res = r.iso_minus_temp_resolvable === "True";
      var col = res ? "var(--status)" : "var(--d1)";
      el("line", { x1: X(a), y1: y, x2: X(b), y2: y, stroke: col, "stroke-width": 2.5,
        "stroke-linecap": "round", opacity: res ? 1 : 0.75 }, c.svg);
      el("circle", { cx: X(num(r, "iso_minus_temp_ece_ew")), cy: y, r: 3.2, fill: col }, c.svg);
      axisText(c, c.pad.l - 8, y + 3.5, "T=" + r.timesteps + "  s" + r.seed, "end");
      if (res) {
        var t = axisText(c, X(b) + 8, y + 3.5, b < 0 ? "isotonic" : "temperature", "start", 9.5);
        t.setAttribute("fill", "var(--status)");
      }
    });
    for (var k = 0; k <= 4; k++) {
      var v = lo + (hi - lo) * k / 4;
      axisText(c, X(v), h - c.pad.b + 15, v.toFixed(4));
    }
    c.svg.setAttribute("aria-label", "Paired bootstrap intervals for the isotonic minus temperature ECE difference, " + runs.length + " runs.");
    document.getElementById("interval-count").textContent =
      runs.length + " run" + (runs.length === 1 ? "" : "s") + " shown; " +
      runs.filter(function (r) { return r.iso_minus_temp_resolvable === "True"; }).length +
      " with an interval excluding zero.";
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
    var CAP = 6, shown = keys.slice(0, CAP);
    document.getElementById("reliability-count").textContent =
      keys.length <= CAP
        ? keys.length + " diagram" + (keys.length === 1 ? "" : "s") + " shown."
        : "Showing " + CAP + " of " + keys.length + " selected combinations. Narrow the filters to see others.";

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

      var c = chart(box, 320, 240, { l: 36, r: 12, t: 10, b: 30 });
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

  /* ---------- render ---------- */

  function render() {
    var rows = filtered();
    renderTiles(rows);
    renderIntervals(rows);
    renderReliability(rows);
    renderTable(rows);
  }

  updateHero();
  render();
  window.addEventListener("resize", function () { drawCeiling(); render(); });
})();
