/* SpikeCalib: chart-first summary of the paper, and the local briefing page.

   Plain JavaScript, hand-drawn SVG, no build step; it opens from file://. Every number on
   the page is computed here from window.SPIKECALIB (docs/data.js), which
   scripts/export_dashboard_data.py writes from the result artifacts. The HTML types none.

   Two rules from the old page are kept. The interface is wired before any data is read,
   so a missing field cannot leave the page dead. Each renderer runs on its own, inside
   try/catch, and draws only into elements that exist, which is how the briefing page
   reuses this file. */
(function () {
  "use strict";

  /* ---------- interface first: the theme toggle and the tooltip ---------- */
  var root = document.documentElement;
  function setTheme(t) {
    root.setAttribute("data-theme", t);
    var b = document.getElementById("theme-toggle");
    if (b) { b.textContent = t === "dark" ? "Light" : "Dark"; b.setAttribute("aria-pressed", String(t === "dark")); }
    try { localStorage.setItem("spikecalib-theme", t); } catch (e) { /* storage may be unavailable */ }
  }
  (function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("spikecalib-theme"); } catch (e) { saved = null; }
    setTheme(saved === "dark" ? "dark" : "light");          // light by default
    var b = document.getElementById("theme-toggle");
    if (b) b.addEventListener("click", function () {
      setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
    });
  })();

  var tipEl = document.getElementById("tip");
  function showTip(evt, title, rows) {
    if (!tipEl) return;
    tipEl.textContent = "";
    var tt = document.createElement("div"); tt.className = "tt"; tt.textContent = title; tipEl.appendChild(tt);
    rows.forEach(function (r) {
      var row = document.createElement("div"); row.className = "row";
      var i = document.createElement("i"); i.className = r.k ? "k-" + r.k : "k-ink";
      var b = document.createElement("b"); b.textContent = r.v;
      var s = document.createElement("span"); s.textContent = r.label;
      row.appendChild(i); row.appendChild(b); row.appendChild(s); tipEl.appendChild(row);
    });
    tipEl.style.display = "block";
    var x, y;
    if (evt && evt.clientX !== undefined && evt.type !== "focus") { x = evt.clientX; y = evt.clientY; }
    else { var bb = evt.target.getBoundingClientRect(); x = bb.left + bb.width / 2; y = bb.top; }
    var w = tipEl.offsetWidth, h = tipEl.offsetHeight;
    var left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 14));
    var top = y - h - 12 < 8 ? y + 16 : y - h - 12;
    tipEl.style.left = left + "px"; tipEl.style.top = top + "px";
  }
  function hideTip() { if (tipEl) tipEl.style.display = "none"; }

  /* ---------- data and small helpers ---------- */
  var D = window.SPIKECALIB || {};
  var CSV = window.SPIKECALIB_CSV || "";

  function mean(a) { return a.reduce(function (s, v) { return s + v; }, 0) / a.length; }
  function where(rows, f) { return (rows || []).filter(f); }
  function uniq(a) { return a.filter(function (v, i) { return a.indexOf(v) === i; }); }
  function fx(x, d) { return Number(x).toFixed(d); }
  function sg(x, d) { return (x < 0 ? "−" : "+") + Math.abs(x).toFixed(d); }
  function pc(x, d) { return (100 * x).toFixed(d) + "%"; }
  function setN(key, text) {
    Array.prototype.forEach.call(document.querySelectorAll('[data-n="' + key + '"]'), function (e) { e.textContent = text; });
  }
  var SEEDS = function (rows) { return uniq(rows.map(function (r) { return r.seed; })).length; };

  function parseCSV(text) {
    var lines = text.trim().split("\n"), head = lines[0].split(",");
    return lines.slice(1).map(function (l) {
      var c = l.split(","), o = {};
      head.forEach(function (h, i) { o[h] = c[i]; });
      o.timesteps = +o.timesteps; o.seed = +o.seed;
      return o;
    });
  }
  var SWEEP = CSV ? parseCSV(CSV) : [];

  /* ---------- SVG primitives ---------- */
  var NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(s, attrs, parent) { var t = el("text", attrs, parent); t.textContent = s; return t; }
  function lin(d0, d1, r0, r1) { return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); }; }
  function niceTicks(lo, hi, n) {
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var t = [], v = Math.ceil(lo / step - 1e-9) * step;
    for (; v <= hi + step * 1e-6; v += step) t.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return t;
  }
  function decimals(ticks) {
    var step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
    return Math.max(0, Math.min(5, -Math.floor(Math.log10(step) + 1e-9)));
  }

  /* A plot frame: grid, axes and scales, sized to the container's current width. */
  function frame(host, o) {
    host.textContent = "";
    var W = Math.max(240, host.clientWidth || 300);
    var H = o.height || Math.round(Math.min(300, Math.max(200, W * 0.58)));
    var m = { t: 10, r: o.right || 14, b: o.xLabel ? 40 : 26, l: o.left || 48 };
    var svg = el("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.aria || "" }, host);
    var x = lin(o.x[0], o.x[1], m.l, W - m.r), y = lin(o.y[0], o.y[1], H - m.b, m.t);
    var g = el("g", { "class": "grid" }, svg), ax = el("g", { "class": "axis" }, svg);
    var yt = o.yTicks || niceTicks(o.y[0], o.y[1], 4), yd = o.yDec !== undefined ? o.yDec : decimals(yt);
    yt.forEach(function (v) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, g);
      txt(o.yFmt ? o.yFmt(v) : (o.ySigned && v !== 0 ? sg(v, yd) : fx(v, yd)), { x: m.l - 6, y: y(v) + 3.5, "text-anchor": "end" }, svg);
    });
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, ax);
    (o.xTicks || niceTicks(o.x[0], o.x[1], 5)).forEach(function (v) {
      el("line", { x1: x(v), x2: x(v), y1: H - m.b, y2: H - m.b + 4 }, ax);
      txt(o.xFmt ? o.xFmt(v) : String(v), { x: x(v), y: H - m.b + 16, "text-anchor": "middle" }, svg);
    });
    if (o.xLabel) txt(o.xLabel, { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", "class": "t-ink" }, svg);
    if (o.zero) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), "class": "zero" }, svg);
      if (o.zeroLabel) txt(o.zeroLabel, { x: W - m.r + 12, y: y(0) + 4, "text-anchor": "start", "class": "t-ink" }, svg);
    }
    return { svg: svg, x: x, y: y, W: W, H: H, m: m, plot: el("g", {}, svg) };
  }

  function path(f, pts, cls, parent) {
    var d = pts.map(function (p, i) { return (i ? "L" : "M") + f.x(p[0]).toFixed(1) + "," + f.y(p[1]).toFixed(1); }).join("");
    return el("path", { d: d, "class": "ln " + cls }, parent || f.plot);
  }
  function dot(f, xv, yv, key, open, parent) {
    return el("circle", { cx: f.x(xv), cy: f.y(yv), r: 4.5, "class": open ? "open c-" + key : "mk f-" + key }, parent || f.plot);
  }
  function diamond(f, xv, yv, key, parent) {
    var cx = f.x(xv), cy = f.y(yv), s = 5.5;
    return el("path", { d: "M" + cx + "," + (cy - s) + "L" + (cx + s) + "," + cy + "L" + cx + "," + (cy + s) + "L" + (cx - s) + "," + cy + "Z",
      "class": "open c-" + key }, parent || f.plot);
  }
  /* Column hit targets: a crosshair that finds the x, and one tooltip listing every series. */
  function columns(f, xs, rowsAt, titleAt) {
    var hair = el("line", { y1: f.m.t, y2: f.H - f.m.b, "class": "zero", opacity: 0 }, f.svg);
    xs.forEach(function (xv, i) {
      var left = i === 0 ? f.m.l : (f.x(xs[i - 1]) + f.x(xv)) / 2;
      var right = i === xs.length - 1 ? f.W - f.m.r : (f.x(xv) + f.x(xs[i + 1])) / 2;
      var r = el("rect", { x: left, y: f.m.t, width: Math.max(1, right - left), height: f.H - f.m.b - f.m.t,
        "class": "hit", tabindex: 0, "aria-label": titleAt(xv) }, f.svg);
      function on(e) { hair.setAttribute("x1", f.x(xv)); hair.setAttribute("x2", f.x(xv)); hair.setAttribute("opacity", 0.35); showTip(e, titleAt(xv), rowsAt(xv)); }
      function off() { hair.setAttribute("opacity", 0); hideTip(); }
      r.addEventListener("pointermove", on); r.addEventListener("pointerleave", off);
      r.addEventListener("focus", on); r.addEventListener("blur", off);
    });
  }
  /* Row hit targets for dot plots and interval plots. */
  function rowHit(f, yc, h, title, rows) {
    var r = el("rect", { x: 0, y: yc - h / 2, width: f.W, height: h, "class": "hit", tabindex: 0, "aria-label": title }, f.svg);
    function on(e) { r.classList.add("on"); showTip(e, title, rows); }
    function off() { r.classList.remove("on"); hideTip(); }
    r.addEventListener("pointermove", on); r.addEventListener("pointerleave", off);
    r.addEventListener("focus", on); r.addEventListener("blur", off);
  }
  function table(id, head, rows, textCols) {
    var t = document.getElementById(id);
    if (!t) return;
    t.textContent = "";
    var thead = document.createElement("thead"), tr = document.createElement("tr");
    head.forEach(function (h, i) { var th = document.createElement("th"); th.textContent = h; if (textCols && textCols.indexOf(i) >= 0) th.className = "t"; tr.appendChild(th); });
    thead.appendChild(tr); t.appendChild(thead);
    var tb = document.createElement("tbody");
    rows.forEach(function (r) {
      var tr2 = document.createElement("tr");
      r.forEach(function (c, i) { var td = document.createElement("td"); td.textContent = c; if (textCols && textCols.indexOf(i) >= 0) td.className = "t"; tr2.appendChild(td); });
      tb.appendChild(tr2);
    });
    t.appendChild(tb);
  }
  function pad(lo, hi, f) { var s = (hi - lo) || Math.abs(hi) || 1; return [lo - s * f, hi + s * f]; }

  /* ---------- derived series ---------- */
  var READOUT_KEY = { "count": "count", "accumulator": "accum", "membrane": "membrane", "weighted-count": "wcount" };
  var READOUT_NAME = { "count": "count", "accumulator": "accumulator", "membrane": "membrane", "weighted-count": "weighted count" };
  function byT(rows, readout, key) {
    var r = where(rows, function (x) { return x.readout === readout; });
    return uniq(r.map(function (x) { return x.timesteps; })).sort(function (a, b) { return a - b; }).map(function (T) {
      return [T, mean(where(r, function (x) { return x.timesteps === T; }).map(function (x) { return x[key]; }))];
    });
  }
  function ceiling(T, C) { return Math.exp(T) / (Math.exp(T) + C - 1); }
  function sweepMean(T, arm, col) {
    return mean(where(SWEEP, function (r) { return r.timesteps === T && r.arm === arm; }).map(function (r) { return +r[col]; }));
  }
  function pooledReliability(T, arm) {
    var bins = {};
    Object.keys(D.reliability || {}).forEach(function (k) {
      var p = k.split("|");
      if (+p[0] !== T || p[2] !== arm) return;
      (D.reliability[k].equal_width || []).forEach(function (b) {
        var key = b.lo.toFixed(6), o = bins[key] || (bins[key] = { n: 0, c: 0, a: 0, lo: b.lo, hi: b.hi });
        o.n += b.n; o.c += b.conf * b.n; o.a += b.acc * b.n;
      });
    });
    return Object.keys(bins).map(function (k) { var o = bins[k]; return { lo: o.lo, hi: o.hi, n: o.n, conf: o.c / o.n, acc: o.a / o.n }; })
      .filter(function (b) { return b.n > 0; }).sort(function (a, b) { return a.lo - b.lo; });
  }

  /* ---------- 1. hero tiles and every counted number ---------- */
  function renderNumbers() {
    var S = D.summary, L = D.ledger;
    if (S) {
      var FR = S.floor_ratio_t2 || S.share_explained_t2;
      setN("share-c10", pc(FR.c10, 1));
      setN("share-c101", pc(FR.c101, 1));
      setN("ece-raw-ew", fx(S.ece_t2.uncalibrated.ew, 4)); setN("ece-temp-ew", fx(S.ece_t2.temperature.ew, 4));
      setN("ece-raw-em", fx(S.ece_t2.uncalibrated.em, 4)); setN("ece-temp-em", fx(S.ece_t2.temperature.em, 4));
      setN("auroc-count", fx(S.auroc_t2.count, 4)); setN("auroc-accum", fx(S.auroc_t2.accumulator, 4));
      setN("cells-all", String(S.cells_union));
    }
    if (L) {
      setN("pred-held", String(L.paper.held)); setN("pred-false", String(L.paper.falsified));
      setN("pred-total", String(L.paper.total));
      setN("early-held", String(L.early.held)); setN("early-false", String(L.early.falsified));
      setN("all-held", String(L.all.held)); setN("all-false", String(L.all.falsified));
    }
    setN("c-small", String(D.n_classes));
    var cc = D.caltech && D.caltech[0]; if (cc) setN("c-large", String(cc.n_classes));
    setN("n-bins", String(D.n_bins));
    if (D.readout) setN("n-seeds", String(SEEDS(D.readout)));
    if (CSV) {
      var lines = CSV.trim().split("\n");
      setN("csv-columns", String(lines[0].split(",").length));
      setN("csv-rows", String(lines.length - 1));
    }
    var cal = D.caltech_calibrators;
    if (cal) {
      var cnt = where(cal.paired, function (p) { return p.readout === "count"; });
      setN("nc-runs", String(cnt.length));
      setN("nc-differ", String(where(cnt, function (p) { return p.differ && p.favours === "isotonic"; }).length));
    }
    var sc = D.scaled;
    if (sc) {
      var six = where(sc.rows, function (r) { return r.dataset === "ncaltech101" && r.timesteps === 6; });
      setN("scale-cost", fx(100 * (mean(six.map(function (r) { return r.count_test_accuracy; })) -
                                   mean(six.map(function (r) { return r.test_accuracy; }))), 1));
    }
    var fu = D.fusion;
    if (fu) {
      var t2 = where(fu.rows, function (r) { return r.readout === "count" && r.timesteps === 2; });
      var e0 = 1 - mean(t2.map(function (r) { return r.acc_r0; })), e1 = 1 - mean(t2.map(function (r) { return r.acc_r1; }));
      setN("fu-cut", pc(1 - e1 / e0, 0));
      setN("fu-points", fx(100 * mean(t2.map(function (r) { return r.delta; })), 2));
    }
  }

  /* ---------- 2. the cap ---------- */
  function capPanel(hostId, titleId, rows, C, name) {
    var host = document.getElementById(hostId);
    if (!host || !rows || !rows.length) return;
    var T = uniq(rows.map(function (r) { return r.timesteps; })).sort(function (a, b) { return a - b; });
    var mx = byT(rows, "count", "max_confidence_observed"), ac = byT(rows, "count", "test_accuracy");
    var tTitle = document.getElementById(titleId);
    if (tTitle) tTitle.textContent = name + ", C = " + C;
    var Tmax = T[T.length - 1];
    var f = frame(host, { x: [1, Tmax], y: [0, 1], xTicks: niceTicks(2, Tmax, 5).filter(function (v) { return v % 2 === 0; }),
      yTicks: [0, 0.25, 0.5, 0.75, 1], yDec: 2, xLabel: "timesteps T", aria: "ceiling against T for C = " + C });
    var curve = []; for (var t = 1; t <= Tmax + 1e-9; t += 0.05) curve.push([t, ceiling(t, C)]);
    path(f, curve, "c-ink");
    ac.forEach(function (p) { diamond(f, p[0], p[1], "muted"); });
    mx.forEach(function (p) { dot(f, p[0], p[1], "count"); });
    var lt = Math.log(0.6 * (C - 1) / 0.4);
    if (lt > 1 && lt < Tmax) txt("ceiling", { x: f.x(lt) - 10, y: f.y(0.6) - 2, "text-anchor": "end", "class": "t-ink" }, f.svg);
    var la = ac[0]; txt("accuracy", { x: f.x(la[0]) + 8, y: f.y(la[1]) - 9, "class": "t-ink" }, f.svg);
    columns(f, T, function (Tv) {
      var m = mx.filter(function (p) { return p[0] === Tv; })[0], a = ac.filter(function (p) { return p[0] === Tv; })[0];
      return [{ k: "ink", v: fx(ceiling(Tv, C), 4), label: "ceiling" },
              { k: "count", v: fx(m[1], 4), label: "max confidence" },
              { k: "muted", v: fx(a[1], 4), label: "accuracy" }];
    }, function (Tv) { return name + ", T = " + Tv; });
    return T.map(function (Tv) {
      var m = mx.filter(function (p) { return p[0] === Tv; })[0], a = ac.filter(function (p) { return p[0] === Tv; })[0];
      return [name, String(Tv), fx(ceiling(Tv, C), 4), fx(m[1], 4), fx(a[1], 4)];
    });
  }
  function renderCap() {
    var cc = D.caltech && D.caltech[0];
    var a = capPanel("chart-cap-a", "cap-a-title", D.readout, D.n_classes, "N-MNIST") || [];
    var b = cc ? (capPanel("chart-cap-b", "cap-b-title", D.caltech, cc.n_classes, "N-Caltech101") || []) : [];
    table("table-cap", ["dataset", "T", "ceiling", "max confidence", "accuracy"], a.concat(b));
  }

  /* ---------- 3. calibration fixes the average, not the ranking ---------- */
  function reliabilityPanel(hostId, T, compact) {
    var host = document.getElementById(hostId);
    if (!host || !D.reliability) return;
    var raw = pooledReliability(T, "uncalibrated"), tem = pooledReliability(T, "temperature");
    var W = host.clientWidth || 280;
    var f = frame(host, { x: [0, 1], y: [0, 1], xTicks: [0, 0.5, 1], yTicks: [0, 0.5, 1], yDec: 1, xFmt: function (v) { return fx(v, 1); },
      height: compact ? Math.min(200, W * 0.75) : Math.min(260, Math.max(200, W * 0.85)), xLabel: "confidence", left: 36,
      aria: "reliability diagram at T = " + T });
    el("line", { x1: f.x(0), y1: f.y(0), x2: f.x(1), y2: f.y(1), "class": "diag" }, f.plot);
    var ang = -Math.atan2(f.y(0) - f.y(1), f.x(1) - f.x(0)) * 180 / Math.PI;
    txt("calibrated", { x: f.x(0.55), y: f.y(0.55) + 13, "class": "t-ink", transform: "rotate(" + ang + " " + f.x(0.55) + " " + (f.y(0.55) + 13) + ")" }, f.svg);
    var nMax = Math.max.apply(null, raw.concat(tem).map(function (b) { return b.n; }));
    function marks(bins, key, label) {
      bins.forEach(function (b) {
        el("circle", { cx: f.x(b.conf), cy: f.y(b.acc), r: (2.5 + 9 * Math.sqrt(b.n / nMax)).toFixed(1), "class": "mk f-" + key, "fill-opacity": 0.85 }, f.plot);
        var hit = el("circle", { cx: f.x(b.conf), cy: f.y(b.acc), r: 12, "class": "hit", tabindex: 0, "aria-label": label }, f.svg);
        var rows = [{ k: key, v: fx(b.acc, 3), label: "accuracy" }, { k: key, v: fx(b.conf, 3), label: "mean confidence" },
                    { k: key, v: b.n.toLocaleString(), label: "predictions in bin" }];
        var title = label + ", confidence " + fx(b.lo, 2) + "–" + fx(b.hi, 2);
        hit.addEventListener("pointermove", function (e) { showTip(e, title, rows); });
        hit.addEventListener("pointerleave", hideTip);
        hit.addEventListener("focus", function (e) { showTip(e, title, rows); });
        hit.addEventListener("blur", hideTip);
      });
    }
    marks(raw, "muted", "raw, T = " + T);
    marks(tem, "count", "temperature-scaled, T = " + T);
  }
  function renderAuroc(hostId) {
    var host = document.getElementById(hostId || "chart-auroc");
    if (!host || !D.readout) return;
    var c = byT(D.readout, "count", "auroc"), a = byT(D.readout, "accumulator", "auroc");
    var all = c.concat(a).map(function (p) { return p[1]; });
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), yr = pad(lo, hi, 0.12);
    var W = host.clientWidth || 280;
    var f = frame(host, { x: [2, 10], y: yr, xTicks: [2, 4, 6, 8, 10], xLabel: "timesteps T", height: Math.min(260, Math.max(200, W * 0.85)), left: 46,
      aria: "error-detection AUROC against T" });
    path(f, c, "c-count"); path(f, a, "c-accum");
    c.forEach(function (p) { dot(f, p[0], p[1], "count"); });
    a.forEach(function (p) { dot(f, p[0], p[1], "accum"); });
    var ce = c[c.length - 1], ae = a[a.length - 1], up = ce[1] >= ae[1];
    txt("count", { x: f.x(ce[0]) - 2, y: f.y(ce[1]) + (up ? -9 : 16), "text-anchor": "end", "class": "t-ink" }, f.svg);
    txt("accumulator", { x: f.x(ae[0]) - 2, y: f.y(ae[1]) + (up ? 16 : -9), "text-anchor": "end", "class": "t-ink" }, f.svg);
    columns(f, c.map(function (p) { return p[0]; }), function (T) {
      var rows = [{ k: "count", v: fx(c.filter(function (p) { return p[0] === T; })[0][1], 4), label: "count" }];
      var ap = a.filter(function (p) { return p[0] === T; })[0];
      if (ap) rows.push({ k: "accum", v: fx(ap[1], 4), label: "accumulator" });
      return rows;
    }, function (T) { return "AUROC, T = " + T; });
    return { c: c, a: a };
  }
  function renderCalibration() {
    reliabilityPanel("chart-rel-2", 2); reliabilityPanel("chart-rel-10", 10);
    var au = renderAuroc();
    if (!SWEEP.length || !au) return;
    var Ts = uniq(SWEEP.map(function (r) { return r.timesteps; })).sort(function (a, b) { return a - b; });
    table("table-calibration", ["T", "ECE raw (w / m)", "ECE temperature (w / m)", "AUROC count", "AUROC accumulator"],
      Ts.map(function (T) {
        var ap = au.a.filter(function (p) { return p[0] === T; })[0];
        return [String(T),
          fx(sweepMean(T, "uncalibrated", "ece_equal_width"), 4) + " / " + fx(sweepMean(T, "uncalibrated", "ece_equal_mass"), 4),
          fx(sweepMean(T, "temperature", "ece_equal_width"), 4) + " / " + fx(sweepMean(T, "temperature", "ece_equal_mass"), 4),
          fx(au.c.filter(function (p) { return p[0] === T; })[0][1], 4), ap ? fx(ap[1], 4) : "–"];
      }));
  }

  /* ---------- 4. the readout sets the direction of failure ---------- */
  function gapPanel(hostId, rows, readouts, name) {
    var host = document.getElementById(hostId);
    if (!host || !rows) return [];
    var series = readouts.map(function (r) { return { r: r, pts: byT(rows, r, "mean_conf_minus_acc") }; });
    var all = []; series.forEach(function (s) { s.pts.forEach(function (p) { all.push(p[1]); }); });
    var lo = Math.min(0, Math.min.apply(null, all)), hi = Math.max(0, Math.max.apply(null, all));
    var Ts = uniq(rows.map(function (r) { return r.timesteps; })).sort(function (a, b) { return a - b; });
    var f = frame(host, { x: [Ts[0], Ts[Ts.length - 1]], y: pad(lo, hi, 0.08), xTicks: Ts, xLabel: "timesteps T", ySigned: true, left: 52,
      right: 72, zero: true, zeroLabel: "calibrated", aria: name + ": confidence minus accuracy against T" });
    var drawOrder = series.slice().reverse();          // count last, so it sits on top
    drawOrder.forEach(function (s) { path(f, s.pts, "c-" + READOUT_KEY[s.r]); });
    drawOrder.forEach(function (s) { s.pts.forEach(function (p) { dot(f, p[0], p[1], READOUT_KEY[s.r]); }); });
    var sc = series.filter(function (s) { return s.r === "count"; })[0], sw = series.filter(function (s) { return s.r === "weighted-count"; })[0];
    if (sc && sw) {
      var gapMax = Math.max.apply(null, sc.pts.map(function (p, i) { return Math.abs(p[1] - sw.pts[i][1]); }));
      if (gapMax < 0.01) txt("count and weighted count overlap", { x: f.x(sc.pts[0][0]) + 10, y: f.y(sc.pts[0][1]) + 4, "class": "t-ink" }, f.svg);
    }
    columns(f, Ts, function (T) {
      return series.map(function (s) {
        var p = s.pts.filter(function (q) { return q[0] === T; })[0];
        return p ? { k: READOUT_KEY[s.r], v: sg(p[1], 4), label: READOUT_NAME[s.r] } : null;
      }).filter(Boolean);
    }, function (T) { return name + ", T = " + T; });
    var out = [];
    series.forEach(function (s) { s.pts.forEach(function (p) { out.push([name, READOUT_NAME[s.r], String(p[0]), sg(p[1], 4)]); }); });
    return out;
  }
  function renderGap() {
    var a = gapPanel("chart-gap-a", D.readout, ["count", "weighted-count", "membrane", "accumulator"], "N-MNIST");
    var b = gapPanel("chart-gap-b", D.caltech, ["count", "accumulator"], "N-Caltech101");
    table("table-gap", ["dataset", "readout", "T", "confidence − accuracy"], a.concat(b));
  }

  /* ---------- 5. a learned scale ---------- */
  function scaleCells() {
    var sc = D.scaled; if (!sc) return [];
    var cells = [["nmnist", 2, "N-MNIST, T = 2"], ["ncaltech101", 2, "N-Caltech101, T = 2"], ["ncaltech101", 6, "N-Caltech101, T = 6"]];
    return cells.map(function (c) {
      var r = where(sc.rows, function (x) { return x.dataset === c[0] && x.timesteps === c[1]; });
      function m(k) { return mean(r.map(function (x) { return x[k]; })); }
      return { label: c[2], gap: { count: m("count_mean_conf_minus_acc"), learned: m("mean_conf_minus_acc"), accum: m("accumulator_mean_conf_minus_acc") },
               acc: { count: m("count_test_accuracy"), learned: m("test_accuracy"), accum: m("accumulator_test_accuracy") } };
    });
  }
  function dotRows(hostId, cells, key, signed, note) {
    var host = document.getElementById(hostId);
    if (!host || !cells.length) return;
    var vals = []; cells.forEach(function (c) { ["count", "learned", "accum"].forEach(function (k) { vals.push(c[key][k]); }); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (signed) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    var dom = pad(lo, hi, 0.1);
    host.textContent = "";
    var W = Math.max(240, host.clientWidth || 300), rowH = 46, top = 8, bottom = 30, left = Math.min(150, Math.max(104, W * 0.3));
    var H = top + rowH * cells.length + bottom;
    var svg = el("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": hostId }, host);
    var x = lin(dom[0], dom[1], left, W - 16);
    var ticks = niceTicks(dom[0], dom[1], 4), dd = decimals(ticks);
    var g = el("g", { "class": "grid" }, svg);
    ticks.forEach(function (v) {
      el("line", { x1: x(v), x2: x(v), y1: top, y2: H - bottom }, g);
      txt(signed && v !== 0 ? sg(v, dd) : fx(v, dd), { x: x(v), y: H - bottom + 16, "text-anchor": "middle" }, svg);
    });
    if (signed) el("line", { x1: x(0), x2: x(0), y1: top, y2: H - bottom, "class": "zero" }, svg);
    var f = { svg: svg, W: W };
    cells.forEach(function (c, i) {
      var yc = top + rowH * (i + 0.5);
      txt(c.label, { x: 0, y: yc + 4, "class": "t-ink" }, svg);
      var v = c[key], xs = [v.count, v.learned, v.accum].map(x);
      el("line", { x1: Math.min.apply(null, xs), x2: Math.max.apply(null, xs), y1: yc, y2: yc, "class": "zero", opacity: 0.25 }, svg);
      el("circle", { cx: x(v.learned), cy: yc, r: 7, "class": "open c-count" }, svg);
      el("circle", { cx: x(v.accum), cy: yc, r: 4.5, "class": "mk f-accum" }, svg);
      el("circle", { cx: x(v.count), cy: yc, r: 4.5, "class": "mk f-count" }, svg);
      if (note) note(svg, c, x, yc);
      var fm = function (z) { return signed ? sg(z, 4) : fx(z, 4); };
      rowHit(f, yc, rowH, c.label, [{ k: "count", v: fm(v.count), label: "count" }, { k: "count", v: fm(v.learned), label: "count, learned scale" },
                                    { k: "accum", v: fm(v.accum), label: "accumulator" }]);
    });
  }
  function renderScale() {
    var cells = scaleCells();
    if (!cells.length) return;
    dotRows("chart-scale-gap", cells, "gap", true);
    dotRows("chart-scale-acc", cells, "acc", false, function (svg, c, x, yc) {
      var cost = c.acc.count - c.acc.learned;
      if (c.label.indexOf("T = 6") < 0 || cost <= 0) return;
      txt("−" + fx(100 * cost, 1) + " points", { x: (x(c.acc.count) + x(c.acc.learned)) / 2, y: yc - 10, "text-anchor": "middle", "class": "t-strong" }, svg);
    });
    table("table-scale", ["cell", "gap: count", "learned scale", "accumulator", "accuracy: count", "learned scale", "accumulator"],
      cells.map(function (c) {
        return [c.label, sg(c.gap.count, 4), sg(c.gap.learned, 4), sg(c.gap.accum, 4), fx(c.acc.count, 4), fx(c.acc.learned, 4), fx(c.acc.accum, 4)];
      }));
  }

  /* ---------- 6. calibrators ---------- */
  function forest(hostId, runs, est, dom) {
    var host = document.getElementById(hostId);
    if (!host || !runs.length) return;
    host.textContent = "";
    var W = Math.max(240, host.clientWidth || 300), rowH = 16, top = 6, bottom = 30, left = 62;
    var H = top + rowH * runs.length + bottom;
    var svg = el("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": hostId }, host);
    var x = lin(dom[0], dom[1], left, W - 14), ticks = niceTicks(dom[0], dom[1], 4), dd = decimals(ticks);
    var g = el("g", { "class": "grid" }, svg);
    ticks.forEach(function (v) {
      el("line", { x1: x(v), x2: x(v), y1: top, y2: H - bottom }, g);
      txt(v === 0 ? "0" : sg(v, dd), { x: x(v), y: H - bottom + 16, "text-anchor": "middle" }, svg);
    });
    el("line", { x1: x(0), x2: x(0), y1: top, y2: H - bottom, "class": "zero" }, svg);
    var f = { svg: svg, W: W }, lastT = null;
    runs.forEach(function (r, i) {
      var e = r[est], yc = top + rowH * (i + 0.5);
      if (r.T !== lastT) { txt("T = " + r.T, { x: 0, y: yc + 4, "class": "t-ink" }, svg); lastT = r.T; }
      txt("s" + r.seed, { x: left - 8, y: yc + 4, "text-anchor": "end" }, svg);
      el("line", { x1: x(e.lo), x2: x(e.hi), y1: yc, y2: yc, "class": "whisk c-count" }, svg);
      el("circle", { cx: x(e.d), cy: yc, r: 4, "class": e.excl ? "mk f-count" : "open c-count" }, svg);
      rowHit(f, yc, rowH, "T = " + r.T + ", seed " + r.seed + (est === "ew" ? ", equal-width" : ", equal-mass"),
        [{ k: "count", v: sg(e.d, 4), label: "isotonic − temperature" },
         { k: "ink", v: "[" + sg(e.lo, 4) + ", " + sg(e.hi, 4) + "]", label: "95% interval" },
         { k: "ink", v: e.excl ? "excludes zero" : "includes zero", label: r.both ? "differs under both" : "not under both" }]);
    });
  }
  function calRuns() {
    var nm = [], seen = {};
    SWEEP.forEach(function (r) {
      var k = r.timesteps + "|" + r.seed; if (seen[k] || r.iso_minus_temp_ece_ew === "" || r.iso_minus_temp_ece_ew === undefined) return; seen[k] = 1;
      var ew = { d: +r.iso_minus_temp_ece_ew, lo: +r.iso_minus_temp_ci_low, hi: +r.iso_minus_temp_ci_high };
      var em = { d: +r.iso_minus_temp_ece_em, lo: +r.iso_minus_temp_em_ci_low, hi: +r.iso_minus_temp_em_ci_high };
      ew.excl = ew.lo > 0 || ew.hi < 0; em.excl = em.lo > 0 || em.hi < 0;
      nm.push({ T: r.timesteps, seed: r.seed, ew: ew, em: em, both: ew.excl && em.excl && (ew.d < 0) === (em.d < 0) });
    });
    var nc = where((D.caltech_calibrators || {}).paired, function (p) { return p.readout === "count"; }).map(function (p) {
      return { T: p.timesteps, seed: p.seed, both: p.differ,
        ew: { d: p.iso_minus_temp_ece_ew, lo: p.ew_ci[0], hi: p.ew_ci[1], excl: p.ew_excludes_zero },
        em: { d: p.iso_minus_temp_ece_em, lo: p.em_ci[0], hi: p.em_ci[1], excl: p.em_excludes_zero } };
    });
    var srt = function (a, b) { return a.T - b.T || a.seed - b.seed; };
    return { nm: nm.sort(srt), nc: nc.sort(srt) };
  }
  function domainOf(runs) {
    var v = [0]; runs.forEach(function (r) { ["ew", "em"].forEach(function (k) { v.push(r[k].lo, r[k].hi); }); });
    return pad(Math.min.apply(null, v), Math.max.apply(null, v), 0.06);
  }
  function renderCalibrators() {
    var R = calRuns();
    var dA = domainOf(R.nm), dB = domainOf(R.nc);
    forest("chart-cal-a-ew", R.nm, "ew", dA); forest("chart-cal-a-em", R.nm, "em", dA);
    forest("chart-cal-b-ew", R.nc, "ew", dB); forest("chart-cal-b-em", R.nc, "em", dB);
    var rows = [];
    [["N-MNIST", R.nm], ["N-Caltech101", R.nc]].forEach(function (g) {
      g[1].forEach(function (r) {
        rows.push([g[0], String(r.T), String(r.seed), sg(r.ew.d, 4), "[" + sg(r.ew.lo, 4) + ", " + sg(r.ew.hi, 4) + "]",
          sg(r.em.d, 4), "[" + sg(r.em.lo, 4) + ", " + sg(r.em.hi, 4) + "]", r.both ? "differ" : "not resolved"]);
      });
    });
    table("table-cal", ["dataset", "T", "seed", "equal-width", "interval", "equal-mass", "interval", "verdict"], rows);
  }

  /* ---------- 7. fusion pilot ---------- */
  function renderFusion(hostId, compact) {
    var host = document.getElementById(hostId || "chart-fusion");
    var fu = D.fusion;
    if (!host || !fu) return;
    var Ts = uniq(fu.rows.map(function (r) { return r.timesteps; })).sort(function (a, b) { return a - b; });
    var v = [0]; fu.rows.forEach(function (r) { v.push(r.delta_ci[0], r.delta_ci[1]); });
    var W = host.clientWidth || 300;
    var f = frame(host, { x: [Ts[0] - 0.6, Ts[Ts.length - 1] + 0.6], y: pad(Math.min.apply(null, v), Math.max.apply(null, v), 0.08), xTicks: Ts,
      xLabel: "timesteps T", ySigned: true, left: 58, right: compact ? 14 : 60, zero: true, zeroLabel: compact ? "" : "no gain",
      height: compact ? Math.min(200, W * 0.7) : Math.min(300, Math.max(220, W * 0.36)), aria: "fusion pilot: accuracy gained against T" });
    var meanLine = {};
    ["count", "accumulator"].forEach(function (ro, j) {
      var key = READOUT_KEY[ro], off = j ? 0.15 : -0.15;
      var rows = where(fu.rows, function (r) { return r.readout === ro; });
      var pts = uniq(rows.map(function (r) { return r.timesteps; })).sort(function (a, b) { return a - b; }).map(function (T) {
        return [T + off, mean(where(rows, function (r) { return r.timesteps === T; }).map(function (r) { return r.delta; }))];
      });
      meanLine[ro] = pts;
      path(f, pts, "c-" + key);
      rows.forEach(function (r) {
        var xv = r.timesteps + off + (r.seed - 1) * 0.06;
        el("line", { x1: f.x(xv), x2: f.x(xv), y1: f.y(r.delta_ci[0]), y2: f.y(r.delta_ci[1]), "class": "whisk c-" + key }, f.plot);
        dot(f, xv, r.delta, key);
      });
    });
    var cnt = where(fu.rows, function (r) { return r.readout === "count"; });
    Ts.forEach(function (T) {
      var r = where(cnt, function (q) { return q.timesteps === T; });
      if (r.length) diamond(f, T - 0.34, mean(r.map(function (q) { return q.predicted_on_s; })), "ink");
    });
    if (!compact) {
      var c0 = meanLine.count[0];
      txt("count", { x: f.x(c0[0]) + 10, y: f.y(c0[1]) - 6, "class": "t-ink" }, f.svg);
    }
    columns(f, Ts, function (T) {
      var rows = [];
      ["count", "accumulator"].forEach(function (ro) {
        var r = where(fu.rows, function (q) { return q.readout === ro && q.timesteps === T; });
        if (!r.length) return;
        var ex = where(r, function (q) { return q.delta_ci[0] > 0 || q.delta_ci[1] < 0; }).length;
        rows.push({ k: READOUT_KEY[ro], v: sg(mean(r.map(function (q) { return q.delta; })), 4), label: READOUT_NAME[ro] + ", " + ex + " of " + r.length + " seeds exclude zero" });
      });
      var c = where(cnt, function (q) { return q.timesteps === T; });
      if (c.length) rows.push({ k: "ink", v: sg(mean(c.map(function (q) { return q.predicted_on_s; })), 4), label: "predicted, saturated pairs" });
      return rows;
    }, function (T) { return "Fusion pilot, T = " + T; });
    if (compact) return;
    var trows = [];
    ["count", "accumulator"].forEach(function (ro) {
      Ts.forEach(function (T) {
        var r = where(fu.rows, function (q) { return q.readout === ro && q.timesteps === T; });
        if (!r.length) return;
        trows.push([READOUT_NAME[ro], String(T), sg(mean(r.map(function (q) { return q.delta; })), 4),
          String(where(r, function (q) { return q.delta_ci[0] > 0 || q.delta_ci[1] < 0; }).length) + " of " + r.length,
          ro === "count" ? fx(mean(r.map(function (q) { return q.p_s; })), 3) : "–",
          ro === "count" ? sg(mean(r.map(function (q) { return q.predicted_on_s; })), 4) : "–"]);
      });
    });
    table("table-fusion", ["readout", "T", "gain δ", "seeds excluding zero", "P(both saturate)", "predicted"], trows);
  }

  /* ---------- 8. ledger ---------- */
  function renderLedger() {
    var t = document.getElementById("table-ledger"), L = D.ledger;
    if (!t || !L) return;
    t.textContent = "";
    var thead = document.createElement("thead"), hr = document.createElement("tr");
    [["", ""], ["recorded", ""], ["prediction", "t"], ["measured", "t"], ["outcome", ""]].forEach(function (h) {
      var th = document.createElement("th"); th.textContent = h[0]; if (h[1]) th.className = h[1]; hr.appendChild(th);
    });
    thead.appendChild(hr); t.appendChild(thead);
    var tb = document.createElement("tbody");
    [["paper", "The paper's predictions: " + L.paper.held + " held, " + L.paper.falsified + " falsified"],
     ["early", "Earlier predictions: " + L.early.held + " held, " + L.early.falsified + " falsified"]].forEach(function (g) {
      var gr = document.createElement("tr"); gr.className = "group";
      var gc = document.createElement("td"); gc.colSpan = 5; gc.textContent = g[1]; gr.appendChild(gc); tb.appendChild(gr);
      L.rows.filter(function (r) { return r.group === g[0]; }).forEach(function (r) {
        var tr = document.createElement("tr");
        [[r.id, "id"], [r.recorded, "when"], [r.statement, "t"], [r.measured, "t"]].forEach(function (c) {
          var td = document.createElement("td"); td.textContent = c[0]; td.className = c[1]; tr.appendChild(td);
        });
        var td = document.createElement("td"), p = document.createElement("span");
        p.className = "pill " + (r.held ? "held" : "falsified"); p.textContent = r.held ? "held" : "falsified";
        td.appendChild(p); tr.appendChild(td); tb.appendChild(tr);
      });
    });
    t.appendChild(tb);
  }

  /* ---------- briefing: the three small charts ---------- */
  function renderBriefing() {
    var cc = D.caltech && D.caltech[0];
    if (document.getElementById("mini-cap")) capPanel("mini-cap", "mini-cap-title", D.readout, D.n_classes, "N-MNIST");
    if (document.getElementById("mini-auroc")) renderAuroc("mini-auroc");
    if (document.getElementById("mini-fusion")) renderFusion("mini-fusion", true);
    void cc;
  }

  function renderUpdated() {
    var e = document.getElementById("last-updated");
    if (!e || !D.last_updated) return;
    e.textContent = "Last updated " + D.last_updated + ", the newest of " + D.n_source_artifacts +
      " result artifacts (" + D.last_updated_source + ").";
  }

  /* ---------- run: each renderer isolated ---------- */
  var CHARTS = [renderCap, renderCalibration, renderGap, renderScale, renderCalibrators, function fusion() { renderFusion(); }, renderBriefing];
  [renderNumbers, renderLedger, renderUpdated].concat(CHARTS).forEach(function (fn) {
    try { fn(); } catch (err) { console.error("spikecalib: " + (fn.name || "renderer") + " failed", err); }
  });
  var lastW = window.innerWidth, timer = null;
  window.addEventListener("resize", function () {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth;
    clearTimeout(timer);
    timer = setTimeout(function () {
      hideTip();
      CHARTS.forEach(function (fn) { try { fn(); } catch (err) { console.error("spikecalib: resize", err); } });
    }, 150);
  });
})();
