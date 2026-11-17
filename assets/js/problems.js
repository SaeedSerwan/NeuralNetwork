/* Part 4 · Training problems: imbalance, vanishing and exploding gradients
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$;
  const pct = (v) => (v * 100).toFixed(1) + '%';

  /* ------------------------------------------------------------ 01 imbalance */
  (function () {
    const r = UL.rng(2024), N = 10000, data = [];
    for (let i = 0; i < N; i++) { const fraud = i % 50 === 0; data.push({ x: fraud ? [1.5 + r.normal() * 0.95, 1.2 + r.normal() * 0.95] : [r.normal(), r.normal()], y: fraud ? 1 : 0 }); }
    for (let i = N - 1; i > 0; i--) { const j = r.int(i + 1); [data[i], data[j]] = [data[j], data[i]]; }
    const train = data.slice(0, 7000), test = data.slice(7000);
    const nF = test.filter((d) => d.y).length;
    function fit(set, wPos = 1) { // logistic regression by full-batch gradient descent
      let w = [0, 0], b = 0; const n = set.length;
      for (let it = 0; it < 500; it++) {
        let g0 = 0, g1 = 0, gb = 0, ws = 0;
        set.forEach((d) => { const p = UL.sigmoid(w[0] * d.x[0] + w[1] * d.x[1] + b), k = d.y ? wPos : 1, e = (p - d.y) * k; g0 += e * d.x[0]; g1 += e * d.x[1]; gb += e; ws += k; });
        w = [w[0] - (1.5 * g0) / ws, w[1] - (1.5 * g1) / ws]; b -= (1.5 * gb) / ws;
      }
      return { w, b };
    }
    function smote(set, k = 3) {
      const fr = set.filter((d) => d.y), out = set.slice(), need = set.length - 2 * fr.length, rr = UL.rng(7);
      const nbrs = fr.map((a) => fr.filter((b) => b !== a).map((b) => [Math.hypot(a.x[0] - b.x[0], a.x[1] - b.x[1]), b]).sort((p, q) => p[0] - q[0]).slice(0, k).map((p) => p[1]));
      for (let i = 0; i < need; i++) { const j = rr.int(fr.length), a = fr[j], b = nbrs[j][rr.int(k)], t = rr(); out.push({ x: [a.x[0] + t * (b.x[0] - a.x[0]), a.x[1] + t * (b.x[1] - a.x[1])], y: 1 }); }
      return out;
    }
    const under = () => { const fr = train.filter((d) => d.y), lg = train.filter((d) => !d.y).slice(0, fr.length * 4); return fr.concat(lg); };
    const nTrF = train.filter((d) => d.y).length;
    const M = { always: { w: [0, 0], b: -20 }, plain: fit(train), weights: fit(train, (train.length - nTrF) / nTrF), smote: fit(smote(train)), under: fit(under()) };
    const evalM = (m, t) => { let tp = 0, fp = 0, fn = 0, tn = 0; test.forEach((d) => { const p = UL.sigmoid(m.w[0] * d.x[0] + m.w[1] * d.x[1] + m.b) >= t ? 1 : 0; if (p && d.y) tp++; else if (p) fp++; else if (d.y) fn++; else tn++; }); const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp / (tp + fn); return { tp, fp, fn, tn, acc: (tp + tn) / test.length, rec, prec, f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0 }; };
    const NAMES = { always: 'Always predict “legitimate”', plain: 'Trained on the data as it is', weights: '+ class weights', smote: '+ SMOTE (training set only)', under: '+ undersampling (4 : 1)' };
    $('#im-table').innerHTML = `<thead><tr><th>Logistic regression, test set, threshold 0.5</th><th>Accuracy</th><th>Recall</th><th>Precision</th><th>Frauds caught</th></tr></thead><tbody>${Object.keys(M).map((k) => { const e = evalM(M[k], 0.5); return `<tr><td style="font-family:var(--font)">${NAMES[k]}</td><td>${pct(e.acc)}</td><td>${Math.round(e.rec * 100)}%</td><td>${e.tp + e.fp ? Math.round(e.prec * 100) + '%' : '—'}</td><td>${e.tp} of ${nF}</td></tr>`; }).join('')}</tbody>`;
    const P = new UL.Plot('#im-plot', { w: 520, h: 420, pad: { l: 30, r: 10, t: 10, b: 28 } }); P.domain(-3.5, 4.5, -3.5, 4.5, true);
    let s = 'plain';
    const thr = UL.range('im-t', () => draw(), (v) => v.toFixed(2));
    function draw() {
      const m = M[s], t = thr.v, e = evalM(m, t), L = Math.log(t / (1 - t));
      P.clear(); P.regions((x, y) => (UL.sigmoid(m.w[0] * x + m.w[1] * y + m.b) >= t ? 7 : null), 6, 0.1); P.axes({ xlabel: 'amount (scaled)' });
      test.forEach((d, i) => { if (!d.y && i % 3 === 0) P.marker(d.x[0], d.x[1], { r: 1.8, fill: '#b9b9b4', ring: false }); });
      test.forEach((d) => { if (d.y) P.marker(d.x[0], d.x[1], { shape: 'cross', r: 4, fill: '#d64045', lw: 2 }); });
      if (Math.abs(m.w[1]) > 1e-6) P.clipped(() => P.line(-3.5, (L - m.b - m.w[0] * -3.5) / m.w[1], 4.5, (L - m.b - m.w[0] * 4.5) / m.w[1], { color: UL.INK, w: 2 }));
      $('#im-conf').innerHTML = `<tr><th></th><th>said fraud</th><th>said legitimate</th></tr><tr><th>real fraud</th><td class="good">TP ${e.tp}</td><td class="bad">FN ${e.fn}</td></tr><tr><th>real legitimate</th><td class="bad">FP ${e.fp}</td><td class="good">TN ${e.tn}</td></tr>`;
      UL.stats('im-stats', [{ k: 'Accuracy', v: pct(e.acc) }, { k: 'Recall', v: Math.round(e.rec * 100) + '%', hi: 1 }, { k: 'Precision', v: e.tp + e.fp ? Math.round(e.prec * 100) + '%' : '—' }, { k: 'F1', v: e.f1.toFixed(2) }]);
      $('#im-foot').innerHTML = `${NAMES[s]}: caught <b>${e.tp} of ${nF}</b> frauds, with <b>${e.fp}</b> false alarms among ${test.length - nF} legitimate payments. ${s === 'plain' && t >= 0.4 ? 'Lower the threshold: the same model catches far more fraud.' : ''}${s === 'always' ? 'High accuracy, zero recall.' : ''}`;
    }
    UL.seg('im-s', (v) => { s = v; draw(); }); draw();
  })();

  /* ------------------------------------------------------------ 02 SMOTE */
  (function () {
    const P = new UL.Plot('#sm-plot', { w: 640, h: 340, pad: { l: 10, r: 10, t: 10, b: 10 } }); P.domain(-3, 4, -2.5, 3.5, true);
    let r, maj, mino, syn, lines;
    const reset = () => { r = UL.rng(31); maj = Array.from({ length: 160 }, () => [r.normal() * 0.9 - 0.4, r.normal() * 0.9]); mino = Array.from({ length: 9 }, () => [1.9 + r.normal() * 0.7, 1.4 + r.normal() * 0.7]); syn = []; lines = []; draw(); };
    function add() {
      lines = [];
      for (let k = 0; k < 5; k++) {
        const j = r.int(mino.length), a = mino[j], nb = mino.filter((b) => b !== a).sort((p, q) => Math.hypot(p[0] - a[0], p[1] - a[1]) - Math.hypot(q[0] - a[0], q[1] - a[1])).slice(0, 3), b = nb[r.int(3)], t = r();
        syn.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]); lines.push([a, b]);
      }
      draw();
    }
    function draw() {
      P.clear(); maj.forEach((p) => P.marker(p[0], p[1], { r: 3, fill: '#b9b9b4', ring: false }));
      lines.forEach(([a, b]) => P.line(a[0], a[1], b[0], b[1], { color: UL.COLORS[1], dash: [4, 3], w: 1.4 }));
      syn.forEach((p, i) => P.marker(p[0], p[1], { shape: 'diamond', r: 4.5, fill: i >= syn.length - 5 ? UL.COLORS[1] : '#f2b08f' }));
      mino.forEach((p) => P.marker(p[0], p[1], { shape: 'cross', r: 5, fill: '#d64045', lw: 2.6 }));
      $('#sm-status').innerHTML = `real frauds <b>${mino.length}</b> · synthetic <b>${syn.length}</b> · legitimate <b>${maj.length}</b>`;
    }
    $('#sm-add').addEventListener('click', add); $('#sm-reset').addEventListener('click', reset); reset();
  })();

  /* ------------------------------------------------------------ 03 vanishing gradients */
  (function () {
    const P = new UL.Plot('#vg-plot', { w: 560, h: 320, pad: { l: 50, r: 10, t: 14, b: 32 } }), calc = $('#vg-calc');
    let act = 'sigmoid';
    const depth = UL.range('vg-depth', () => run()), init = $('#vg-init'), res = $('#vg-res');
    function run() {
      const L = depth.v, H = 32, B = 64, r = UL.rng(3), A = UL.ACT[act];
      const sc = { small: 0.1, xavier: 1 / Math.sqrt(H), he: Math.sqrt(2 / H), big: 3 / Math.sqrt(H) }[init.value];
      const W = Array.from({ length: L }, () => Float64Array.from({ length: H * H }, () => r.normal() * sc));
      const X = Array.from({ length: B }, () => Float64Array.from({ length: H }, () => r.normal())), Y = Array.from({ length: B }, () => Float64Array.from({ length: H }, () => r.normal() * 0.5));
      const g = new Array(L).fill(0);
      X.forEach((x, n) => {
        const hs = [x], zs = [];
        for (let l = 0; l < L; l++) { const h = hs[l], z = new Float64Array(H), a = new Float64Array(H); for (let j = 0; j < H; j++) { let s = 0; for (let i = 0; i < H; i++) s += W[l][j * H + i] * h[i]; z[j] = s; a[j] = A.f(s) + (res.checked ? h[j] : 0); } zs.push(z); hs.push(a); }
        let d = Float64Array.from(hs[L], (v, j) => v - Y[n][j]);
        for (let l = L - 1; l >= 0; l--) {
          const dz = Float64Array.from(d, (v, j) => v * A.d(zs[l][j], A.f(zs[l][j]))); let s = 0;
          for (let j = 0; j < H; j++) for (let i = 0; i < H; i++) s += Math.abs(dz[j] * hs[l][i]);
          g[l] += s / (H * H * B);
          const nd = new Float64Array(H); for (let i = 0; i < H; i++) { let t = 0; for (let j = 0; j < H; j++) t += W[l][j * H + i] * dz[j]; nd[i] = t + (res.checked ? d[i] : 0); } d = nd;
        }
      });
      const lg = g.map((v) => Math.log10(Math.max(1e-30, v))), lo = Math.floor(Math.min(...lg, -1)), hi = Math.ceil(Math.max(...lg, 0));
      P.clear(); P.domain(0.4, L + 0.6, lo, hi); P.axes({ xlabel: 'layer (1 = first, next to the input)', fy: (v) => (Number.isInteger(v) ? '1e' + v : ''), nx: Math.min(L, 10) });
      const c = P.ctx, bw = Math.max(2, (P.sx(1) - P.sx(0)) * 0.7);
      lg.forEach((v, l) => { c.fillStyle = l === 0 ? UL.COLORS[1] : UL.COLORS[0]; const X = P.sx(l + 1), Y0 = P.sy(lo), Y1 = P.sy(v); c.fillRect(X - bw / 2, Y1, bw, Y0 - Y1); });
      const ratio = g[L - 1] / g[0];
      calc.innerHTML = `<span class="k">Last layer</span><span>${g[L - 1].toExponential(1)}</span><span class="k">First layer</span><span class="hl">${g[0].toExponential(1)}</span><span class="k">Last ÷ first</span><span class="hl" style="font-size:1.05rem">${ratio > 1 ? ratio > 1e6 ? '×' + ratio.toExponential(0) : '×' + Math.round(ratio).toLocaleString('en-US') : '×' + ratio.toFixed(2)}</span>` +
        `<span class="k">Reading</span><span style="font-family:var(--font)">${ratio > 1000 ? 'The first layers get almost no gradient: they will barely learn. <b>Vanishing.</b>' : ratio < 0.01 ? 'The first layers get a huge gradient. <b>Exploding.</b>' : 'The gradient reaches the first layers at a healthy size.'}</span>`;
    }
    UL.seg('vg-act', (v) => { act = v; run(); }); init.addEventListener('change', run); res.addEventListener('change', run); run();
  })();

  /* ------------------------------------------------------------ 04 exploding product */
  (function () {
    const P = new UL.Plot('#ex-plot', { w: 560, h: 300, pad: { l: 50, r: 10, t: 12, b: 32 } }), calc = $('#ex-calc');
    const f = UL.range('ex-f', () => draw(), (v) => v.toFixed(2));
    function draw() {
      const v = f.v, pts = []; for (let t = 0; t <= 100; t++) pts.push([t, t * Math.log10(v)]);
      const lo = Math.min(-2, Math.floor(100 * Math.log10(Math.min(v, 1)))), hi = Math.max(2, Math.ceil(100 * Math.log10(Math.max(v, 1))));
      P.clear(); P.domain(0, 100, lo, hi); P.axes({ xlabel: 'steps back in time (or layers)', fy: (y) => (Number.isInteger(y) ? '1e' + y : '') });
      P.line(0, 0, 100, 0, { color: UL.FAINT, dash: [4, 4], w: 1 }); P.clipped(() => P.poly(pts, { color: v > 1 ? UL.COLORS[7] : v < 1 ? UL.COLORS[0] : UL.INK, w: 2.6 }));
      const fin = v ** 100;
      calc.innerHTML = `<span class="k">Factor</span><span>${v.toFixed(2)}</span><span class="k">After 10</span><span>${(v ** 10).toPrecision(3)}</span><span class="k">After 50</span><span>${(v ** 50).toPrecision(3)}</span><span class="k">After 100</span><span class="hl">${fin.toPrecision(3)}</span><span class="k">Reading</span><span style="font-family:var(--font)">${v > 1.005 ? '<b class="neg">Exploding</b>: the gradient becomes enormous, the step throws the weights far away, the loss becomes NaN.' : v < 0.995 ? '<b class="pos">Vanishing</b>: almost nothing reaches the early steps.' : 'Exactly 1: the only stable case, and impossible to keep by luck.'}</span>`;
    }
    draw();
  })();

  /* ------------------------------------------------------------ 04 the cliff */
  (function () {
    const C = (w) => 0.3 * (w - 3) ** 2 + 8 * UL.sigmoid(-20 * (w - 0.5)), dC = (w) => { const s = UL.sigmoid(-20 * (w - 0.5)); return 0.6 * (w - 3) - 160 * s * (1 - s); };
    const P = new UL.Plot('#cl-plot', { w: 460, h: 280, pad: { l: 36, r: 10, t: 12, b: 32 } }), LP = new UL.Plot('#cl-loss', { w: 460, h: 280, pad: { l: 44, r: 10, t: 12, b: 32 } }), on = $('#cl-on'), st = $('#cl-status');
    let w, path, losses, n;
    const reset = () => { w = -0.8; path = [w]; losses = [C(w)]; n = 0; };
    function draw() {
      P.clear(); P.domain(-1.2, 11, 0, 18); P.axes({ xlabel: 'weight w' });
      const pts = []; for (let x = -1.2; x <= 11; x += 0.005) pts.push([x, C(x)]); P.clipped(() => P.poly(pts, { color: UL.INK, w: 2 }));
      P.clipped(() => P.poly(path.map((p) => [p, C(p)]), { color: UL.COLORS[0], w: 1, alpha: 0.5 })); P.clipped(() => path.forEach((p, i) => P.marker(p, C(p), { r: i === path.length - 1 ? 6 : 2.5, fill: UL.COLORS[0], alpha: i === path.length - 1 ? 1 : 0.4, ring: i === path.length - 1 })));
      const last = path[path.length - 1]; if (last > 11) P.textPx('the ball flew off to w = ' + last.toFixed(0) + ' →', P.W - 14, 30, { align: 'right', color: '#d64045', font: '12px "IBM Plex Mono", monospace' });
      UL.chart(LP, [{ data: losses, color: UL.COLORS[7], w: 2 }], { xlabel: 'step', ylabel: 'cost', x0: 0, ymin: 0, xmax: 40, log: false, ymax: Math.min(300, Math.max(10, ...losses)) });
      st.innerHTML = `step <b>${n}</b> · w = <b>${last.toFixed(2)}</b> · cost <b>${C(last).toFixed(2)}</b>${on.checked ? ' · clipped' : ''}`;
    }
    const step = () => { let g = dC(w); if (on.checked && Math.abs(g) > 1) g = Math.sign(g); w -= 0.3 * g; path.push(w); losses.push(C(w)); n++; };
    let t = 0; const loop = UL.loop((dt) => { t += dt; if (t > 0.18) { t = 0; step(); draw(); if (n >= 40) { $('#cl-run').textContent = 'Run'; return false; } } });
    $('#cl-run').addEventListener('click', () => { loop.stop(); reset(); draw(); loop.start(); $('#cl-run').textContent = 'Running…'; });
    on.addEventListener('change', () => { loop.stop(); $('#cl-run').textContent = 'Run'; reset(); draw(); });
    reset(); draw();
  })();

  /* ------------------------------------------------------------ 05 exercise 4 */
  UL.exercises('ex4-list', [
    { q: 'A fraud model reports 99.5% accuracy, but catches only 2 of the 50 frauds in the test set.', a: '<b>Data imbalance.</b> Cause: “legitimate” is almost always right, so the cost function barely notices the frauds. Fixes: class weights or SMOTE; judge by recall / F1; lower the threshold.' },
    { q: 'A 30-layer network with sigmoid activations trains for hours; the loss stays almost flat and the first layers hardly change.', a: '<b>Vanishing gradient.</b> Cause: slopes ≤ 0.25 multiplied over 30 layers. Fixes: ReLU; He initialisation, batch normalisation, residual connections. Try it in § 03 with 30 layers.' },
    { q: 'An RNN on long sensor sequences trains well for a few hundred steps, then the loss suddenly becomes NaN.', a: '<b>Exploding gradient.</b> Cause: factors &gt; 1 multiplied over many time steps. Fixes: gradient clipping; a smaller learning rate; LSTM / GRU.' },
  ]);

  UL.quiz('qz', [
    { q: 'Only 1% of patients have a disease. A model always says “healthy”. Its accuracy is…', o: ['1%', '50%', '99%', 'impossible to say'], a: 2, why: 'It is right for the 99% who are healthy, and catches none of the sick: recall 0%.' },
    { q: 'Which metric answers “of all real frauds, how many did we catch”?', o: ['Accuracy', 'Precision', 'Recall', 'Cost'], a: 2, why: 'Recall = TP / (TP + FN).' },
    { q: 'Where may you apply SMOTE?', o: ['The test set only', 'The training set only', 'Both', 'Neither'], a: 1, why: 'The test set must stay like real data, or your results are fiction.' },
    { q: 'Why do deep sigmoid networks suffer from vanishing gradients?', o: ['Sigmoid is slow to compute', 'Its slope is at most 0.25, and backpropagation multiplies one slope per layer', 'It has no bias', 'It outputs negative numbers'], a: 1, why: 'Ten factors of at most 0.25 give at most 0.25¹⁰ ≈ 0.000001.' },
    { q: 'What does gradient clipping do?', o: ['Removes small gradients', 'Shrinks a gradient that is too long to a maximum length, keeping its direction', 'Stops training', 'Deletes layers'], a: 1, why: 'Steps can no longer be huge, so one bad batch cannot throw the weights to NaN.' },
    { q: 'Which change helps against vanishing gradients?', o: ['More sigmoid layers', 'Residual connections', 'A larger test set', 'A higher threshold'], a: 1, why: 'Shortcuts let the gradient pass back without being multiplied by every layer. Tick “residual connections” in § 03.' },
  ]);
  UL.start();
})();
