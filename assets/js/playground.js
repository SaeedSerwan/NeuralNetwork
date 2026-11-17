/* Neural network playground: a small network learning 2-D data in the browser
   Inspired by the TensorFlow Playground (Smilkov & Carter). Written from scratch for this lab.
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$;
  const BLUE = [42, 120, 214], ORANGE = [235, 104, 52];
  const colOf = (t) => { // t in −1 … 1 → orange · white · blue
    const k = Math.min(1, Math.abs(t)) ** 0.9, to = t >= 0 ? BLUE : ORANGE;
    return [0, 1, 2].map((i) => Math.round(255 + (to[i] - 255) * k));
  };
  const FEATS = [
    ['x1', 'x₁', (x) => x[0]], ['x2', 'x₂', (x) => x[1]], ['x1sq', 'x₁²', (x) => x[0] * x[0]], ['x2sq', 'x₂²', (x) => x[1] * x[1]],
    ['x1x2', 'x₁x₂', (x) => x[0] * x[1]], ['sin1', 'sin x₁', (x) => Math.sin(3 * x[0])], ['sin2', 'sin x₂', (x) => Math.sin(3 * x[1])],
  ];
  const st = { data: 'circle', noise: 0.1, split: 0.5, feats: new Set(['x1', 'x2']), hidden: [4, 2], act: 'tanh', lr: 0.1 };
  let pts = [], train = [], test = [], net, epoch = 0, hist = [], hover = null, rngData = UL.rng(1);

  /* ------------------------------------------------------------ data (coordinates in −1 … 1) */
  function makeData() {
    const r = UL.rng(Math.floor(Math.random() * 1e6)), N = 300, n = st.noise, out = [];
    const add = (x, y, c) => out.push({ x: [x + r.normal() * n * 0.35, y + r.normal() * n * 0.35], c });
    for (let i = 0; i < N; i++) {
      if (st.data === 'circle') { const inside = i % 2 === 0, rad = inside ? r() * 0.45 : 0.62 + r() * 0.3, a = r() * 2 * Math.PI; add(rad * Math.cos(a), rad * Math.sin(a), inside ? 1 : 0); }
      else if (st.data === 'xor') { let x = r() * 1.8 - 0.9, y = r() * 1.8 - 0.9; x += x > 0 ? 0.05 : -0.05; y += y > 0 ? 0.05 : -0.05; add(x, y, x * y > 0 ? 1 : 0); }
      else if (st.data === 'gauss') { const c = i % 2; add((c ? 0.45 : -0.45) + r.normal() * 0.18, (c ? 0.4 : -0.4) + r.normal() * 0.18, c); }
      else if (st.data === 'moons') { const c = i % 2, a = r() * Math.PI; add(c ? 0.5 - Math.cos(a) * 0.8 - 0.25 : Math.cos(a) * 0.8 - 0.25, c ? 0.25 - Math.sin(a) * 0.6 + 0.05 : Math.sin(a) * 0.6 - 0.2, c); }
      else { const c = i % 2, t = (i / N) * 2.2 + 0.15, a = t * Math.PI * 1.25 + (c ? Math.PI : 0), rad = t * 0.42; add(rad * Math.cos(a), rad * Math.sin(a), c); }
    }
    pts = out; const k = Math.round(N * st.split); const idx = pts.map((_, i) => i).sort(() => Math.random() - 0.5);
    train = idx.slice(0, k).map((i) => pts[i]); test = idx.slice(k).map((i) => pts[i]);
  }
  const feats = () => FEATS.filter((f) => st.feats.has(f[0]));
  const input = (x) => feats().map((f) => f[2](x));

  /* ------------------------------------------------------------ the network */
  function build() {
    const F = feats(); if (!F.length) { st.feats.add('x1'); }
    net = new UL.MLP([feats().length, ...st.hidden, 1], { act: st.act, out: 'sigmoid', loss: 'bce', seed: Math.floor(Math.random() * 1e6), scale: st.act === 'relu' ? undefined : undefined });
    epoch = 0; hist = []; record(); draw();
  }
  const costOf = (set) => { let c = 0; set.forEach((p) => { net.forward(input(p.x)); c += net.cost([p.c]); }); return set.length ? c / set.length : 0; };
  const record = () => hist.push([epoch, costOf(train), costOf(test)]);
  function trainEpoch() {
    const order = train.map((_, i) => i).sort(() => Math.random() - 0.5), B = 10;
    for (let k = 0; k < order.length; k += B) { const b = order.slice(k, k + B); b.forEach((i) => { net.forward(input(train[i].x)); net.backward([train[i].c]); }); net.step(st.lr, b.length); }
    epoch++; record(); if (hist.length > 600) hist = hist.filter((_, i) => i % 2 === 0 || i === hist.length - 1);
  }

  /* ------------------------------------------------------------ drawing */
  const G = 36; // grid resolution of every neuron picture
  const NP = new UL.Plot('#pg-net', { w: 680, h: 330, pad: { l: 0, r: 0, t: 0, b: 0 } });
  const OP = new UL.Plot('#pg-out', { w: 380, h: 380, pad: { l: 0, r: 0, t: 0, b: 0 } }); OP.domain(-1.15, 1.15, -1.15, 1.15);
  const LP = new UL.Plot('#pg-loss', { w: 680, h: 130, pad: { l: 40, r: 10, t: 8, b: 26 } });
  const off = document.createElement('canvas'); off.width = G; off.height = G; const octx = off.getContext('2d'); const img = octx.createImageData(G, G);
  let layout = [];
  function maps() { // activation of every neuron at every grid point
    const L = net.sizes, M = L.map((n) => Array.from({ length: n }, () => new Float32Array(G * G)));
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
      const x = [-1.15 + (2.3 * (gx + 0.5)) / G, 1.15 - (2.3 * (gy + 0.5)) / G]; net.forward(input(x));
      for (let l = 0; l < L.length; l++) for (let j = 0; j < L[l]; j++) M[l][j][gy * G + gx] = net.a[l][j];
    }
    return M;
  }
  function paint(ctx, arr, x, y, w, h, kind, soft) {
    let mx = 1e-9; if (kind === 'hidden') for (let i = 0; i < arr.length; i++) mx = Math.max(mx, Math.abs(arr[i]));
    for (let i = 0; i < arr.length; i++) {
      let t = kind === 'out' ? (arr[i] - 0.5) * 2 * (soft ? 0.6 : 1) : kind === 'feat' ? arr[i] : (st.act === 'sigmoid' ? (arr[i] - 0.5) * 2 : arr[i] / mx);
      if (kind === 'feat') t = Math.max(-1, Math.min(1, t)); const c = colOf(t);
      img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, x, y, w, h);
  }
  function draw() {
    const M = maps(), L = net.sizes, c = NP.ctx; NP.clear();
    const cols = L.length, S = 38, xs = Array.from({ length: cols }, (_, i) => 30 + (i * (NP.W - 100)) / (cols - 1));
    layout = L.map((n, l) => { const gap = Math.min(S + 12, (NP.H - 40) / Math.max(1, n)); const y0 = 24 + (NP.H - 24 - n * gap) / 2; return Array.from({ length: n }, (_, j) => [xs[l], y0 + j * gap + (gap - S) / 2]); });
    // edges
    for (let l = 0; l < cols - 1; l++) {
      const W = net.W[l], nIn = L[l]; let mx = 1e-9; W.forEach((w) => { mx = Math.max(mx, Math.abs(w)); });
      for (let j = 0; j < L[l + 1]; j++) for (let i = 0; i < nIn; i++) {
        const w = W[j * nIn + i], [x1, y1] = layout[l][i], [x2, y2] = layout[l + 1][j];
        c.strokeStyle = w >= 0 ? `rgba(42,120,214,${0.25 + 0.6 * Math.abs(w) / mx})` : `rgba(235,104,52,${0.25 + 0.6 * Math.abs(w) / mx})`; c.lineWidth = 0.5 + 4 * Math.abs(w) / mx;
        c.beginPath(); c.moveTo(x1 + S, y1 + S / 2); c.bezierCurveTo(x1 + S + 30, y1 + S / 2, x2 - 30, y2 + S / 2, x2, y2 + S / 2); c.stroke();
      }
    }
    // neurons
    const F = feats();
    layout.forEach((col, l) => col.forEach(([x, y], j) => {
      paint(c, M[l][j], x, y, S, S, l === 0 ? 'feat' : l === cols - 1 ? 'out' : 'hidden');
      c.lineWidth = hover && hover[0] === l && hover[1] === j ? 2.5 : 1; c.strokeStyle = hover && hover[0] === l && hover[1] === j ? '#111' : '#c9c9c4'; c.strokeRect(x + 0.5, y + 0.5, S - 1, S - 1);
      if (l === 0) { c.font = '11px "IBM Plex Mono", monospace'; c.fillStyle = '#3d3d3b'; c.textAlign = 'right'; c.fillText(F[j][1], x - 5, y + S / 2 + 4); }
    }));
    c.font = '500 11px "IBM Plex Sans", sans-serif'; c.fillStyle = '#6b6b68'; c.textAlign = 'center';
    xs.forEach((x, l) => c.fillText(l === 0 ? 'inputs' : l === cols - 1 ? 'output' : `layer ${l} · ${L[l]}`, x + S / 2, 14));
    // big picture
    const show = hover ? M[hover[0]][hover[1]] : M[cols - 1][0], kind = !hover ? 'out' : hover[0] === 0 ? 'feat' : hover[0] === cols - 1 ? 'out' : 'hidden';
    OP.clear(); paint(OP.ctx, show, 0, 0, OP.W, OP.H, kind, true);
    if (!hover || hover[0] === cols - 1) { // the boundary at 0.5
      const o = M[cols - 1][0], cs = OP.W / G; OP.ctx.fillStyle = 'rgba(17,17,17,.75)';
      for (let gy = 0; gy < G - 1; gy++) for (let gx = 0; gx < G - 1; gx++) { const a = o[gy * G + gx] > 0.5, b = o[gy * G + gx + 1] > 0.5, d = o[(gy + 1) * G + gx] > 0.5; if (a !== b || a !== d) OP.ctx.fillRect((gx + 1) * cs - 1, (gy + 1) * cs - 1, 2, 2); }
    }
    const P = $('#pg-showtest').checked ? test : train;
    P.forEach((p) => OP.marker(p.x[0], p.x[1], { r: 3.8, fill: p.c ? '#1a4f96' : '#b4410f', ringColor: '#fff', ringW: 1.2, shape: $('#pg-showtest').checked ? 'square' : 'circle' }));
    $('#pg-cap').innerHTML = hover ? (hover[0] === 0 ? `<b>Input ${F[hover[1]][1]}</b>: the raw input, blue = high, orange = low` : hover[0] === cols - 1 ? '<b>Output neuron</b>: blue = class 1, orange = class 0' : `<b>Layer ${hover[0]}, neuron ${hover[1] + 1}</b>: what this neuron computes, blue = high, orange = low`) : '<b>Output neuron</b>: blue = class 1, orange = class 0; black dots mark the decision boundary';
    // loss
    LP.clear(); const ymax = Math.max(0.1, ...hist.map((h) => Math.min(1.5, Math.max(h[1], h[2])))) * 1.05; LP.domain(0, Math.max(10, epoch), 0, ymax); LP.axes({ ny: 3, nx: 6 });
    LP.clipped(() => { LP.poly(hist.map((h) => [h[0], h[1]]), { color: UL.INK, w: 1.8 }); LP.poly(hist.map((h) => [h[0], h[2]]), { color: UL.COLORS[1], w: 1.8 }); });
    const last = hist[hist.length - 1];
    const acc = (set) => set.length ? set.filter((p) => (net.forward(input(p.x))[0] > 0.5 ? 1 : 0) === p.c).length / set.length : 0;
    $('#pg-status').innerHTML = `epoch <b>${epoch}</b> · training cost <b>${last[1].toFixed(3)}</b> · test cost <b>${last[2].toFixed(3)}</b> · test accuracy <b>${Math.round(acc(test) * 100)}%</b> · ${net.nParams} parameters`;
  }
  NP.cv.addEventListener('mousemove', (e) => {
    const [X, Y] = NP.pointer(e); let h = null; layout.forEach((col, l) => col.forEach(([x, y], j) => { if (X >= x && X <= x + 38 && Y >= y && Y <= y + 38) h = [l, j]; }));
    if (JSON.stringify(h) !== JSON.stringify(hover)) { hover = h; if (!loop.on) draw(); }
  });
  NP.cv.addEventListener('mouseleave', () => { hover = null; if (!loop.on) draw(); });

  /* ------------------------------------------------------------ controls */
  const loop = UL.loop(() => { trainEpoch(); draw(); if (epoch >= 3000) { $('#pg-play').textContent = 'Play'; return false; } });
  const stop = () => { loop.stop(); $('#pg-play').textContent = 'Play'; };
  $('#pg-play').addEventListener('click', () => { if (loop.on) stop(); else { loop.start(); $('#pg-play').textContent = 'Pause'; } });
  $('#pg-step').addEventListener('click', () => { stop(); trainEpoch(); draw(); });
  $('#pg-reset').addEventListener('click', () => { stop(); build(); });
  const segData = UL.seg('pg-data', (v) => { st.data = v; stop(); makeData(); build(); });
  const noise = UL.range('pg-noise', (v) => { st.noise = v / 100; stop(); makeData(); build(); }, (v) => v + '%');
  $('#pg-split').addEventListener('change', (e) => { st.split = +e.target.value; stop(); makeData(); build(); });
  const segAct = UL.seg('pg-act', (v) => { st.act = v; stop(); build(); });
  const eta = UL.range('pg-eta', (v) => { st.lr = 10 ** v; }, (v) => { const x = 10 ** v; return x >= 0.1 ? x.toFixed(2) : x.toFixed(3); });
  const fe = $('#pg-feat');
  function featChips() { fe.innerHTML = FEATS.map((f) => `<label data-f="${f[0]}" class="${st.feats.has(f[0]) ? 'on' : ''}">${f[1]}</label>`).join(''); $$('label', fe).forEach((l) => l.addEventListener('click', () => { const k = l.dataset.f; if (st.feats.has(k)) { if (st.feats.size > 1) st.feats.delete(k); } else st.feats.add(k); stop(); featChips(); build(); })); }
  const hid = $('#pg-hidden');
  const setHidden = (txt) => { hid.value = txt; st.hidden = txt.split(/[ ,]+/).map((v) => parseInt(v, 10)).filter((v) => v > 0).slice(0, 5).map((v) => Math.min(8, v)); stop(); build(); };
  hid.addEventListener('change', () => setHidden(hid.value));
  $$('[data-h]').forEach((b) => b.addEventListener('click', () => setHidden(b.dataset.h)));
  $('#pg-showtest').addEventListener('change', () => draw());

  /* ------------------------------------------------------------ challenges */
  const CH = [
    { t: 'One neuron, the circle', set: { data: 'circle', feats: ['x1', 'x2'], hidden: '', act: 'tanh', eta: -1 }, q: 'With no hidden layer and only x₁, x₂ as inputs, can the network separate the inner dots from the outer ring?', a: 'No. A single neuron computes σ(w₁x₁ + w₂x₂ + b): its boundary is always a straight line, and no straight line separates a disc from a ring. The cost stays high.' },
    { t: 'Give it better inputs', set: { data: 'circle', feats: ['x1sq', 'x2sq'], hidden: '', act: 'tanh', eta: -1 }, q: 'Same single neuron, but the inputs are now x₁² and x₂². What happens, and why?', a: 'It solves it. x₁² + x₂² is the squared distance from the centre, so a straight line in these new inputs is a circle in the original plane. Choosing features by hand is what we did before deep learning; hidden layers learn such features themselves.' },
    { t: 'Let hidden layers find the features', set: { data: 'circle', feats: ['x1', 'x2'], hidden: '4', act: 'tanh', eta: -1 }, q: 'Back to x₁, x₂, but with 4 hidden neurons. Hover each hidden neuron: what does each one draw, and how does the output combine them?', a: 'Each hidden neuron draws one straight boundary (one tanh “cliff”). The output neuron adds them up so that only the middle region, inside all four lines, is blue: a rough circle made of lines.' },
    { t: 'XOR needs a hidden layer', set: { data: 'xor', feats: ['x1', 'x2'], hidden: '', act: 'tanh', eta: -1 }, q: 'Blue in two opposite corners, orange in the other two. Try it with no hidden layer, then with hidden layers “4”. What is the smallest network that works?', a: 'No straight line can separate XOR, so a single neuron fails (this famous result slowed neural network research in 1969). Two hidden neurons are enough in principle; three or four learn it reliably. Also try the input x₁x₂ alone.' },
    { t: 'No activation, no curves', set: { data: 'circle', feats: ['x1', 'x2'], hidden: '8, 8, 4', act: 'linear', eta: -1.5 }, q: 'Three hidden layers with 20 neurons, but the activation is “none”. Can it learn the circle?', a: 'No: the boundary stays a straight line however long you train. Without activation functions, all the layers multiply out into one linear model. Switch to tanh or ReLU and press Play again.' },
    { t: 'The spiral', set: { data: 'spiral', feats: ['x1', 'x2'], hidden: '8, 8, 4', act: 'relu', eta: -1 }, q: 'The hardest one. Watch the layers: do the first-layer neurons look like the spiral? Where does the spiral shape appear?', a: 'First-layer neurons are still straight lines; the second and third layers combine them into curved, banded regions, and only the deeper layers look spiral-like. That is “each layer builds bigger pieces out of the layer below”. It takes one or two hundred epochs. Try tanh as well: it is usually slower here.' },
    { t: 'Too large a learning rate', set: { data: 'gauss', feats: ['x1', 'x2'], hidden: '4, 2', act: 'tanh', eta: 0.5 }, q: 'An easy dataset, but η ≈ 3. Watch the cost curve. What do you see?', a: 'The cost jumps up and down instead of falling smoothly, and the boundary flips around: the steps overshoot, exactly like the ball on the curve in chapter 4. Lower η to 0.03 and it settles.' },
    { t: 'Overfitting', set: { data: 'spiral', feats: ['x1', 'x2', 'x1sq', 'x2sq', 'x1x2', 'sin1', 'sin2'], hidden: '8, 8, 8', act: 'relu', eta: -1, split: '0.2', noise: 35 }, q: 'Only 20% of the noisy points are used for training. Run for a few hundred epochs and compare the two cost curves. Tick “show test points”.', a: 'The training cost keeps falling while the test cost stops falling and starts to rise: the network bends its boundary around individual noisy training points. It is memorising, not generalising. More data, a smaller network or stopping early all help.' },
  ];
  $('#pg-ch').innerHTML = CH.map((c, i) => `<li><p style="margin:0"><b>${c.t}.</b> ${c.q}</p><button class="btn small" data-c="${i}">Set up</button> <button class="btn ghost small" data-r="${i}">Show answer</button><div class="reveal" hidden>${c.a}</div></li>`).join('');
  $$('[data-c]').forEach((b) => b.addEventListener('click', () => {
    const s = CH[+b.dataset.c].set; stop();
    st.data = s.data; segData.set(s.data, false); st.feats = new Set(s.feats); featChips(); st.act = s.act; segAct.set(s.act, false);
    eta.set(s.eta); st.lr = 10 ** s.eta; st.split = +(s.split || 0.5); $('#pg-split').value = String(s.split || 0.5); st.noise = (s.noise != null ? s.noise : 10) / 100; noise.set(s.noise != null ? s.noise : 10);
    hid.value = s.hidden; st.hidden = s.hidden.split(/[ ,]+/).map((v) => parseInt(v, 10)).filter((v) => v > 0);
    makeData(); build(); document.getElementById('lab').scrollIntoView({ behavior: 'smooth' });
  }));
  $$('[data-r]').forEach((b) => b.addEventListener('click', () => { const r = b.nextElementSibling; r.hidden = !r.hidden; b.textContent = r.hidden ? 'Show answer' : 'Hide answer'; }));

  featChips(); makeData(); build();
  UL.start();
})();
