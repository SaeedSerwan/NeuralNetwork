/* Neural Network Lab — the network in 3D
   A 784-300-100-10 network drawn as stacked layers of neurons, in the spirit of Adam W. Harley's
   "An Interactive Node-Link Visualization of Convolutional Neural Networks" (ISVC 2015).
   Written from scratch for this lab, with no libraries. Additions: the backward pass (blame flowing down),
   one learning step on your own drawing, and training the whole network from random weights in the browser.
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$;
  const SIZES = [784, 300, 100, 10], L = 3;

  /* ================================================================ the network */
  class Net {
    constructor() {
      this.W = [1, 2, 3].map((l) => new Float32Array(SIZES[l] * SIZES[l - 1])); this.b = [1, 2, 3].map((l) => new Float32Array(SIZES[l]));
      this.gW = this.W.map((w) => new Float32Array(w.length)); this.gb = this.b.map((b) => new Float32Array(b.length));
      this.vW = this.W.map((w) => new Float32Array(w.length)); this.vb = this.b.map((b) => new Float32Array(b.length));
      this.a = SIZES.map((n) => new Float32Array(n)); this.z = SIZES.map((n) => new Float32Array(n)); this.d = SIZES.map((n) => new Float32Array(n));
    }
    loadTrained() {
      const P = window.NN_BIG, bin = atob(P.b64), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      const q = new Int8Array(u8.buffer); let off = 0; const rowsQ = [];
      for (let l = 0; l < L; l++) { rowsQ.push(off); off += this.W[l].length; }
      const scales = []; for (let l = 0; l < L; l++) { scales.push(new Float32Array(u8.buffer, off, SIZES[l + 1])); off += 4 * SIZES[l + 1]; }
      for (let l = 0; l < L; l++) { this.b[l].set(new Float32Array(u8.buffer, off, SIZES[l + 1])); off += 4 * SIZES[l + 1]; }
      for (let l = 0; l < L; l++) { const nIn = SIZES[l]; for (let k = 0; k < this.W[l].length; k++) this.W[l][k] = q[rowsQ[l] + k] * scales[l][Math.floor(k / nIn)]; }
      this.vW.forEach((v) => v.fill(0)); this.vb.forEach((v) => v.fill(0));
    }
    randomize(seed = 1) {
      const r = UL.rng(seed);
      for (let l = 0; l < L; l++) { const s = Math.sqrt(2 / SIZES[l]); for (let k = 0; k < this.W[l].length; k++) this.W[l][k] = r.normal() * s; this.b[l].fill(0); }
      this.vW.forEach((v) => v.fill(0)); this.vb.forEach((v) => v.fill(0));
    }
    forward(x) {
      const a0 = this.a[0]; let nz = []; for (let i = 0; i < 784; i++) { a0[i] = x[i]; if (x[i] > 0) nz.push(i); }
      for (let l = 0; l < L; l++) {
        const nIn = SIZES[l], nOut = SIZES[l + 1], W = this.W[l], b = this.b[l], ain = this.a[l], z = this.z[l + 1], a = this.a[l + 1];
        for (let j = 0; j < nOut; j++) { let s = b[j]; const o = j * nIn; if (l === 0) { for (let t = 0; t < nz.length; t++) s += W[o + nz[t]] * ain[nz[t]]; } else for (let i = 0; i < nIn; i++) s += W[o + i] * ain[i]; z[j] = s; }
        if (l < L - 1) for (let j = 0; j < nOut; j++) a[j] = z[j] > 0 ? z[j] : 0;
        else { let m = -1e9; for (let j = 0; j < nOut; j++) m = Math.max(m, z[j]); let S = 0; for (let j = 0; j < nOut; j++) { a[j] = Math.exp(z[j] - m); S += a[j]; } for (let j = 0; j < nOut; j++) a[j] /= S; }
      }
      this.nz = nz; return this.a[L];
    }
    backward(label, acc = true) { // softmax + cross-entropy: δ = p − y at the output; then δ = (Wᵀδ) ⊙ relu'
      const d3 = this.d[3]; for (let j = 0; j < 10; j++) d3[j] = this.a[3][j] - (j === label ? 1 : 0);
      for (let l = L - 1; l >= 0; l--) {
        const nIn = SIZES[l], nOut = SIZES[l + 1], W = this.W[l], dout = this.d[l + 1], ain = this.a[l], gW = this.gW[l], gb = this.gb[l];
        if (acc) for (let j = 0; j < nOut; j++) { const dj = dout[j]; if (dj === 0) continue; gb[j] += dj; const o = j * nIn; if (l === 0) { for (let t = 0; t < this.nz.length; t++) { const i = this.nz[t]; gW[o + i] += dj * ain[i]; } } else for (let i = 0; i < nIn; i++) gW[o + i] += dj * ain[i]; }
        const din = this.d[l]; for (let i = 0; i < nIn; i++) { let s = 0; for (let j = 0; j < nOut; j++) s += W[j * nIn + i] * dout[j]; din[i] = l > 0 ? (ain[i] > 0 ? s : 0) : s; }
      }
    }
    step(lr, n = 1, mom = 0.9) {
      for (let l = 0; l < L; l++) {
        const W = this.W[l], gW = this.gW[l], vW = this.vW[l], b = this.b[l], gb = this.gb[l], vb = this.vb[l];
        for (let k = 0; k < W.length; k++) { if (gW[k] !== 0) { vW[k] = mom * vW[k] - (lr * gW[k]) / n; gW[k] = 0; } else vW[k] *= mom; W[k] += vW[k]; }
        for (let k = 0; k < b.length; k++) { vb[k] = mom * vb[k] - (lr * gb[k]) / n; gb[k] = 0; b[k] += vb[k]; }
      }
    }
    copyFrom(o) { for (let l = 0; l < L; l++) { this.W[l].set(o.W[l]); this.b[l].set(o.b[l]); } }
  }
  const net = new Net(); net.loadTrained();

  /* ================================================================ scene geometry */
  const GRID = [{ cols: 28, rows: 28, sp: 0.55, y: 0, size: 0.46 }, { cols: 20, rows: 15, sp: 0.8, y: 9, size: 0.5 }, { cols: 10, rows: 10, sp: 1.15, y: 17.5, size: 0.62 }, { cols: 10, rows: 1, sp: 2.0, y: 25, size: 1.1 }];
  const nodes = [];
  SIZES.forEach((n, l) => { const g = GRID[l]; for (let k = 0; k < n; k++) { const c = k % g.cols, r = Math.floor(k / g.cols); nodes.push({ l, k, x: (c - (g.cols - 1) / 2) * g.sp, y: g.y, z: (r - (g.rows - 1) / 2) * g.sp, s: g.size }); } });
  const nodeOf = SIZES.map(() => []); nodes.forEach((n) => { nodeOf[n.l][n.k] = n; });

  /* ================================================================ camera and drawing */
  const view = document.getElementById('v3-canvas'), ctx = view.getContext('2d');
  let Wd = 0, Ht = 0, dpr = 1;
  const cam = { yaw: -0.5, pitch: 0.32, dist: 46, tx: 0, ty: 12.5, tz: 0 };
  let basis;
  function resize() { const r = view.getBoundingClientRect(); dpr = Math.min(2, window.devicePixelRatio || 1); Wd = r.width; Ht = r.height; view.width = Math.round(Wd * dpr); view.height = Math.round(Ht * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); dirty = true; }
  function updateBasis() {
    const cp = Math.cos(cam.pitch), px = cam.tx + cam.dist * cp * Math.sin(cam.yaw), py = cam.ty + cam.dist * Math.sin(cam.pitch), pz = cam.tz + cam.dist * cp * Math.cos(cam.yaw);
    let f = [cam.tx - px, cam.ty - py, cam.tz - pz]; const fl = Math.hypot(...f); f = f.map((v) => v / fl);
    let r = [f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0]; const rl = Math.hypot(...r); r = r.map((v) => v / rl);
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    basis = { p: [px, py, pz], f, r, u, fo: Math.min(Wd, Ht * 1.35) * 1.05 };
  }
  function proj(x, y, z) { const B = basis, dx = x - B.p[0], dy = y - B.p[1], dz = z - B.p[2], d = dx * B.f[0] + dy * B.f[1] + dz * B.f[2]; return [Wd / 2 + (B.fo * (dx * B.r[0] + dy * B.r[1] + dz * B.r[2])) / d, Ht / 2 - (B.fo * (dx * B.u[0] + dy * B.u[1] + dz * B.u[2])) / d, d]; }

  // colours: activation ramp on a dark background; blame: orange (should go down) / blue (should go up)
  const RAMP = [[38, 44, 58], [32, 94, 128], [44, 174, 196], [236, 228, 160], [255, 255, 255]];
  const ramp = (t) => { t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1); const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i, a = RAMP[i], b = RAMP[i + 1]; return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; };
  const blame = (t) => { const k = Math.min(1, Math.abs(t)) ** 0.7, base = RAMP[0], to = t > 0 ? [255, 132, 64] : [90, 170, 255]; return [base[0] + (to[0] - base[0]) * k, base[1] + (to[1] - base[1]) * k, base[2] + (to[2] - base[2]) * k]; };
  const rgb = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

  /* ================================================================ state */
  let px = new Float32Array(784), mode = 'forward', reveal = 4, breveal = 0, sel = null, hover = null, dirty = true, paths = true, label = null, autoRot = false;
  const layerMax = () => [1, ...[1, 2].map((l) => { let m = 1e-6; for (const v of net.a[l]) m = Math.max(m, v); return m; }), 1];
  const blameMax = () => SIZES.map((n, l) => { let m = 1e-9; for (const v of net.d[l]) m = Math.max(m, Math.abs(v)); return m; });
  function value(l, k, LM, BM) { // the number that colours neuron k of layer l
    if (mode === 'back' && l > 0) { const vis = Math.max(0, Math.min(1, breveal - (3 - l))); return { c: blame((vis * net.d[l][k]) / BM[l]), glow: vis * Math.abs(net.d[l][k]) / BM[l] }; }
    const vis = l === 0 ? 1 : Math.max(0, Math.min(1, reveal - (l - 1))), t = (net.a[l][k] / LM[l]) * vis;
    return { c: ramp(l === 3 ? Math.sqrt(t) : t), glow: t };
  }

  function draw() {
    dirty = false; updateBasis(); ctx.clearRect(0, 0, Wd, Ht);
    const LM = layerMax(), BM = blameMax();
    // faint layer planes
    GRID.forEach((g, l) => { const hw = ((g.cols - 1) / 2) * g.sp + g.sp, hd = ((g.rows - 1) / 2) * g.sp + g.sp, P = [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].map(([x, z]) => proj(x, g.y - 0.3, z)); ctx.beginPath(); P.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.fillStyle = 'rgba(255,255,255,0.025)'; ctx.fill(); ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.stroke(); });
    // connections
    ctx.lineCap = 'round';
    const edge = (a, b, col, alpha, w) => { const p = proj(a.x, a.y, a.z), q = proj(b.x, b.y, b.z); ctx.strokeStyle = rgb(col, alpha); ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };
    if (paths) for (let l = 0; l < L; l++) {
      const vis = mode === 'back' ? Math.max(0, Math.min(1, breveal - (2 - l))) : Math.max(0, Math.min(1, reveal - l)); if (vis <= 0) continue;
      const nIn = SIZES[l], nOut = SIZES[l + 1], W = net.W[l], items = [];
      const src = l === 0 ? net.nz : null;
      for (let j = 0; j < nOut; j++) { const o = j * nIn; if (src) for (const i of src) items.push([W[o + i] * (mode === 'back' ? net.d[l + 1][j] : net.a[l][i]), i, j]); else for (let i = 0; i < nIn; i++) items.push([W[o + i] * (mode === 'back' ? net.d[l + 1][j] : net.a[l][i]), i, j]); }
      if (mode !== 'back') for (const it of items) if (net.a[l + 1][it[2]] <= 0 && l < 2) it[0] *= 0.15;
      items.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0])); const top = items.slice(0, l === 2 ? 60 : 140), mx = Math.abs(top[0] ? top[0][0] : 1) || 1;
      top.forEach(([v, i, j]) => edge(nodeOf[l][i], nodeOf[l + 1][j], mode === 'back' ? (v > 0 ? [255, 132, 64] : [90, 170, 255]) : v > 0 ? [80, 220, 170] : [240, 90, 90], vis * (0.12 + 0.55 * Math.abs(v) / mx), 0.5 + 1.6 * Math.abs(v) / mx));
    }
    if (sel && sel.l > 0) { // all incoming connections of the selected neuron
      const l = sel.l - 1, nIn = SIZES[l], o = sel.k * nIn, W = net.W[l], items = [];
      for (let i = 0; i < nIn; i++) if (l > 0 || net.a[0][i] > 0) items.push([W[o + i] * net.a[l][i], i]);
      items.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0])); const mx = Math.abs(items[0] ? items[0][0] : 1) || 1;
      items.slice(0, 200).forEach(([v, i]) => edge(nodeOf[l][i], sel, v > 0 ? [80, 220, 170] : [240, 90, 90], 0.25 + 0.7 * Math.abs(v) / mx, 0.6 + 2.2 * Math.abs(v) / mx));
    }
    // neurons, far ones first
    const list = nodes.map((n) => { const p = proj(n.x, n.y, n.z); return [p[2], n, p]; }).sort((a, b) => b[0] - a[0]);
    for (const [d, n, p] of list) {
      const { c, glow } = value(n.l, n.k, LM, BM), s = (basis.fo * n.s) / d;
      if (glow > 0.55 && n.l > 0) { ctx.fillStyle = rgb(c, 0.18); ctx.fillRect(p[0] - s * 1.1, p[1] - s * 1.1, s * 2.2, s * 2.2); }
      ctx.fillStyle = rgb(c); ctx.fillRect(p[0] - s / 2, p[1] - s / 2, s, s);
      if (n === sel || n === hover) { ctx.strokeStyle = n === sel ? '#ffd166' : 'rgba(255,255,255,.8)'; ctx.lineWidth = 2; ctx.strokeRect(p[0] - s / 2 - 2, p[1] - s / 2 - 2, s + 4, s + 4); }
      if (n.l === 3) { ctx.fillStyle = 'rgba(255,255,255,.85)'; ctx.font = `600 ${Math.max(10, s * 0.8)}px "IBM Plex Mono", monospace`; ctx.textAlign = 'center'; ctx.fillText(String(n.k), p[0], p[1] - s * 0.75); }
    }
    // layer captions
    const caps = ['784 inputs (pixels)', 'hidden layer 1 · 300', 'hidden layer 2 · 100', 'output · 10 digits'];
    GRID.forEach((g, l) => { const hw = ((g.cols - 1) / 2) * g.sp + g.sp * 1.4, p = proj(-hw, g.y, 0); ctx.fillStyle = 'rgba(255,255,255,.55)'; ctx.font = '12px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'right'; ctx.fillText(caps[l], p[0] - 8, p[1] + 4); });
    lastList = list;
  }
  let lastList = [];
  const pick = (mx, my) => { let best = null, bd = 1e9; for (let i = lastList.length - 1; i >= 0; i--) { const [d, n, p] = lastList[i], s = (basis.fo * n.s) / d, e = Math.max(Math.abs(mx - p[0]), Math.abs(my - p[1])); if (e < Math.max(4, s * 0.7) && d < bd) { best = n; bd = d; } } return best; };

  /* ================================================================ mouse: orbit, zoom, pick */
  let drag = null;
  view.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, yaw: cam.yaw, pitch: cam.pitch, moved: false }; view.setPointerCapture(e.pointerId); });
  view.addEventListener('pointermove', (e) => {
    const r = view.getBoundingClientRect();
    if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true; cam.yaw = drag.yaw - dx * 0.006; cam.pitch = Math.max(-0.2, Math.min(1.45, drag.pitch + dy * 0.005)); dirty = true; return; }
    const h = pick(e.clientX - r.left, e.clientY - r.top); if (h !== hover) { hover = h; dirty = true; tip(h, e.clientX - r.left, e.clientY - r.top); } else if (h) tip(h, e.clientX - r.left, e.clientY - r.top);
  });
  view.addEventListener('pointerup', (e) => { const r = view.getBoundingClientRect(); if (drag && !drag.moved) { sel = pick(e.clientX - r.left, e.clientY - r.top); inspect(); dirty = true; } drag = null; });
  view.addEventListener('wheel', (e) => { e.preventDefault(); cam.dist = Math.max(18, Math.min(110, cam.dist * (1 + Math.sign(e.deltaY) * 0.08))); dirty = true; }, { passive: false });
  const tipEl = $('#v3-tip');
  function tip(n, x, y) {
    if (!n) { tipEl.hidden = true; return; }
    const v = mode === 'back' && n.l > 0 ? `blame δ = ${net.d[n.l][n.k].toFixed(4)}` : n.l === 3 ? `probability ${(net.a[3][n.k] * 100).toFixed(1)}%` : `activation ${net.a[n.l][n.k].toFixed(2)}`;
    tipEl.hidden = false; tipEl.style.left = x + 14 + 'px'; tipEl.style.top = y + 10 + 'px';
    tipEl.innerHTML = `<b>${n.l === 0 ? `pixel (${Math.floor(n.k / 28) + 1}, ${(n.k % 28) + 1})` : n.l === 3 ? `output “${n.k}”` : `layer ${n.l} · neuron ${n.k + 1}`}</b><br>${v}<br><span>click for details</span>`;
  }

  /* ================================================================ inspector: weighted input, calculation, output */
  const ins = $('#v3-ins'), f3 = (v) => UL.fmt(v, 3);
  function inspect() {
    if (!sel) { ins.innerHTML = '<p class="v3-hint">Click any neuron to see how it computes its number.</p>'; return; }
    const { l, k } = sel;
    if (l === 0) { ins.innerHTML = `<h4>Input pixel (${Math.floor(k / 28) + 1}, ${(k % 28) + 1})</h4><p>No calculation: it simply holds the brightness of one pixel of your drawing.</p><div class="v3-out">${net.a[0][k].toFixed(2)}</div>`; return; }
    const nIn = SIZES[l - 1], o = k * nIn, W = net.W[l - 1], items = []; let sum = 0;
    for (let i = 0; i < nIn; i++) { const p = W[o + i] * net.a[l - 1][i]; sum += p; if (net.a[l - 1][i] !== 0) items.push([p, i, W[o + i], net.a[l - 1][i]]); }
    items.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0])); const mx = Math.abs(items[0] ? items[0][0] : 1) || 1, b = net.b[l - 1][k], z = sum + b;
    const name = (i) => (l === 1 ? `pixel ${Math.floor(i / 28) + 1},${(i % 28) + 1}` : `L${l - 1} #${i + 1}`);
    const rows = items.slice(0, 9).map(([p, i, w, a]) => `<div class="v3-row"><span>${name(i)}</span><span>${a.toFixed(2)} × ${w >= 0 ? '' : '−'}${Math.abs(w).toFixed(2)}</span><span class="${p >= 0 ? 'pos' : 'neg'}">${p >= 0 ? '+' : '−'}${Math.abs(p).toFixed(3)}</span><i style="width:${(Math.abs(p) / mx) * 100}%;background:${p >= 0 ? '#3ccf9b' : '#f05a5a'}"></i></div>`).join('');
    const out = l === 3 ? `softmax: e<sup>z</sup> / Σ e<sup>z</sup> = <b>${(net.a[3][k] * 100).toFixed(1)}%</b>` : `ReLU(z) = max(0, ${f3(z)}) = <b>${net.a[l][k].toFixed(3)}</b>`;
    const back = mode === 'back' ? `<h5>Backward pass</h5><p>Blame on this neuron: δ = <b>${net.d[l][k].toFixed(4)}</b>. ${Math.abs(net.d[l][k]) < 1e-6 ? 'Zero: it was not active (ReLU slope 0), so it gets no blame and its weights will not change.' : net.d[l][k] > 0 ? 'Positive: its value should go <b>down</b>; its weights from active inputs will shrink.' : 'Negative: its value should go <b>up</b>; its weights from active inputs will grow.'} Each incoming weight changes by −η × (input) × δ.</p>` : '';
    ins.innerHTML = `<h4>${l === 3 ? `Output neuron “${k}”` : `Layer ${l}, neuron ${k + 1}`}</h4>
      <h5>1 · Weighted input</h5><p>${items.length} of its ${nIn} inputs are non-zero. The strongest:</p><div class="v3-rows">${rows}</div>${items.length > 9 ? `<p class="v3-more">… and ${items.length - 9} smaller products</p>` : ''}
      <h5>2 · Calculation</h5><p>sum of all products = ${f3(sum)}<br>+ bias ${f3(b)}<br>= z = <b>${f3(z)}</b></p>
      <h5>3 · Output</h5><p>${out}</p>${back}`;
  }

  /* ================================================================ drawing pad and guesses */
  const small = UL.pixPlot('#v3-small', 84);
  function guesses() {
    const p = net.a[3], ord = Array.from(p).map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]), empty = px.every((v) => v === 0);
    $('#v3-g1').innerHTML = empty ? '–' : ord[0][1]; $('#v3-g1p').textContent = empty ? '' : (ord[0][0] * 100).toFixed(1) + '%';
    $('#v3-g2').innerHTML = empty ? '–' : ord[1][1]; $('#v3-g2p').textContent = empty ? '' : (ord[1][0] * 100).toFixed(1) + '%';
  }
  function compute() { net.forward(px); if (mode === 'back' && label != null) net.backward(label, false); UL.drawDigit(small.ctx, px, 0, 0, 84); guesses(); inspect(); dirty = true; }
  const pad = new UL.DrawPad('#v3-pad', { size: 280, onchange: (p) => { px = p; if (mode === 'back') setMode('forward'); reveal = 4; compute(); } });
  const clock = new UL.Clock();
  async function playForward() { setMode('forward'); clock.cancel(); try { await clock.tween(1700, (t, raw) => { reveal = raw * 3.2; dirty = true; }); } catch (e) { return; } reveal = 4; dirty = true; }
  pad.cv.addEventListener('pointerup', playForward);
  $('#v3-clear').addEventListener('click', () => pad.clear());
  const S = UL.samples(); let si = 0;
  $('#v3-rand').addEventListener('click', () => { si = (si + 7 + Math.floor(Math.random() * 20)) % S.n; pad.setPixels(S.get(si)); playForward(); });
  $('#v3-fwd').addEventListener('click', playForward);

  /* ================================================================ backward pass and one learning step */
  const nar = $('#v3-nar');
  function setMode(m) { mode = m; $$('.v3-modes button').forEach((b) => b.classList.toggle('on', b.dataset.m === m)); if (m === 'forward') nar.innerHTML = NARR.forward; dirty = true; }
  const NARR = {
    forward: '<b>Forward pass.</b> Your drawing lights up the 784 inputs at the bottom. Each neuron above adds up its inputs × weights, adds a bias and applies ReLU; brighter = more active. Lines show the strongest contributions (green +, red −).',
  };
  $('#v3-labels').innerHTML = [...Array(10).keys()].map((d) => `<button data-d="${d}">${d}</button>`).join('');
  $$('#v3-labels button').forEach((b) => b.addEventListener('click', () => { label = +b.dataset.d; $$('#v3-labels button').forEach((x) => x.classList.toggle('on', x === b)); playBackward(); }));
  async function playBackward() {
    if (label == null) { nar.innerHTML = 'Pick the <b>correct digit</b> first, then the error is sent back from the top.'; return; }
    mode = 'back'; $$('.v3-modes button').forEach((b) => b.classList.toggle('on', b.dataset.m === 'back')); net.forward(px); net.backward(label, false); inspect();
    const p = net.a[3][label];
    nar.innerHTML = `<b>Backward pass.</b> We wanted “${label}”; the network gave it ${(p * 100).toFixed(1)}%. At the top, error δ = output − wanted: <span style="color:#ff8440">orange</span> outputs are too high, <span style="color:#5aaaff">blue</span> too low. The blame flows down the same connections, weighted by the same weights. Inactive neurons (ReLU = 0) pass no blame.`;
    clock.cancel(); try { await clock.tween(2200, (t, raw) => { breveal = raw * 3; dirty = true; }); } catch (e) { return; } breveal = 3; dirty = true;
  }
  $('#v3-learn').addEventListener('click', () => {
    if (label == null) { nar.innerHTML = 'Pick the <b>correct digit</b> under “Backward pass” first.'; return; }
    const before = net.forward(px)[label]; net.backward(label, true); net.step(0.005, 1, 0); const after = net.forward(px)[label];
    net.backward(label, false); mode = 'back'; breveal = 3; guesses(); inspect(); dirty = true;
    nar.innerHTML = `<b>One learning step</b> on your drawing: every weight moved by −η × input × δ (η = 0.005). The probability of “${label}” went from <b>${(before * 100).toFixed(1)}%</b> to <b>${(after * 100).toFixed(1)}%</b>. Press again to keep teaching it, or “Restore trained weights” to undo.`;
  });
  $('#v3-restore').addEventListener('click', () => { stopTrain(); net.loadTrained(); setMode('forward'); compute(); $('#v3-train-st').textContent = 'trained weights (98.8% on 10,000 test digits)'; });

  /* ================================================================ training from random weights */
  let D = null, trainOn = false, seen = 0, order = [], ptr = 0, accHist = [];
  const TP = new UL.Plot('#v3-acc', { w: 260, h: 90, pad: { l: 28, r: 6, t: 6, b: 16 } });
  const testAcc = (n = 400) => { let ok = 0; for (let k = 0; k < n; k++) { const i = D.nTrain + k; const p = net.forward(D.get(i)); if (UL.argmax(p) === D.y[i]) ok++; } net.forward(px); return ok / n; };
  function accPlot() { TP.clear(); TP.domain(0, Math.max(5000, seen), 0, 1); TP.ctx.strokeStyle = 'rgba(255,255,255,.15)'; TP.line(0, 0.9, Math.max(5000, seen), 0.9, { color: 'rgba(255,255,255,.25)', dash: [3, 3], w: 1 }); if (accHist.length) TP.poly(accHist, { color: '#ffd166', w: 2 }); }
  const trainLoop = UL.loop(() => {
    const t0 = performance.now(); let n = 0;
    while (performance.now() - t0 < 22) { for (let b = 0; b < 16; b++) { if (ptr >= order.length) { order.sort(() => Math.random() - 0.5); ptr = 0; } const i = order[ptr++]; net.forward(D.get(i)); net.backward(D.y[i], true); } net.step(0.02, 16, 0.9); seen += 16; n += 16; }
    if (seen % 800 < n) { const a = testAcc(); accHist.push([seen, a]); accPlot(); $('#v3-train-st').innerHTML = `trained on <b>${seen.toLocaleString('en-US')}</b> digits · test accuracy <b>${(a * 100).toFixed(1)}%</b>`; }
    net.forward(px); guesses(); if (sel) inspect(); dirty = true;
    if (seen >= 60000) { stopTrain(); }
  });
  function stopTrain() { if (trainLoop.on) trainLoop.stop(); trainOn = false; $('#v3-train').textContent = '▶ Train from random weights'; }
  $('#v3-train').addEventListener('click', async () => {
    if (trainOn) { stopTrain(); return; }
    $('#v3-train-st').textContent = 'loading 6,000 digits…'; if (!D) D = await UL.digits();
    if (!seen || $('#v3-train').dataset.fresh !== '1') { net.randomize(Math.floor(Math.random() * 1e6)); seen = 0; accHist = [[0, testAcc()]]; order = Array.from({ length: D.nTrain }, (_, i) => i); ptr = order.length; $('#v3-train').dataset.fresh = '1'; }
    setMode('forward'); reveal = 4; trainOn = true; $('#v3-train').textContent = '❚❚ Pause training'; trainLoop.start();
  });
  $('#v3-restore').addEventListener('click', () => { $('#v3-train').dataset.fresh = '0'; seen = 0; accHist = []; accPlot(); });

  /* ================================================================ toggles and render loop */
  $('#v3-paths').addEventListener('change', (e) => { paths = e.target.checked; dirty = true; });
  $('#v3-rot').addEventListener('change', (e) => { autoRot = e.target.checked; });
  $$('.v3-modes button').forEach((b) => b.addEventListener('click', () => { if (b.dataset.m === 'forward') playForward(); else playBackward(); }));
  $('#v3-reset-view').addEventListener('click', () => { Object.assign(cam, { yaw: -0.5, pitch: 0.32, dist: 46 }); dirty = true; });
  window.addEventListener('resize', resize);
  function frame() { if (autoRot && !drag) { cam.yaw += 0.0025; dirty = true; } if (dirty) draw(); requestAnimationFrame(frame); }
  resize(); setMode('forward'); pad.setPixels(S.get(S.ofDigit(3, 1))); reveal = 0; accPlot(); requestAnimationFrame(frame); setTimeout(playForward, 400);
})();
