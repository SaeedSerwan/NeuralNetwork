/* Part 1 · Neurons & layers
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$, S = UL.samples();
  const f2 = (v) => UL.fmt(v, 2);

  /* ------------------------------------------------------------ 01 pixels */
  (function () {
    const grid = UL.pixPlot('#px-grid', 336), read = $('#px-read'), strip = $('#px-strip');
    let px = new Float32Array(784);
    function draw(hi = -1) {
      const c = grid.ctx, s = 12; grid.clear();
      for (let i = 0; i < 784; i++) {
        const g = Math.round(255 * px[i]); c.fillStyle = `rgb(${g},${g},${g})`; c.fillRect((i % 28) * s, Math.floor(i / 28) * s, s, s);
      }
      c.strokeStyle = 'rgba(255,255,255,.08)'; c.lineWidth = 1;
      for (let k = 0; k <= 28; k++) { c.beginPath(); c.moveTo(k * s + 0.5, 0); c.lineTo(k * s + 0.5, 336); c.moveTo(0, k * s + 0.5); c.lineTo(336, k * s + 0.5); c.stroke(); }
      if (hi >= 0) { c.strokeStyle = '#eb6834'; c.lineWidth = 2; c.strokeRect((hi % 28) * s + 1, Math.floor(hi / 28) * s + 1, s - 2, s - 2); }
      const nz = px.reduce((n, v) => n + (v > 0.05 ? 1 : 0), 0);
      strip.innerHTML = `As a list: <code>[${Array.from(px.slice(0, 1)).map((v) => v.toFixed(2)).join(', ')}, …, ${Array.from(px.slice(370, 378)).map((v) => v.toFixed(2)).join(', ')}, …]</code> · 784 numbers, ${nz} of them not black.`;
    }
    function info(i) {
      if (i < 0) { read.innerHTML = '<span class="k">Neuron</span><span>hover the grid</span>'; return; }
      read.innerHTML = `<span class="k">Neuron number</span><span class="hl">${i + 1} of 784</span><span class="k">Row, column</span><span>${Math.floor(i / 28) + 1}, ${(i % 28) + 1}</span><span class="k">Activation</span><span class="hl">${px[i].toFixed(2)}</span><span class="k">Meaning</span><span>${px[i] < 0.05 ? 'black: 0' : px[i] > 0.95 ? 'white: 1' : 'grey'}</span>`;
    }
    const pad = new UL.DrawPad('#px-pad', { onchange: (p) => { px = p; draw(); } });
    $('#px-clear').addEventListener('click', () => { pad.clear(); th.clear(); });
    const th = UL.thumbs('px-thumbs', [0, 1, 2, S.ofDigit(7), S.ofDigit(8), S.ofDigit(2), S.ofDigit(5)], (i) => pad.setPixels(S.get(i)));
    grid.cv.addEventListener('mousemove', (e) => { const [X, Y] = grid.pointer(e), i = Math.floor(Y / 12) * 28 + Math.floor(X / 12); if (i >= 0 && i < 784) { draw(i); info(i); } });
    grid.cv.addEventListener('mouseleave', () => { draw(); info(-1); });
    pad.setPixels(S.get(0)); info(-1);
  })();

  /* ------------------------------------------------------------ 02 network */
  (function () {
    const layers = [{ n: 784, shown: 22, label: 'Input · 784' }, { n: 16, label: 'Hidden · 16' }, { n: 16, label: 'Hidden · 16' }, { n: 10, label: 'Output · 10' }];
    const N = UL.netSVG('net-svg', layers, { w: 620, h: 360, r: 6.5, digits: true });
    const info = $('#net-info');
    const TXT = [
      ['Input layer: 784 neurons', 'One neuron per pixel, holding its brightness from 0 to 1. Nothing is computed here; the picture is simply copied in.'],
      ['Hidden layer 1: 16 neurons', 'Each neuron computes a weighted sum of all 784 pixels, adds its bias and applies σ. The hope: these find small edges.'],
      ['Hidden layer 2: 16 neurons', 'Each reads all 16 neurons of hidden layer 1. The hope: these find bigger parts, such as loops and lines.'],
      ['Output layer: 10 neurons', 'One per digit 0 … 9. Each reads all 16 neurons of hidden layer 2. The brightest one is the network’s answer.'],
    ];
    const show = (l, k) => {
      N.edges().forEach((e) => { const el = +e.dataset.l, i = +e.dataset.i, j = +e.dataset.j; const on = k == null ? (el === l || el === l - 1) : (el === l && i === k) || (el === l - 1 && j === k); e.setAttribute('stroke', on ? '#111' : '#e7e7e4'); e.setAttribute('stroke-width', on ? 0.9 : 0.5); });
      $$('circle.nd', N.svg).forEach((c) => c.setAttribute('fill', +c.dataset.l === l && (k == null || +c.dataset.k === k) ? '#ffe9a8' : '#fff'));
      const nIn = l > 0 ? layers[l - 1].n : 0;
      info.innerHTML = `<div><b>${TXT[l][0]}</b></div><div style="font-family:var(--font);color:var(--ink-2)">${TXT[l][1]}</div>` +
        (k != null && l > 0 ? `<div style="font-family:var(--font)">This one neuron has <b>${nIn}</b> weights coming in and <b>1</b> bias: ${nIn + 1} numbers to learn.</div>` : '') +
        (k != null && l === 3 ? `<div style="font-family:var(--font)">It is the neuron for the digit <b>${k}</b>.</div>` : '');
    };
    $$('circle.nd', N.svg).forEach((c) => c.addEventListener('mouseenter', () => show(+c.dataset.l, +c.dataset.k)));
    N.svg.addEventListener('mouseleave', () => show(1));
    show(1);
  })();

  /* ------------------------------------------------------------ 03 the hope */
  (function () {
    // parts drawn on a 100 × 100 box; each part lists its edges (short segments)
    const PARTS = {
      top: { name: 'loop up top', d: 'M50 12 a 22 20 0 1 1 -0.1 0', edges: ['M30 22 l10 -10', 'M60 12 l10 10', 'M72 34 l-10 12', 'M40 48 l-12 -12'] },
      bot: { name: 'loop at the bottom', d: 'M50 52 a 22 22 0 1 1 -0.1 0', edges: ['M30 62 l10 -10', 'M60 52 l10 10', 'M72 78 l-10 14', 'M40 94 l-12 -14'] },
      right: { name: 'line on the right', d: 'M72 20 L72 92', edges: ['M72 20 l0 24', 'M72 48 l0 22', 'M72 72 l0 20'] },
      left: { name: 'line on the left', d: 'M34 12 L30 58', edges: ['M34 12 l-2 22', 'M32 38 l-2 20'] },
      mid: { name: 'bar across', d: 'M24 58 L86 58', edges: ['M24 58 l20 0', 'M48 58 l18 0', 'M68 58 l18 0'] },
      vert: { name: 'long line', d: 'M52 10 L52 94', edges: ['M52 10 l0 28', 'M52 40 l0 26', 'M52 68 l0 26'] },
    };
    const DIG = { 9: ['top', 'right'], 8: ['top', 'bot'], 4: ['left', 'mid', 'right'], 1: ['vert'], 0: ['top', 'bot'] };
    const DIGITS = ['9', '8', '4', '1'];
    const box = (x, y, inner, cls, key) => `<g class="${cls}" data-k="${key}" transform="translate(${x},${y})" style="cursor:pointer"><rect width="100" height="104" rx="8" fill="#fff" stroke="#e7e7e4"/><g transform="translate(0,0)">${inner}</g></g>`;
    let edgesCol = '', partsCol = '', digCol = '';
    const allEdges = [];
    Object.keys(PARTS).forEach((k) => PARTS[k].edges.forEach((e) => { if (!allEdges.includes(e)) allEdges.push(e); }));
    allEdges.forEach((e, i) => { edgesCol += box(10 + (i % 4) * 58, 30 + Math.floor(i / 4) * 64, `<g transform="scale(.5)"><rect width="100" height="104" fill="none"/><path d="${e}" stroke="#111" stroke-width="7" stroke-linecap="round" fill="none"/></g>`, 'eg', i); });
    Object.keys(PARTS).forEach((k, i) => { partsCol += box(270 + (i % 2) * 112, 30 + Math.floor(i / 2) * 136, `<path d="${PARTS[k].d}" stroke="#111" stroke-width="7" stroke-linecap="round" fill="none"/><text x="50" y="118" text-anchor="middle">${PARTS[k].name}</text>`, 'pt', k); });
    digCol = DIGITS.map((d, i) => box(530 + (i % 2) * 112, 30 + Math.floor(i / 2) * 150, DIG[d].map((k) => `<path d="${PARTS[k].d}" stroke="#111" stroke-width="7" stroke-linecap="round" fill="none"/>`).join('') + `<text x="50" y="118" text-anchor="middle">the digit ${d}</text>`, 'dg', d)).join('');
    const head = (x, t) => `<text x="${x}" y="16" style="font:500 12px var(--font);fill:#111">${t}</text>`;
    // edge boxes are 50 wide (scaled), fix their rects
    document.getElementById('hope-svg').innerHTML = `<svg class="netsvg" viewBox="0 0 760 440">${head(10, 'Layer 1 · small edges')}${head(270, 'Layer 2 · parts')}${head(530, 'Output · digits')}${edgesCol}${partsCol}${digCol}</svg>`;
    const svg = document.querySelector('#hope-svg svg');
    $$('g.eg', svg).forEach((g) => { const r = g.querySelector('rect'); r.setAttribute('width', 50); r.setAttribute('height', 52); });
    const light = (parts) => {
      const edges = new Set(); parts.forEach((k) => PARTS[k].edges.forEach((e) => edges.add(allEdges.indexOf(e))));
      $$('g.pt', svg).forEach((g) => g.querySelector('rect').setAttribute('fill', parts.includes(g.dataset.k) ? '#ffe9a8' : '#fff'));
      $$('g.eg', svg).forEach((g) => g.querySelector('rect').setAttribute('fill', edges.has(+g.dataset.k) ? '#ffe9a8' : '#fff'));
    };
    $$('g.dg', svg).forEach((g) => g.addEventListener('mouseenter', () => { $$('g.dg rect', svg).forEach((r) => r.setAttribute('fill', '#fff')); g.querySelector('rect').setAttribute('fill', '#ffe9a8'); light(DIG[g.dataset.k]); }));
    $$('g.pt', svg).forEach((g) => g.addEventListener('mouseenter', () => { $$('g.dg rect', svg).forEach((r) => r.setAttribute('fill', '#fff')); light([g.dataset.k]); }));
    svg.addEventListener('mouseleave', () => { light([]); $$('g.dg rect', svg).forEach((r) => r.setAttribute('fill', '#fff')); });
  })();

  /* ------------------------------------------------------------ 04 weighted sum */
  (function () {
    const net = UL.pretrained();
    const PA = UL.pixPlot('#ws-a', 224), PW = UL.pixPlot('#ws-wt', 224), PP = UL.pixPlot('#ws-p', 224), calc = $('#ws-calc');
    const mk = (fn) => { const w = new Float32Array(784); for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) w[r * 28 + c] = fn(r, c); return w; };
    const W = {
      h: mk((r, c) => (c < 4 || c > 23 ? 0 : r >= 12 && r <= 15 ? 1 : (r >= 7 && r <= 11) || (r >= 16 && r <= 20) ? -0.8 : 0)),
      v: mk((r, c) => (r < 4 || r > 23 ? 0 : c >= 12 && c <= 15 ? 1 : (c >= 7 && c <= 11) || (c >= 16 && c <= 20) ? -0.8 : 0)),
      loop: mk((r, c) => { const d = Math.hypot(r - 9, c - 14); return d >= 4 && d <= 7.5 ? 0.7 : d < 3.5 ? -1 : 0; }),
      right: mk((r, c) => (r < 5 || r > 24 ? 0 : c >= 17 && c <= 20 ? 0.8 : c >= 10 && c <= 14 ? -0.6 : 0)),
      learned: (() => { let best = 0, bs = -1; for (let k = 0; k < 16; k++) { let s = 0; for (let i = 0; i < 784; i++) s += Math.abs(net.W[0][k * 784 + i]); if (s > bs) { bs = s; best = k; } } return Float32Array.from(net.W[0].subarray(best * 784, best * 784 + 784)); })(),
    };
    const TH = [S.ofDigit(4), S.ofDigit(7), S.ofDigit(1), S.ofDigit(9), S.ofDigit(0), S.ofDigit(2), 3, S.ofDigit(5)];
    let wk = 'h', hover = -1, idx = TH.reduce((best, i) => { const sc = (j) => S.get(j).reduce((t, v, k) => t + v * W.h[k], 0); return sc(i) > sc(best) ? i : best; }, TH[0]);
    const bias = UL.range('ws-b', () => draw());
    function draw() {
      const a = S.get(idx), w = W[wk], sc = wk === 'learned' ? 1.2 : 1, prod = new Float32Array(784); let sum = 0, pos = 0, neg = 0;
      for (let i = 0; i < 784; i++) { prod[i] = w[i] * a[i]; sum += prod[i]; if (prod[i] > 0) pos += prod[i]; else neg += prod[i]; }
      UL.drawDigit(PA.ctx, a, 0, 0, 224); UL.drawDigit(PW.ctx, w, 0, 0, 224, { signed: true, scale: sc }); UL.drawDigit(PP.ctx, prod, 0, 0, 224, { signed: true, scale: sc });
      if (hover >= 0) [PA, PW, PP].forEach((P) => { P.ctx.strokeStyle = '#eb6834'; P.ctx.lineWidth = 2; P.ctx.strokeRect((hover % 28) * 8, Math.floor(hover / 28) * 8, 8, 8); });
      const b = bias.v, z = sum + b, act = UL.sigmoid(z);
      calc.innerHTML = (hover >= 0 ? `<span class="k">This pixel</span><span>a = ${a[hover].toFixed(2)}, w = ${f2(w[hover])}, w × a = <b>${f2(prod[hover])}</b></span>` : '') +
        `<span class="k">Sum of green products</span><span class="pos">+${pos.toFixed(1)}</span><span class="k">Sum of red products</span><span class="neg">${UL.fmt(neg, 1)}</span>` +
        `<span class="k">Weighted sum</span><span class="hl">${UL.fmt(sum, 1)}</span><span class="k">+ bias</span><span>${UL.fmt(b, 0)} → z = ${UL.fmt(z, 1)}</span>` +
        `<span class="k">Activation σ(z)</span><span class="hl" style="font-size:1.1rem">${act.toFixed(2)} ${act > 0.5 ? '· the neuron is <b>on</b>' : '· the neuron stays <b>off</b>'}</span>`;
    }
    UL.seg('ws-w', (v) => { wk = v; if (v === 'learned') bias.set(0); else if (bias.v === 0) bias.set(-10); draw(); });
    UL.thumbs('ws-thumbs', TH, (i) => { idx = i; draw(); }, { select: TH.indexOf(idx) });
    PP.cv.addEventListener('mousemove', (e) => { const [X, Y] = PP.pointer(e); hover = Math.floor(Y / 8) * 28 + Math.floor(X / 8); draw(); });
    PP.cv.addEventListener('mouseleave', () => { hover = -1; draw(); });
    draw();
  })();

  /* ------------------------------------------------------------ 05 activation functions */
  (function () {
    const P = new UL.Plot('#sg-plot', { w: 560, h: 340, pad: { l: 40, r: 14, t: 14, b: 34 } }), calc = $('#sg-calc');
    let fk = 'sigmoid';
    const z = UL.range('sg-z', () => draw(), (v) => v.toFixed(1)), slope = $('#sg-slope');
    function draw() {
      const A = UL.ACT[fk], ymax = fk === 'relu' || fk === 'linear' ? 8 : 1.1, ymin = fk === 'tanh' ? -1.1 : fk === 'linear' ? -8 : -0.1;
      P.clear(); P.domain(-8, 8, ymin, ymax); P.axes({ xlabel: 'z (the weighted sum + bias)' });
      const pts = [], dpts = []; for (let x = -8; x <= 8.001; x += 0.05) { const a = A.f(x); pts.push([x, a]); dpts.push([x, A.d(x, a)]); }
      P.clipped(() => { if (slope.checked) P.poly(dpts, { color: UL.COLORS[1], w: 1.6, dash: [5, 4] }); P.poly(pts, { color: UL.INK, w: 2.4 }); });
      const zz = z.v, a = A.f(zz), d = A.d(zz, a);
      P.line(zz, ymin, zz, a, { color: UL.FAINT, dash: [3, 3], w: 1 }); P.marker(zz, a, { r: 5.5, fill: UL.COLORS[0] });
      calc.innerHTML = `<span class="k">Function</span><span class="hl">${A.name}</span><span class="k">z</span><span>${zz.toFixed(1)}</span><span class="k">output</span><span class="hl">${a.toFixed(3)}</span><span class="k">slope here</span><span class="pos">${d.toFixed(3)}</span>` +
        `<span class="k">Range</span><span>${{ sigmoid: '0 … 1', relu: '0 … ∞', tanh: '−1 … 1', linear: '−∞ … ∞' }[fk]}</span><span class="k">Largest slope</span><span>${{ sigmoid: '0.25 (at z = 0)', relu: '1 (for every z > 0)', tanh: '1 (at z = 0)', linear: '1 everywhere' }[fk]}</span>` +
        `<span class="k">Quick check</span><span>${fk === 'relu' ? 'ReLU(−3) = 0, ReLU(2.5) = 2.5' : fk === 'sigmoid' ? 'σ(0) = 0.5, σ(−1) = 0.27, σ(2) = 0.88' : fk === 'tanh' ? 'tanh(0) = 0' : 'output = input'}</span>`;
    }
    UL.seg('sg-f', (v) => { fk = v; draw(); }); slope.addEventListener('change', draw); draw();
  })();

  /* ------------------------------------------------------------ 06 bias */
  (function () {
    const P = new UL.Plot('#bi-plot', { w: 560, h: 320, pad: { l: 40, r: 14, t: 14, b: 34 } }), calc = $('#bi-calc');
    const b = UL.range('bi-b', () => draw(), (v) => v.toFixed(1)), s = UL.range('bi-s', () => draw(), (v) => v.toFixed(1));
    function draw() {
      P.clear(); P.domain(-5, 20, -0.05, 1.08); P.axes({ xlabel: 'weighted sum (before the bias)' });
      const pts = [], p0 = []; for (let x = -5; x <= 20.001; x += 0.1) { pts.push([x, UL.sigmoid(x + b.v)]); p0.push([x, UL.sigmoid(x)]); }
      P.clipped(() => { P.poly(p0, { color: UL.GREY, w: 1.5, dash: [4, 4] }); P.poly(pts, { color: UL.INK, w: 2.4 }); });
      P.line(-b.v, -0.05, -b.v, 1.08, { color: UL.COLORS[1], dash: [3, 3], w: 1.2 });
      P.text('fires above ' + UL.fmt(-b.v, 1), -b.v, 1.02, { color: UL.COLORS[1], align: 'left', font: '11px "IBM Plex Mono", monospace' });
      const a = UL.sigmoid(s.v + b.v); P.marker(s.v, a, { r: 6, fill: UL.COLORS[0] });
      calc.innerHTML = `<span class="k">Weighted sum</span><span>${s.v.toFixed(1)}</span><span class="k">+ bias</span><span>${UL.fmt(b.v, 1)}</span><span class="k">z</span><span>${UL.fmt(s.v + b.v, 1)}</span><span class="k">σ(z)</span><span class="hl">${a.toFixed(3)}</span><span class="k">Neuron</span><span>${a > 0.5 ? '<b>active</b>' : 'inactive'}</span><span class="k">Grey dashes</span><span>the same neuron with no bias</span>`;
    }
    draw();
  })();

  /* ------------------------------------------------------------ 07 count */
  (function () {
    const inp = $('#ct-in'), tb = $('#ct-table'), foot = $('#ct-foot'), n = (v) => v.toLocaleString('en-US');
    function upd() {
      const sz = inp.value.split(/[ ,→\-]+/).map((x) => parseInt(x, 10)).filter((x) => x > 0);
      if (sz.length < 2) { tb.innerHTML = '<tr><td>Give at least two layer sizes.</td></tr>'; foot.textContent = ''; return; }
      let rows = '', tw = 0, tb2 = 0;
      for (let l = 0; l < sz.length - 1; l++) { const w = sz[l] * sz[l + 1], b = sz[l + 1]; tw += w; tb2 += b; rows += `<tr><td>${n(sz[l])} → ${n(sz[l + 1])}</td><td>${n(sz[l])} × ${n(sz[l + 1])} = ${n(w)}</td><td>${n(b)}</td><td>${n(w + b)}</td></tr>`; }
      tb.innerHTML = `<thead><tr><th>Connection</th><th>Weights</th><th>Biases</th><th>Total</th></tr></thead><tbody>${rows}<tr style="font-weight:600"><td>Whole network</td><td>${n(tw)}</td><td>${n(tb2)}</td><td>${n(tw + tb2)}</td></tr></tbody>`;
      foot.innerHTML = `${n(tw + tb2)} knobs to turn. At one second per knob, setting them by hand would take ${(tw + tb2) / 3600 > 1 ? n(Math.round((tw + tb2) / 3600)) + ' hours' : n(Math.round((tw + tb2) / 60)) + ' minutes'}, and you would still not know which values are right.`;
    }
    inp.addEventListener('input', upd); $$('[data-p]').forEach((b) => b.addEventListener('click', () => { inp.value = b.dataset.p; upd(); })); upd();
  })();

  /* ------------------------------------------------------------ 08 matrix */
  (function () {
    const el = $('#mx'), calc = $('#mx-calc'); let r = UL.rng(5), W, a, b;
    const fresh = () => { W = Array.from({ length: 3 }, () => Array.from({ length: 4 }, () => Math.round(r.normal() * 10) / 10)); a = Array.from({ length: 4 }, () => Math.round(r() * 10) / 10); b = Array.from({ length: 3 }, () => Math.round(r.normal() * 5) / 10); };
    function draw(hi = -1) {
      const z = W.map((row, j) => row.reduce((s, w, i) => s + w * a[i], 0) + b[j]), act = z.map(UL.sigmoid);
      const m = (cells, cols, cls = '') => `<span class="mat" style="grid-template-columns:repeat(${cols},auto)">${cells.map((c) => `<span class="${c[1] ? 'hl' : ''}">${c[0]}</span>`).join('')}</span>`;
      el.innerHTML = `<span>σ(</span>${m(W.flat().map((w, k) => [UL.fmt(w, 1), Math.floor(k / 4) === hi]), 4)}<span>×</span>${m(a.map((v) => [v.toFixed(1), hi >= 0]), 1)}<span>+</span>${m(b.map((v, j) => [UL.fmt(v, 1), j === hi]), 1)}<span>) =</span>${m(act.map((v, j) => [v.toFixed(2), j === hi]), 1)}`;
      $$('.mat:last-child span', el).forEach((s, j) => { s.style.cursor = 'pointer'; s.addEventListener('mouseenter', () => draw(j)); });
      if (hi < 0) { calc.innerHTML = '<span class="k">W</span><span>3 × 4: one row per neuron, one column per input</span><span class="k">a⁽⁰⁾</span><span>the 4 input activations</span><span class="k">b</span><span>one bias per neuron</span><span class="k">Result</span><span>3 activations, all at once</span>'; return; }
      calc.innerHTML = `<span class="k">Neuron ${hi + 1}</span><span>${W[hi].map((w, i) => `${UL.fmt(w, 1)}·${a[i].toFixed(1)}`).join(' + ')} ${b[hi] < 0 ? '−' : '+'} ${Math.abs(b[hi]).toFixed(1)}</span><span class="k">z</span><span>${UL.fmt(z[hi], 2)}</span><span class="k">σ(z)</span><span class="hl">${act[hi].toFixed(2)}</span><span class="k">In words</span><span>row ${hi + 1} of W times the input column, plus bias ${hi + 1}: one neuron’s weighted sum.</span>`;
    }
    $('#mx-new').addEventListener('click', () => { fresh(); draw(); });
    el.addEventListener('mouseleave', () => draw()); fresh(); draw();
  })();

  /* ------------------------------------------------------------ 09 non-linearity */
  (function () {
    const P = new UL.Plot('#nl-plot', { w: 760, h: 300, pad: { l: 36, r: 14, t: 14, b: 30 } }), st = $('#nl-status');
    let tk = 'sin', ak = 'linear', net, it = 0;
    const T = { sin: (x) => Math.sin(2 * x) * 0.8, abs: (x) => Math.abs(x) * 0.6 - 0.6, step: (x) => (x > 0.3 ? 0.7 : -0.7) };
    const xs = Array.from({ length: 41 }, (_, i) => -2.5 + i * 0.125);
    const reset = () => { net = new UL.MLP([1, 8, 8, 1], { act: ak, out: 'linear', seed: 3, scale: ak === 'relu' ? 0.9 : 0.8 }); it = 0; draw(); };
    function draw() {
      P.clear(); P.domain(-2.6, 2.6, -1.4, 1.4); P.axes({ xlabel: 'input x' });
      xs.forEach((x) => P.marker(x, T[tk](x), { r: 3.2, fill: UL.COLORS[1] }));
      const pts = []; let c = 0; for (let x = -2.6; x <= 2.6; x += 0.02) pts.push([x, net.forward([x])[0]]);
      xs.forEach((x) => { const o = net.forward([x])[0]; c += (o - T[tk](x)) ** 2; });
      P.clipped(() => P.poly(pts, { color: UL.COLORS[0], w: 2.6 }));
      st.innerHTML = `step <b>${it}</b> · cost <b>${(c / xs.length).toFixed(3)}</b>`;
    }
    const loop = UL.loop(() => {
      for (let s = 0; s < 40; s++) { xs.forEach((x) => { net.forward([x]); net.backward([T[tk](x)]); }); net.step(ak === 'sigmoid' ? 0.3 : 0.03, xs.length, { momentum: 0.9 }); it++; }
      draw(); if (it >= 4000) { $('#nl-go').textContent = 'Train'; return false; }
    });
    $('#nl-go').addEventListener('click', () => { if (loop.on) { loop.stop(); $('#nl-go').textContent = 'Train'; } else { loop.start(); $('#nl-go').textContent = 'Pause'; } });
    $('#nl-reset').addEventListener('click', () => { loop.stop(); $('#nl-go').textContent = 'Train'; reset(); });
    UL.seg('nl-t', (v) => { tk = v; draw(); }); UL.seg('nl-a', (v) => { ak = v; loop.stop(); $('#nl-go').textContent = 'Train'; reset(); });
    reset();
  })();

  /* ------------------------------------------------------------ 10 exercise 1 */
  (function () {
    const words = [['prize', 2.0], ['money', 1.5], ['dear', -1.0], ['win', 1.0]], on = new Set(['money', 'dear']), b = -1.5, el = $('#sp-words'), calc = $('#sp-calc');
    el.innerHTML = words.map(([w, v]) => `<label data-w="${w}">${w} <small>${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</small></label>`).join('') + `<span class="pill">bias b = −1.5</span>`;
    function upd() {
      $$('label', el).forEach((l) => l.classList.toggle('on', on.has(l.dataset.w)));
      const terms = words.map(([w, v]) => [w, on.has(w) ? 1 : 0, v]), s = terms.reduce((t, [, x, v]) => t + x * v, 0) + b, a = UL.sigmoid(s);
      calc.innerHTML = `<span class="k">Inputs</span><span>${terms.map(([w, x]) => `${w}=${x}`).join(', ')}</span><span class="k">Weighted sum + b</span><span>${terms.filter(([, x]) => x).map(([, , v]) => UL.fmt(v, 1)).join(' + ') || '0'} − 1.5 = <b>${UL.fmt(s, 1)}</b></span><span class="k">σ</span><span class="hl" style="font-size:1.1rem">${a.toFixed(2)}</span><span class="k">Verdict</span><span>${a > 0.5 + 1e-9 ? '<b class="neg">spam</b>' : Math.abs(a - 0.5) < 1e-9 ? '<b>exactly on the edge (0.50)</b>' : '<b class="pos">not spam</b>'}</span>`;
    }
    $$('label', el).forEach((l) => l.addEventListener('click', () => { const w = l.dataset.w; if (on.has(w)) on.delete(w); else on.add(w); upd(); }));
    upd();
    UL.exercises('ex1-list', [
      { q: 'An email contains only “money” and “dear”. Compute the weighted sum and σ. Is it spam if the rule is σ ≥ 0.5?', a: '1.5 − 1.0 − 1.5 = −1.0; σ(−1) = 0.27 &lt; 0.5 → <b>not spam</b>.' },
      { q: 'Which one word would you add to make it spam? What is the new σ?', a: 'Add “prize”: the sum becomes 1.0 and σ = 0.73 → spam. (Adding “win” gives a sum of 0 and σ = 0.50: exactly on the edge.)' },
      { q: 'Count the parameters of a 784 → 16 → 16 → 10 network without looking back. And of 784 → 32 → 10?', a: '12,560 + 272 + 170 = <b>13,002</b>. For 784 → 32 → 10: 25,120 + 330 = <b>25,450</b>.' },
      { q: 'In the edge neuron, a pixel has weight 0. What does that mean?', a: 'That pixel is ignored by this neuron: whatever its brightness, it adds 0 to the sum.' },
      { q: 'You remove every activation function from a 5-layer network. What kind of model is left?', a: 'A single linear model: straight lines on straight lines are still straight. (Try “none” in § 09.)' },
    ]);
  })();

  UL.quiz('qz', [
    { q: 'What does a neuron in the input layer of the digit network hold?', o: ['A weight', 'The brightness of one pixel, between 0 and 1', 'A digit from 0 to 9', 'The cost'], a: 1, why: 'Each of the 784 input neurons simply holds one pixel’s brightness: its activation.' },
    { q: 'ReLU(−3) = ?', o: ['−3', '0', '3', '0.05'], a: 1, why: 'ReLU keeps positive numbers and turns negative ones into 0.' },
    { q: 'σ(0) = ?', o: ['0', '0.25', '0.5', '1'], a: 2, why: '1 / (1 + e⁰) = 1 / 2.' },
    { q: 'A neuron has bias −10. What does that mean?', o: ['It is switched off', 'Its weighted sum must be above about 10 before it becomes active', 'All its weights are −10', 'It fires 10 times'], a: 1, why: 'The bias is added before σ: σ(sum − 10) is above 0.5 only when the sum is above 10.' },
    { q: 'How many weights connect a layer of 784 neurons to a layer of 16?', o: ['800', '784 + 16', '12,544', '13,002'], a: 2, why: 'Every one of the 16 neurons has one weight per input: 784 × 16 = 12,544. (13,002 is the whole network, biases included.)' },
    { q: 'In a = σ(Wa + b), what is row k of W?', o: ['The activations of layer k', 'All the weights going into neuron k', 'The biases', 'The output'], a: 1, why: 'Multiplying row k by the input column gives neuron k’s weighted sum.' },
    { q: 'Why do networks need activation functions between layers?', o: ['To make them faster', 'Without them, all layers together are still just one linear model', 'To count the parameters', 'To store the pixels'], a: 1, why: 'Linear functions of linear functions are linear. The squish is what lets depth build curved, complex functions.' },
  ]);
  UL.start();
})();
