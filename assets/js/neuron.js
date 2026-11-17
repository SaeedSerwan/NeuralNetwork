/* Part 1 · Neurons & layers
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$, S = UL.samples();
  const f2 = (v) => UL.fmt(v, 2);

  /* ------------------------------------------------------------ 01 pixels */
  (function () {
    if (!document.getElementById('px-grid')) return;
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
    if (!document.getElementById('net-svg')) return;
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
    if (!document.getElementById('hope-svg')) return;
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

  /* ------------------------------------------------------------ 04a one neuron, animated */
  (function () {
    if (!document.getElementById('on-net')) return;
    const words = [['prize', 2.0], ['money', 1.5], ['dear', -1.0], ['win', 1.0]], b = -1.5, on = new Set(['money', 'prize']);
    const el = $('#on-words');
    el.innerHTML = words.map(([w, v]) => `<label data-w="${w}">${w} <small>w = ${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</small></label>`).join('') + '<span class="pill">bias b = −1.5</span>';
    const A = new UL.NetAnim('on-net', [{ n: 4, label: 'inputs (words)', names: words.map((w) => w[0]), gap: 62 }, { n: 1, label: 'neuron', names: [''] }], { w: 480, h: 300, r: 20, left: 90, right: 150, labelAt: 0.42 });
    const SP = new UL.Plot('#on-sig', { w: 360, h: 150, pad: { l: 30, r: 8, t: 8, b: 22 } });
    let x, terms, sum, z, a, zShown = null;
    const calc = () => { x = words.map(([w]) => (on.has(w) ? 1 : 0)); terms = words.map(([, w], i) => w * x[i]); sum = terms.reduce((p, q) => p + q, 0); z = sum + b; a = UL.sigmoid(z); };
    function sigPlot(zz) {
      SP.clear(); SP.domain(-5, 5, -0.05, 1.05); SP.axes({ nx: 5, ny: 2 });
      const pts = []; for (let t = -5; t <= 5; t += 0.05) pts.push([t, UL.sigmoid(t)]); SP.poly(pts, { color: UL.INK, w: 2 });
      SP.line(-5, 0.5, 5, 0.5, { color: UL.FAINT, dash: [3, 3], w: 1 });
      if (zz != null) { SP.line(zz, 0, zz, UL.sigmoid(zz), { color: UL.COLORS[0], dash: [3, 3], w: 1.2 }); SP.marker(zz, UL.sigmoid(zz), { r: 6, fill: UL.COLORS[0] }); SP.text(`σ(${UL.fmt(zz, 1)}) = ${UL.sigmoid(zz).toFixed(2)}`, zz, Math.min(0.95, UL.sigmoid(zz) + 0.12), { color: UL.COLORS[0], font: '11px "IBM Plex Mono", monospace', align: zz > 1 ? 'right' : 'left' }); }
    }
    const base = () => { calc(); A.setWeights([[words.map((w) => w[1])]], { max: 2 }); A.setLayer(0, x.map(() => null)); A.setLayer(1, [null]); A.clearEdgeLabels(); A.sub(1, 0, ''); A.nodes[1][0].side.textContent = ''; A.clearRings(); zShown = null; sigPlot(null); A.edgeLabels(0, (i) => ({ text: 'w=' + UL.fmt(words[i][1], 1), color: words[i][1] >= 0 ? '#1b8c5c' : '#d64045' })); };
    const steps = [
      { title: 'Read the inputs', text: () => `Each word becomes a number: 1 if it is in the email, 0 if not. This email: ${words.map(([w], i) => `${w} = <b>${x[i]}</b>`).join(', ')}.`,
        run: async () => { await A.fillLayer(0, x, { texts: x.map(String), ms: 600 }); }, show: () => A.setLayer(0, x, { texts: x.map(String) }) },
      { title: 'Multiply each input by its weight', text: () => `Each input travels along its connection and is multiplied by the weight: ${words.map(([w, v], i) => `${x[i]} × ${UL.fmt(v, 1)} = <b>${UL.fmt(terms[i], 1)}</b>`).join(', ')}. Absent words send 0.`,
        run: async () => { await A.pulse(0, 'fwd', { size: (i) => Math.abs(terms[i]) / 2, color: (i) => (terms[i] >= 0 ? '#1b8c5c' : '#d64045'), ms: 1200 }); }, show: () => A.edgeLabels(0, (i) => ({ text: `${x[i]}×${UL.fmt(words[i][1], 1)} = ${UL.fmt(terms[i], 1)}`, color: terms[i] === 0 ? '#a3a3a0' : terms[i] > 0 ? '#1b8c5c' : '#d64045' })) },
      { title: 'Add them up', text: () => `The neuron adds what arrives: ${terms.map((t) => UL.fmt(t, 1)).join(' + ')} = <b>${UL.fmt(sum, 1)}</b>. This is the <b>weighted sum</b>.`,
        run: async () => { await A.clock.tween(700, (t) => A.setNode(1, 0, 0, { text: UL.fmt(sum * t, 1) })); }, show: () => { A.setNode(1, 0, 0, { text: UL.fmt(sum, 1) }); A.sub(1, 0, 'sum = ' + UL.fmt(sum, 1)); } },
      { title: 'Add the bias', text: () => `Then the bias: ${UL.fmt(sum, 1)} + (−1.5) = <b>z = ${UL.fmt(z, 1)}</b>. The bias says how much evidence is needed before the neuron fires.`,
        run: async () => { await A.clock.tween(600, (t) => A.setNode(1, 0, 0, { text: UL.fmt(sum + b * t, 1) })); }, show: () => { A.setNode(1, 0, 0, { text: UL.fmt(z, 1) }); A.sub(1, 0, 'z = ' + UL.fmt(z, 1)); } },
      { title: 'Squish with the sigmoid', text: () => `The activation function σ turns z into a number between 0 and 1: σ(${UL.fmt(z, 1)}) = <b>${a.toFixed(2)}</b>. Watch the point slide along the curve.`,
        run: async () => { await A.clock.tween(900, (t) => sigPlot(z * t)); await A.fillLayer(1, [a], { texts: [a.toFixed(2)] }); }, show: () => { sigPlot(z); A.setNode(1, 0, a, { text: a.toFixed(2) }); } },
      { title: 'The decision', text: () => `The output is ${a.toFixed(2)}. With the rule “spam if ≥ 0.5” this email is <b>${a >= 0.5 - 1e-9 ? (Math.abs(a - 0.5) < 1e-9 ? 'exactly on the edge' : 'SPAM') : 'NOT spam'}</b>. Change the words above and play again.`,
        run: async () => { await A.flash(1, 0, a >= 0.5 ? '#d64045' : '#1b8c5c', 700); }, show: () => { A.ring(1, 0, a >= 0.5 ? '#d64045' : '#1b8c5c'); A.nodes[1][0].side.textContent = a >= 0.5 ? '→ spam' : '→ not spam'; } },
    ];
    calc();
    const code = new UL.CodePanel('on-code', [
      'x = np.array([' + words.map(([w]) => (on.has(w) ? 1 : 0)).join(', ') + '])   # ' + words.map((w) => w[0]).join(', '),
      'w = np.array([2.0, 1.5, -1.0, 1.0])',
      'b = -1.5',
      'products = w * x          # multiply each input by its weight',
      's = products.sum()        # add them up',
      'z = s + b                 # add the bias',
      'a = 1 / (1 + np.exp(-z))  # sigmoid',
      'spam = a >= 0.5',
    ], { title: 'The same neuron in Python (NumPy)' });
    const fx = (v) => (Number.isInteger(v) ? v + '.' : String(+v.toFixed(2)));
    [[0, 0], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]].forEach(([k, l]) => { steps[k].line = l; });
    steps[0].notes = () => ({ 0: `shape (4,): [${x.join(', ')}]` });
    steps[1].notes = () => ({ 3: `[${terms.map(fx).join(' ')}]` });
    steps[2].notes = () => ({ 4: fx(sum) });
    steps[3].notes = () => ({ 5: fx(z) });
    steps[4].notes = () => ({ 6: a.toFixed(2) });
    steps[5].notes = () => ({ 7: String(a >= 0.5) });
    const P = new UL.Player({ code, controls: 'on-ctl', narration: 'on-narr', progress: 'on-bar', clock: A.clock, steps, base, pause: 900, intro: 'This email contains “prize” and “money”. Press <b>▶ Play</b> to watch the neuron decide.' });
    const chips = () => $$('label', el).forEach((l) => l.classList.toggle('on', on.has(l.dataset.w)));
    $$('label', el).forEach((l) => l.addEventListener('click', () => { const w = l.dataset.w; if (on.has(w)) on.delete(w); else on.add(w); chips(); code.rows[0].querySelector('code').innerHTML = UL.pyHL('x = np.array([' + words.map(([q]) => (on.has(q) ? 1 : 0)).join(', ') + '])   # ' + words.map((q) => q[0]).join(', ')); P.pause(); P.goto(-1); }));
    chips();
  })();

  /* ------------------------------------------------------------ 04 weighted sum */
  (function () {
    if (!document.getElementById('ws-a')) return;
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
    let scan = -1, wk = 'h', hover = -1, idx = TH.reduce((best, i) => { const sc = (j) => S.get(j).reduce((t, v, k) => t + v * W.h[k], 0); return sc(i) > sc(best) ? i : best; }, TH[0]);
    const bias = UL.range('ws-b', () => draw());
    function draw() {
      const a = S.get(idx), w = W[wk], sc = wk === 'learned' ? 1.2 : 1, prod = new Float32Array(784); let sum = 0, pos = 0, neg = 0;
      for (let i = 0; i < 784; i++) { prod[i] = w[i] * a[i]; sum += prod[i]; if (prod[i] > 0) pos += prod[i]; else neg += prod[i]; }
      const shown = scan < 0 ? prod : prod.map((v, i) => (Math.floor(i / 28) <= scan ? v : 0)); let run = 0; if (scan >= 0) for (let i = 0; i < (scan + 1) * 28; i++) run += prod[i];
      UL.drawDigit(PA.ctx, a, 0, 0, 224); UL.drawDigit(PW.ctx, w, 0, 0, 224, { signed: true, scale: sc }); PP.clear(); UL.drawDigit(PP.ctx, shown, 0, 0, 224, { signed: true, scale: sc });
      if (scan >= 0) [PA, PW, PP].forEach((P) => { P.ctx.fillStyle = 'rgba(235,104,52,.25)'; P.ctx.fillRect(0, scan * 8, 224, 8); P.ctx.strokeStyle = '#eb6834'; P.ctx.lineWidth = 1.5; P.ctx.strokeRect(0.5, scan * 8 + 0.5, 223, 7); });
      if (hover >= 0) [PA, PW, PP].forEach((P) => { P.ctx.strokeStyle = '#eb6834'; P.ctx.lineWidth = 2; P.ctx.strokeRect((hover % 28) * 8, Math.floor(hover / 28) * 8, 8, 8); });
      const b = bias.v, z = sum + b, act = UL.sigmoid(z);
      if (scan >= 0) { calc.innerHTML = `<span class="k">Row</span><span>${scan + 1} of 28: multiply its 28 pixels by their weights</span><span class="k">Running sum</span><span class="hl" style="font-size:1.2rem">${UL.fmt(run, 1)}</span><span class="k">Still to add</span><span>${27 - scan} rows</span>`; return; }
      calc.innerHTML = (hover >= 0 ? `<span class="k">This pixel</span><span>a = ${a[hover].toFixed(2)}, w = ${f2(w[hover])}, w × a = <b>${f2(prod[hover])}</b></span>` : '') +
        `<span class="k">Sum of green products</span><span class="pos">+${pos.toFixed(1)}</span><span class="k">Sum of red products</span><span class="neg">${UL.fmt(neg, 1)}</span>` +
        `<span class="k">Weighted sum</span><span class="hl">${UL.fmt(sum, 1)}</span><span class="k">+ bias</span><span>${UL.fmt(b, 0)} → z = ${UL.fmt(z, 1)}</span>` +
        `<span class="k">Activation σ(z)</span><span class="hl" style="font-size:1.1rem">${act.toFixed(2)} ${act > 0.5 ? '· the neuron is <b>on</b>' : '· the neuron stays <b>off</b>'}</span>`;
    }
    UL.seg('ws-w', (v) => { wk = v; if (v === 'learned') bias.set(0); else if (bias.v === 0) bias.set(-10); draw(); });
    UL.thumbs('ws-thumbs', TH, (i) => { idx = i; draw(); }, { select: TH.indexOf(idx) });
    let tacc = 0; const sl = UL.loop((dt) => { tacc += dt; if (tacc > 0.09) { tacc = 0; scan++; if (scan > 27) { scan = -1; draw(); $('#ws-anim').textContent = '▶ Animate the sum'; return false; } draw(); } });
    $('#ws-anim').addEventListener('click', () => { if (sl.on) { sl.stop(); scan = -1; draw(); $('#ws-anim').textContent = '▶ Animate the sum'; } else { scan = -1; hover = -1; sl.start(); $('#ws-anim').textContent = '■ Stop'; } });
    PP.cv.addEventListener('mousemove', (e) => { if (scan >= 0) return; const [X, Y] = PP.pointer(e); hover = Math.floor(Y / 8) * 28 + Math.floor(X / 8); draw(); });
    PP.cv.addEventListener('mouseleave', () => { hover = -1; draw(); });
    draw();
  })();

  /* ------------------------------------------------------------ 05 activation functions */
  (function () {
    if (!document.getElementById('sg-plot')) return;
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
    if (!document.getElementById('bi-plot')) return;
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
    if (!document.getElementById('ct-in')) return;
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
    if (!document.getElementById('mx')) return;
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

  /* ------------------------------------------------------------ 08b array sizes through the network, with code */
  (function () {
    if (!document.getElementById('sz-mm')) return;
    const clock = new UL.Clock(), NS = 'http://www.w3.org/2000/svg';
    let sizes = [784, 16, 16, 10], B = 32, player = null;
    const sc = (n) => 8 + 17 * Math.log2(Math.max(1, n)); // block length on a log scale
    const nf = (v) => v.toLocaleString('en-US');
    const code = new UL.CodePanel('sz-code', [''], { title: 'The forward pass in Python (NumPy)' });
    const strip = document.getElementById('sz-strip'), mm = document.getElementById('sz-mm');
    function codeLines() {
      const L = sizes.length - 1, out = [`X = images[:${B}]`.padEnd(34) + `# (${B}, ${sizes[0]})`];
      out[0] = `A0 = images[:B]`.padEnd(34) + `# B pictures, ${sizes[0]} pixels each`;
      for (let l = 1; l <= L; l++) {
        out.push(`Z${l} = A${l - 1} @ W${l}.T`.padEnd(34) + `# W${l} has shape (${sizes[l]}, ${sizes[l - 1]})`);
        out.push(`A${l} = sigmoid(Z${l} + b${l})`.padEnd(34) + `# b${l} has shape (${sizes[l]},)`);
      }
      out.push(`pred = A${L}.argmax(axis=1)`.padEnd(34) + '# the brightest output per picture');
      return out;
    }
    function drawStrip(cur) { // every activation array side by side, the current one highlighted
      const L = sizes.length, W = 760, H = 100, bx = (n) => Math.max(10, sc(n) * 0.45), y0 = 4;
      let x = 10, s = '';
      sizes.forEach((n, l) => {
        const w = bx(n), h = Math.max(8, sc(B) * 0.45), on = l === cur;
        s += `<rect x="${x}" y="${y0 + 45 - h / 2}" width="${w}" height="${h}" rx="3" fill="${on ? '#2a78d6' : '#e8f0fb'}" stroke="#2a78d6"/>`;
        s += `<text x="${x + w / 2}" y="${y0 + 45 + h / 2 + 16}" text-anchor="middle" style="font:11px var(--mono);fill:${on ? '#111' : '#6b6b68'}">A${l} (${B}, ${n})</text>`;
        if (l < L - 1) { const x2 = x + w + 112; s += `<line x1="${x + w + 6}" y1="${y0 + 45}" x2="${x2 - 8}" y2="${y0 + 45}" stroke="#a3a3a0" stroke-width="1.5" marker-end="url(#szarrow)"/><text x="${(x + w + x2) / 2}" y="${y0 + 36}" text-anchor="middle" style="font:10.5px var(--mono);fill:${cur === l + 1 ? '#eb6834' : '#6b6b68'}">W${l + 1} (${sizes[l + 1]}, ${n})</text>`; x = x2; }
      });
      strip.innerHTML = `<svg class="netsvg" viewBox="0 0 ${Math.max(W, x + 60)} ${H}"><defs><marker id="szarrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#a3a3a0"/></marker></defs>${s}</svg>`;
    }
    // the matrix multiplication of one layer: A (B × nIn) @ W.T (nIn × nOut) = Z (B × nOut), + b, σ
    function drawMM(l, phase, t) {
      if (l < 1) { mm.innerHTML = ''; return; }
      const nIn = sizes[l - 1], nOut = sizes[l], h = (n) => sc(n) * 0.85, w = (n) => sc(n) * 0.85;
      const aW = w(nIn), aH = h(B), wW = w(nOut), wH = h(nIn), zW = w(nOut), zH = h(B), top = 30, gap = 34;
      const ax = 10, wx = ax + aW + gap, zx = wx + wW + gap, bx = zx + zW + gap, H = top + Math.max(aH, wH) + 50;
      const rowsDone = phase === 0 ? t : 1, sweep = phase === 0 ? t : -1;
      let s = '';
      s += `<rect x="${ax}" y="${top}" width="${aW}" height="${aH}" rx="3" fill="#e8f0fb" stroke="#2a78d6"/>`;
      s += `<rect x="${wx}" y="${top}" width="${wW}" height="${wH}" rx="3" fill="#eaf5ef" stroke="#1b8c5c"/>`;
      s += `<rect x="${zx}" y="${top}" width="${zW}" height="${zH}" rx="3" fill="#fff" stroke="#111" stroke-dasharray="3 3"/>`;
      s += `<rect x="${zx}" y="${top}" width="${zW}" height="${zH * rowsDone}" rx="3" fill="${phase >= 1 ? '#2a78d6' : '#9ec2ee'}" opacity="${phase >= 1 ? 0.25 + 0.65 * (phase === 1 ? t : 1) : 0.9}"/>`;
      if (sweep >= 0 && sweep < 1) { const ry = top + aH * sweep; s += `<rect x="${ax}" y="${ry}" width="${aW}" height="${Math.max(3, aH / Math.max(4, Math.min(B, 24)))}" fill="#eb6834" opacity=".55"/><rect x="${wx}" y="${top}" width="${Math.max(3, wW / Math.max(4, Math.min(nOut, 24)))}" height="${wH}" fill="#eb6834" opacity=".45" transform="translate(${(wW - Math.max(3, wW / Math.max(4, Math.min(nOut, 24)))) * ((sweep * 7) % 1)},0)"/>`; }
      if (phase >= 1) { s += `<rect x="${bx}" y="${top}" width="${zW}" height="10" rx="2" fill="#fff4d6" stroke="#c99a2e"/><text x="${bx + zW / 2}" y="${top + 26}" text-anchor="middle" style="font:10.5px var(--mono);fill:#8a6a1d">+ b${l} (${nOut},)</text>`; if (phase === 1) for (let r = 0; r < 4; r++) s += `<rect x="${zx}" y="${top + (zH * r) / 4 + 2}" width="${zW}" height="6" fill="#fff4d6" opacity="${Math.sin(t * Math.PI) * 0.9}"/>`; }
      const lab = (x, wd, y, t1, t2, col) => `<text x="${x + wd / 2}" y="${y}" text-anchor="middle" style="font:500 11.5px var(--mono);fill:${col}">${t1}</text><text x="${x + wd / 2}" y="${y + 15}" text-anchor="middle" style="font:11px var(--mono);fill:#6b6b68">${t2}</text>`;
      s += lab(ax, aW, top + aH + 18, `A${l - 1}`, `(${B}, ${nIn})`, '#2a78d6') + lab(wx, wW, top + wH + 18, `W${l}.T`, `(${nIn}, ${nOut})`, '#1b8c5c') + lab(zx, zW, top + zH + 18, phase >= 1 ? `A${l}` : `Z${l}`, `(${B}, ${nOut})`, '#111');
      s += `<text x="${ax + aW + gap / 2}" y="${top + 18}" text-anchor="middle" style="font:500 16px var(--mono);fill:#3d3d3b">@</text><text x="${wx + wW + gap / 2}" y="${top + 18}" text-anchor="middle" style="font:500 16px var(--mono);fill:#3d3d3b">=</text>`;
      s += `<text x="${ax}" y="16" style="font:11px var(--mono);fill:#eb6834">inner sizes: ${nIn} and ${nIn} ✓ → result keeps the outer sizes (${B}, ${nOut})</text>`;
      mm.innerHTML = `<svg class="netsvg" viewBox="0 0 ${Math.max(560, bx + zW + 20)} ${H + 10}">${s}</svg>`;
    }
    function build() {
      clock.cancel(); const L = sizes.length - 1;
      code.set(codeLines(), 'The forward pass in Python (NumPy)');
      const steps = [{ title: 'A batch of pictures', line: 0, text: () => `We push <b>B = ${B}</b> pictures through at once. Each picture is a row of ${sizes[0]} numbers, so the input is a table of shape <b>(${B}, ${sizes[0]})</b>: ${nf(B * sizes[0])} numbers.`,
        notes: () => ({ 0: `shape (${B}, ${sizes[0]})` }), run: async () => { drawStrip(0); drawMM(0); await clock.wait(500); }, show: () => { drawStrip(0); drawMM(0); } }];
      for (let l = 1; l <= L; l++) {
        const nIn = sizes[l - 1], nOut = sizes[l];
        steps.push({ title: `Layer ${l}: multiply by the weights`, line: 2 * l - 1, text: () => `<code>A${l - 1} @ W${l}.T</code>: (${B}, ${nIn}) @ (${nIn}, ${nOut}) → <b>(${B}, ${nOut})</b>. Each of the ${nf(B * nOut)} results is one weighted sum of ${nf(nIn)} products: ${nf(B * nIn * nOut)} multiply-adds in total. The ${nIn} disappears; ${nOut} neurons remain.`,
          notes: () => ({ [2 * l - 1]: `(${B}, ${nOut})` }), run: async () => { drawStrip(l - 1); await clock.tween(1600, (t, raw) => drawMM(l, 0, raw)); }, show: () => { drawStrip(l); drawMM(l, 0, 1); } });
        steps.push({ title: `Layer ${l}: add the bias, squish`, line: 2 * l, text: () => `<code>+ b${l}</code> has shape (${nOut},): one bias per neuron, added to every one of the ${B} rows (broadcasting). Then σ is applied to every entry. The shape stays <b>(${B}, ${nOut})</b>. This layer has ${nf(nIn * nOut)} weights + ${nOut} biases = ${nf(nIn * nOut + nOut)} parameters.`,
          notes: () => ({ [2 * l]: `(${B}, ${nOut})` }), run: async () => { await clock.tween(900, (t, raw) => drawMM(l, 1, raw)); drawStrip(l); }, show: () => { drawStrip(l); drawMM(l, 2, 1); } });
      }
      steps.push({ title: 'The answers', line: 2 * L + 1, text: () => `<code>argmax(axis=1)</code> picks the brightest of the ${sizes[L]} outputs in each row: shape <b>(${B},)</b>, one answer per picture. All together: ${nf(sizes.slice(1).reduce((s, n, i) => s + sizes[i] * n + n, 0))} parameters, the same for every batch size.`,
        notes: () => ({ [2 * L + 1]: `(${B},)` }), run: async () => { drawStrip(L); await clock.wait(400); }, show: () => drawStrip(L) });
      const ctl = document.getElementById('sz-ctl'); ctl.innerHTML = '';
      player = new UL.Player({ code, clock, controls: 'sz-ctl', narration: 'sz-narr', progress: 'sz-bar', steps, base: () => { drawStrip(0); drawMM(0); }, pause: 900,
        intro: 'Press <b>▶ Play</b> to follow a batch of pictures through the network, one line of code at a time.' });
    }
    document.getElementById('sz-in').addEventListener('change', (e) => { const v = e.target.value.split(/[ ,]+/).map((x) => parseInt(x, 10)).filter((x) => x > 0); if (v.length >= 2) { sizes = v.slice(0, 6); build(); } });
    UL.seg('sz-b', (v) => { B = +v; build(); });
    build();
  })();

  /* ------------------------------------------------------------ 09 non-linearity */
  (function () {
    if (!document.getElementById('nl-plot')) return;
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
    if (!document.getElementById('sp-words')) return;
    const words = [['prize', 2.0], ['money', 1.5], ['dear', -1.0], ['win', 1.0]], on = new Set(['money', 'dear']), b = -1.5, el = $('#sp-words'), calc = $('#sp-calc');
    el.innerHTML = words.map(([w, v]) => `<label data-w="${w}">${w} <small>${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</small></label>`).join('') + `<span class="pill">bias b = −1.5</span>`;
    function upd() {
      $$('label', el).forEach((l) => l.classList.toggle('on', on.has(l.dataset.w)));
      const terms = words.map(([w, v]) => [w, on.has(w) ? 1 : 0, v]), s = terms.reduce((t, [, x, v]) => t + x * v, 0) + b, a = UL.sigmoid(s);
      calc.innerHTML = `<span class="k">Inputs</span><span>${terms.map(([w, x]) => `${w}=${x}`).join(', ')}</span><span class="k">Weighted sum + b</span><span>${terms.filter(([, x]) => x).map(([, , v]) => UL.fmt(v, 1)).join(' + ') || '0'} − 1.5 = <b>${UL.fmt(s, 1)}</b></span><span class="k">σ</span><span class="hl" style="font-size:1.1rem">${a.toFixed(2)}</span><span class="k">Verdict</span><span>${a > 0.5 + 1e-9 ? '<b class="neg">spam</b>' : Math.abs(a - 0.5) < 1e-9 ? '<b>exactly on the edge (0.50)</b>' : '<b class="pos">not spam</b>'}</span>`;
    }
    $$('label', el).forEach((l) => l.addEventListener('click', () => { const w = l.dataset.w; if (on.has(w)) on.delete(w); else on.add(w); upd(); }));
    upd();
    
  })();

  
  UL.start();
})();
