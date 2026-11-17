/* Overview: 40 threes, real vs artificial neuron, the learning loop with one weight
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$, S = UL.samples();

  /* ------------------------------------------------------------ 40 threes */
  (function () {
    const P = new UL.Plot('#th-grid', { w: 480, h: 300, pad: { l: 0, r: 0, t: 0, b: 0 } }), B = UL.pixPlot('#th-big', 280), cap = $('#th-cap'), cell = 60;
    let hi = 0;
    function draw() {
      P.ctx.fillStyle = '#fff'; P.ctx.fillRect(0, 0, 480, 300);
      for (let i = 0; i < 40; i++) { const x = (i % 8) * cell, y = Math.floor(i / 8) * cell; UL.drawDigit(P.ctx, S.get(i), x + 2, y + 2, cell - 4); if (i === hi) { P.ctx.strokeStyle = '#eb6834'; P.ctx.lineWidth = 3; P.ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3); } }
      const a = S.get(hi); UL.drawDigit(B.ctx, a, 0, 0, 280);
      B.ctx.font = '7px "IBM Plex Mono", monospace'; B.ctx.textAlign = 'center'; B.ctx.textBaseline = 'middle';
      for (let k = 0; k < 784; k++) if (a[k] > 0.05) { B.ctx.fillStyle = a[k] > 0.5 ? '#000' : '#fff'; B.ctx.fillText(a[k].toFixed(1).slice(1), (k % 28) * 10 + 5, Math.floor(k / 28) * 10 + 5); }
      cap.innerHTML = `<b>Three number ${hi + 1}</b>: ${Array.from(a).filter((v) => v > 0.05).length} of the 784 pixels are not black. A different 3 lights up different pixels.`;
    }
    P.cv.addEventListener('mousemove', (e) => { const [X, Y] = P.pointer(e), i = Math.floor(Y / cell) * 8 + Math.floor(X / cell); if (i >= 0 && i < 40 && i !== hi) { hi = i; draw(); } });
    draw();
  })();

  /* ------------------------------------------------------------ real vs artificial neuron */
  $$('#nm [data-p]').forEach((d) => {
    d.addEventListener('mouseenter', () => $$('#nm [data-p]').forEach((x) => x.classList.toggle('on', x.dataset.p === d.dataset.p)));
    d.addEventListener('mouseleave', () => $$('#nm [data-p]').forEach((x) => x.classList.remove('on')));
  });

  /* ------------------------------------------------------------ watch a network learn (full training cycle, animated) */
  (function () {
    const sig = UL.sigmoid, f2 = (v) => UL.fmt(v, 2), f3 = (v) => UL.fmt(v, 3);
    const DATA = [[[0.9, 0.8], 1, 'dark, motion'], [[0.9, 0.1], 0, 'dark, no motion'], [[0.1, 0.8], 0, 'daylight, motion'], [[0.1, 0.1], 0, 'daylight, still']];
    const eta = 3;
    let W1, b1, W2, b2, round, ex, R, hist;
    const init = () => { const r = UL.rng(8); W1 = [0, 1, 2].map(() => [r.normal() * 0.9, r.normal() * 0.9]); b1 = [0, 0, 0]; W2 = [[0, 1, 2].map(() => r.normal() * 0.9)]; b2 = [0]; round = 1; ex = 0; hist = []; R = compute(); };
    function forward(x) { const z1 = W1.map((w, j) => w[0] * x[0] + w[1] * x[1] + b1[j]), a1 = z1.map(sig), z2 = W2[0].reduce((s, w, j) => s + w * a1[j], 0) + b2[0]; return { z1, a1, z2, a2: sig(z2) }; }
    function compute() { // everything for the current round, from the current weights
      const [x, y] = DATA[ex], f = forward(x), C = (f.a2 - y) ** 2, d2 = 2 * (f.a2 - y) * f.a2 * (1 - f.a2);
      const gW2 = f.a1.map((a) => d2 * a), d1 = f.a1.map((a, j) => d2 * W2[0][j] * a * (1 - a)), gW1 = d1.map((d) => x.map((xi) => d * xi));
      const nW1 = W1.map((w, j) => w.map((v, i) => v - eta * gW1[j][i])), nb1 = b1.map((b, j) => b - eta * d1[j]), nW2 = [W2[0].map((v, j) => v - eta * gW2[j])], nb2 = [b2[0] - eta * d2];
      const keep = [W1, b1, W2, b2]; W1 = nW1; b1 = nb1; W2 = nW2; b2 = nb2; const Cafter = (forward(x).a2 - y) ** 2; [W1, b1, W2, b2] = keep;
      return { x, y, ...f, C, d2, gW2, d1, gW1, nW1, nb1, nW2, nb2, Cafter };
    }
    const avgCost = () => DATA.reduce((s, [x, y]) => s + (forward(x).a2 - y) ** 2, 0) / 4;
    const A = new UL.NetAnim('wl-net', [{ n: 2, label: 'sensors', names: ['darkness', 'motion'], gap: 130 }, { n: 3, label: 'hidden', gap: 85 }, { n: 1, label: 'light', names: ['on?'] }], { w: 600, h: 310, r: 21, left: 110, right: 120, labelAt: ['stagger', 0.5] });
    const loopBoxes = UL.$$('#lp-steps > div');
    const phase = (p) => loopBoxes.forEach((b, i) => b.classList.toggle('on', i === p));
    const LP = new UL.Plot('#wl-loss', { w: 400, h: 170, pad: { l: 40, r: 10, t: 10, b: 26 } });
    function side() {
      UL.$('#wl-table').innerHTML = `<thead><tr><th>case</th><th>want</th><th>network says</th></tr></thead><tbody>${DATA.map(([x, y, n], i) => { const p = forward(x).a2, ok = (p > 0.5 ? 1 : 0) === y; return `<tr style="${i === ex ? 'background:#fff4d6' : ''}"><td style="font-family:var(--font)">${n}</td><td>${y ? 'on' : 'off'}</td><td><span style="display:inline-block;width:${Math.round(p * 60)}px;height:9px;background:${ok ? '#1b8c5c' : '#d64045'};border-radius:2px;margin-right:6px;vertical-align:middle"></span>${p.toFixed(2)}</td></tr>`; }).join('')}</tbody>`;
      LP.clear(); LP.domain(0, Math.max(20, hist.length), 0, Math.max(0.3, ...hist) * 1.05); LP.axes({ ny: 3, nx: 4 });
      if (hist.length) LP.clipped(() => LP.poly(hist.map((h, i) => [i + 1, h]), { color: UL.COLORS[7], w: 2 }));
      UL.$('#wl-status').innerHTML = `round <b>${round}</b> · example ${ex + 1} of 4 · average cost <b>${avgCost().toFixed(3)}</b>`;
    }
    const W = () => [W1.map((w) => w.slice()), W2.map((w) => w.slice())];
    const base = () => { // the state at the start of a round: weights drawn, nothing computed yet
      A.setWeights([W1, W2.map((r) => r)], { max: 3 }); A.setLayer(0, [null, null]); A.setLayer(1, [null, null, null]); A.setLayer(2, [null]);
      A.clearEdgeLabels(); A.clearRings(); [0, 1, 2].forEach((j) => A.sub(1, j, '')); A.sub(2, 0, ''); A.dim(false); phase(-1); side();
    };
    const wlab = (l) => A.edgeLabels(l, (i, j) => { const w = l === 0 ? W1[j][i] : W2[0][i]; return { text: f2(w), color: w >= 0 ? '#1b8c5c' : '#d64045' }; });
    const glab = (l) => A.edgeLabels(l, (i, j) => { const g = l === 0 ? R.gW1[j][i] : R.gW2[i]; return { text: (g > 0 ? '+' : '') + UL.fmt(g, 3), color: '#eb6834' }; });
    const steps = [
      { title: 'An example comes in', text: () => `Example ${ex + 1}: <b>${DATA[ex][2]}</b>. The sensors read darkness = ${R.x[0]}, motion = ${R.x[1]}. The right answer is <b>${R.y ? 'light ON (1)' : 'light OFF (0)'}</b>, but the network does not know that yet.`,
        run: async () => { phase(0); await A.fillLayer(0, R.x, { texts: R.x.map(String) }); }, show: () => { phase(0); A.setLayer(0, R.x, { texts: R.x.map(String) }); wlab(0); wlab(1); } },
      { title: 'Forward: the hidden layer', text: () => `Each input travels along its line and is multiplied by the line’s weight (bigger dots = bigger product). Each hidden neuron adds what arrives, adds its bias and squishes with σ:<br>${R.z1.map((z, j) => `h${j + 1}: z = ${f2(z)} → σ = <b>${f2(R.a1[j])}</b>`).join('<br>')}`,
        run: async () => { phase(0); await A.pulse(0, 'fwd', { size: (i, j) => Math.abs(W1[j][i] * R.x[i]) / 2 }); await A.fillLayer(1, R.a1); }, show: () => { A.setLayer(1, R.a1); R.z1.forEach((z, j) => A.sub(1, j, 'z=' + f2(z))); } },
      { title: 'Forward: the output', text: () => `The same again, one layer further: z = ${R.a1.map((a, j) => `${f2(W2[0][j])}·${f2(a)}`).join(' + ')} + ${f2(b2[0])} = ${f2(R.z2)}, and σ(${f2(R.z2)}) = <b>${f2(R.a2)}</b>. The network’s guess: light ${R.a2 > 0.5 ? 'ON' : 'OFF'} (${f2(R.a2)}).`,
        run: async () => { await A.pulse(1, 'fwd', { size: (i) => Math.abs(W2[0][i] * R.a1[i]) / 2 }); await A.fillLayer(2, [R.a2]); }, show: () => { A.setLayer(2, [R.a2]); A.sub(2, 0, 'z=' + f2(R.z2)); } },
      { title: 'Measure: how wrong was it?', text: () => `Wanted ${R.y}, got ${f2(R.a2)}. Cost C = (a − y)² = (${f2(R.a2)} − ${R.y})² = <b>${f3(R.C)}</b>. ${R.C < 0.05 ? 'Small: nearly right.' : 'Too large: the weights must change.'}`,
        run: async () => { phase(1); await A.flash(2, 0, R.C < 0.05 ? '#1b8c5c' : '#d64045', 700); }, show: () => { phase(1); A.ring(2, 0, R.C < 0.05 ? '#1b8c5c' : '#d64045'); A.sub(2, 0, `want ${R.y} · C=${f3(R.C)}`, R.C < 0.05 ? '#1b8c5c' : '#d64045'); } },
      { title: 'Backward: blame the last weights', text: () => `Backpropagation starts at the output: ∂C/∂a = 2(a − y) = ${f3(2 * (R.a2 - R.y))}, times the sigmoid slope a(1 − a) gives δ = <b>${UL.fmt(R.d2, 4)}</b>. Each weight into the output gets the blame δ × the activation it carried. The orange numbers are these slopes ∂C/∂w.`,
        run: async () => { phase(2); await A.pulse(1, 'back', { size: (i) => Math.min(1, Math.abs(R.gW2[i]) * 8) + 0.15 }); }, show: () => { phase(2); glab(1); A.sub(2, 0, 'δ=' + UL.fmt(R.d2, 3), '#eb6834'); } },
      { title: 'Backward: pass the blame on', text: () => `Each hidden neuron receives δ × its weight to the output, times its own slope: ${R.d1.map((d, j) => `δ${j + 1} = ${UL.fmt(d, 4)}`).join(', ')}. The first-layer weights get input × δ. Now every one of the 13 knobs knows which way to turn.`,
        run: async () => { await A.pulse(0, 'back', { size: (i, j) => Math.min(1, Math.abs(R.gW1[j][i]) * 12) + 0.15 }); }, show: () => { glab(0); R.d1.forEach((d, j) => A.sub(1, j, 'δ=' + UL.fmt(d, 3), '#eb6834')); } },
      { title: 'Nudge every weight', text: () => `Gradient descent: w ← w − η · ∂C/∂w with η = ${eta}. Watch the lines: positive slopes shrink their weight, negative slopes grow it. Run this example again and the cost would be <b>${f3(R.Cafter)}</b> instead of ${f3(R.C)}. Then the next example comes in.`,
        run: async () => { phase(3); A.clearEdgeLabels(); await A.morphWeights([R.nW1, R.nW2], { ms: 900 }); }, show: () => { phase(3); A.setWeights([R.nW1, R.nW2], { max: 3 }); A.edgeLabels(0, (i, j) => ({ text: f2(R.nW1[j][i]), color: R.nW1[j][i] >= 0 ? '#1b8c5c' : '#d64045' })); A.edgeLabels(1, (i) => ({ text: f2(R.nW2[0][i]), color: R.nW2[0][i] >= 0 ? '#1b8c5c' : '#d64045' })); } },
    ];
    const commit = () => { W1 = R.nW1; b1 = R.nb1; W2 = R.nW2; b2 = R.nb2; if (ex === 3) { hist.push(avgCost()); round++; } ex = (ex + 1) % 4; R = compute(); };
    init();
    const P = new UL.Player({ controls: 'wl-ctl', narration: 'wl-narr', progress: 'wl-bar', clock: A.clock, steps, base, loop: true, onLoop: commit, onReset: init, pause: 900,
      intro: 'Press <b>▶ Play</b> to watch one full round of learning, or <b>Next step</b> to go at your own pace. The weights start random.' });
    UL.$('#wl-ff').addEventListener('click', () => { P.pause(); if (P.k === steps.length - 1) commit(); for (let k = 0; k < 1200; k++) commit(); P.goto(-1); });
  })();

  UL.quiz('qz', [
    { q: 'Why do we not write if-statements to read handwritten digits?', o: ['Computers cannot compare numbers', 'Nobody can write exact rules that cover every way of writing a 3; learning from examples works better', 'It would be too fast', 'Digits are not numbers'], a: 1, why: 'No two 3s have the same pixels. The rules are too many and too fuzzy to write by hand.' },
    { q: 'In an artificial neuron, what plays the role of the synapse strength?', o: ['The bias', 'The weight', 'The activation function', 'The layer'], a: 1, why: 'Each connection has a weight, like the strength of a synapse; learning changes the weights.' },
    { q: 'Put the learning loop in order.', o: ['Nudge, guess, measure, direction', 'Guess, measure the cost, find the gradient, nudge the weights', 'Measure, nudge, guess, direction', 'Guess, nudge, measure, direction'], a: 1, why: 'Forward pass → cost → gradient (backpropagation) → gradient descent step. Repeat.' },
  ]);
  UL.start();
})();
