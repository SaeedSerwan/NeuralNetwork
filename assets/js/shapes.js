/* Warm-up: a network that tells circles, squares and triangles apart, trained in the browser
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$;
  const NAMES = ['circle', 'square', 'triangle'], f2 = (v) => UL.fmt(v, 2);

  /* ------------------------------------------------------------ drawing shapes as 28 × 28 pictures */
  function raster(poly, w) { // anti-aliased outline of a closed polygon
    const px = new Float32Array(784);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const X = x + 0.5, Y = y + 0.5; let d = 1e9;
      for (let i = 0; i < poly.length; i++) {
        const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % poly.length], vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((X - ax) * vx + (Y - ay) * vy) / (vx * vx + vy * vy || 1)));
        d = Math.min(d, Math.hypot(X - ax - t * vx, Y - ay - t * vy));
      }
      px[y * 28 + x] = Math.max(0, Math.min(1, w / 2 + 0.6 - d));
    }
    return px;
  }
  function normalise(src) { // crop, scale into 20 × 20, centre of mass in the middle: as the drawing pad does
    let x0 = 28, x1 = -1, y0 = 28, y1 = -1;
    for (let i = 0; i < 784; i++) if (src[i] > 0.08) { const x = i % 28, y = Math.floor(i / 28); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    if (x1 < 0) return src;
    const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h), ox = (28 - w * s) / 2, oy = (28 - h * s) / 2, out = new Float32Array(784);
    const at = (x, y) => (x < 0 || y < 0 || x > 27 || y > 27 ? 0 : src[y * 28 + x]);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const sx = x0 + (x + 0.5 - ox) / s - 0.5, sy = y0 + (y + 0.5 - oy) / s - 0.5, fx = Math.floor(sx), fy = Math.floor(sy), ax = sx - fx, ay = sy - fy;
      if (sx < x0 - 1 || sx > x1 + 1 || sy < y0 - 1 || sy > y1 + 1) continue;
      out[y * 28 + x] = Math.min(1, (at(fx, fy) * (1 - ax) + at(fx + 1, fy) * ax) * (1 - ay) + (at(fx, fy + 1) * (1 - ax) + at(fx + 1, fy + 1) * ax) * ay);
    }
    let m = 0, mx = 0, my = 0; out.forEach((v, i) => { m += v; mx += v * (i % 28); my += v * Math.floor(i / 28); });
    const dx = Math.round(13.5 - mx / m), dy = Math.round(13.5 - my / m), sh = new Float32Array(784);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) { const yy = y - dy, xx = x - dx; if (yy >= 0 && yy < 28 && xx >= 0 && xx < 28) sh[y * 28 + x] = out[yy * 28 + xx]; }
    return sh;
  }
  function shape(k, r) {
    const cx = 14 + r.normal() * 1.5, cy = 14 + r.normal() * 1.5, R = 6 + r() * 4, rot = (r() - 0.5) * (k === 0 ? 6 : 0.9), j = () => r.normal() * 0.45, asp = 0.8 + r() * 0.4;
    let poly;
    if (k === 0) poly = Array.from({ length: 28 }, (_, i) => { const a = (i / 28) * 2 * Math.PI; return [cx + R * Math.cos(a) * asp + j() * 0.5, cy + R * Math.sin(a) + j() * 0.5]; });
    else { const n = k === 1 ? 4 : 3, off = k === 1 ? Math.PI / 4 : -Math.PI / 2; poly = Array.from({ length: n }, (_, i) => { const a = off + rot + (i / n) * 2 * Math.PI; return [cx + R * Math.cos(a) * (k === 1 ? asp : 1) + j(), cy + R * Math.sin(a) + j()]; }); }
    return normalise(raster(poly, 1.4 + r() * 1.8));
  }
  const r0 = UL.rng(12), TRAIN = [], TEST = [];
  for (let i = 0; i < 3000; i++) TRAIN.push([shape(i % 3, r0), i % 3]);
  for (let i = 0; i < 300; i++) TEST.push([shape(i % 3, r0), i % 3]);
  const EX = [0, 1, 2].map((k) => shape(k, UL.rng(100 + k)));
  const want = (k) => [0, 1, 2].map((j) => (j === k ? 1 : 0));
  const acc = (net) => TEST.filter(([x, y]) => net.predict(x) === y).length / TEST.length;
  const clone = (net) => { const c = new UL.MLP(net.sizes); c.W.forEach((W, l) => W.set(net.W[l])); c.b.forEach((b, l) => b.set(net.b[l])); return c; };

  /* ------------------------------------------------------------ the trained network */
  const net = new UL.MLP([784, 16, 3], { seed: 4 }); let ready = false, px = new Float32Array(784), truth = null;
  const small = UL.pixPlot('#sh-small', 84), st = $('#sh-status');
  function predict() {
    UL.drawDigit(small.ctx, px, 0, 0, 84);
    const out = Float64Array.from(net.forward(px)), sum = out.reduce((a, b) => a + b, 0) || 1, p = Array.from(out, (v) => v / sum), top = UL.argmax(out), empty = px.every((v) => v === 0);
    UL.outbars('sh-out', p, -1, { labels: ['○', '□', '△'] });
    $('#sh-ans').innerHTML = !ready ? '<span class="k">Network</span><span>still training…</span>' : empty ? '<span class="k">Answer</span><span>draw a shape</span>' : `<span class="k">Answer</span><span class="big-num">${NAMES[top]}</span><span class="k">Probability</span><span>${Math.round(p[top] * 100)}%</span>`;
    stepDraw();
  }
  const pad = new UL.DrawPad('#sh-pad', { onchange: (p) => { px = p; truth = null; predict(); } });
  $('#sh-clear').addEventListener('click', () => pad.clear());
  $$('[data-ex]').forEach((b) => b.addEventListener('click', () => { pad.setPixels(EX[+b.dataset.ex]); truth = +b.dataset.ex; stepDraw(); }));
  // train in small chunks so the page stays responsive
  let ep = 0, pos = 0; const order = TRAIN.map((_, i) => i);
  function chunk() {
    for (let k = 0; k < 300; k++) { if (pos % 10 === 0 && pos) net.step(3, 10); const [x, y] = TRAIN[order[pos]]; net.forward(x); net.backward(want(y)); pos++; if (pos >= TRAIN.length) { net.step(3, 10); pos = 0; ep++; order.sort(() => Math.random() - 0.5); break; } }
    st.textContent = `training: pass ${ep + 1} of 8`;
    if (ep < 8) setTimeout(chunk, 0); else { ready = true; st.innerHTML = `trained on 3,000 shapes · accuracy on 300 new shapes <b>${Math.round(acc(net) * 100)}%</b>`; predict(); refresh(); }
  }
  order.sort(() => Math.random() - 0.5); setTimeout(chunk, 50);

  /* ------------------------------------------------------------ step by step, animated */
  let forced = null, R = null, player = null;
  const A = new UL.NetAnim('st-svg', [{ n: 6, label: 'input · 784', names: ['', '', '', '', '', ''] }, { n: 6, label: 'hidden · 16' }, { n: 3, label: 'output · 3', names: ['○ circle', '□ square', '△ triangle'] }], { w: 560, h: 380, r: 17, left: 90, right: 120 });
  function computeR() {
    const out = Float64Array.from(net.forward(px)), a1 = Float64Array.from(net.a[1]), z1 = Float64Array.from(net.z[1]), sum = out.reduce((p, q) => p + q, 0) || 1, p = Array.from(out, (v) => v / sum), top = UL.argmax(out);
    const ins = Array.from(px).map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, 6).map((x) => x[1]);
    const hid = Array.from(a1).map((v, j) => [v * Math.abs(net.W[1][top * 16 + j]), j]).sort((x, y) => y[0] - x[0]).slice(0, 6).map((x) => x[1]).sort((x, y) => x - y);
    const hj = hid.reduce((b, j) => (a1[j] * net.W[1][top * 16 + j] > a1[b] * net.W[1][top * 16 + b] ? j : b), hid[0]), hk = hid.indexOf(hj);
    const T = forced != null ? forced : truth != null ? truth : top, err = out.reduce((s2, v, j) => s2 + (v - (j === T ? 1 : 0)) ** 2, 0);
    const cp = clone(net); cp.forward(px); cp.backward(want(T)); const g0 = hid.map((j) => ins.map((i) => cp.gW[0][j * 784 + i])), g1 = [0, 1, 2].map((o) => hid.map((j) => cp.gW[1][o * 16 + j]));
    cp.step(3, 1); const after = Float64Array.from(cp.forward(px)), err2 = after.reduce((s2, v, j) => s2 + (v - (j === T ? 1 : 0)) ** 2, 0);
    const W0 = hid.map((j) => ins.map((i) => net.W[0][j * 784 + i])), W1 = [0, 1, 2].map((o) => hid.map((j) => net.W[1][o * 16 + j]));
    const W0n = hid.map((j) => ins.map((i) => cp.W[0][j * 784 + i])), W1n = [0, 1, 2].map((o) => hid.map((j) => cp.W[1][o * 16 + j]));
    return { out, a1, z1, p, top, ins, hid, hj, hk, T, err, err2, g0, g1, W0, W1, W0n, W1n, xin: ins.map((i) => px[i]), ah: hid.map((j) => a1[j]) };
  }
  const empty = () => px.every((v) => v === 0);
  const base = () => {
    A.setWeights([R.W0, R.W1]); A.setLayer(0, R.ins.map(() => null)); A.setLayer(1, R.hid.map(() => null)); A.setLayer(2, [null, null, null]);
    A.clearEdgeLabels(); A.clearRings(); A.dim(false); A.nodes[1].forEach((_, k) => A.sub(1, k, '')); A.nodes[2].forEach((_, k) => A.sub(2, k, ''));
    R.ins.forEach((i, k) => { A.nodes[0][k].side.textContent = `pixel ${i + 1}`; });
  };
  const focus = (l, i, j) => l === 0 && j === R.hk;
  const STEPS = [
    { title: 'The layers', text: () => 'A network is layers of neurons. The <b>input layer</b> receives the data, the <b>hidden layer</b> computes, the <b>output layer</b> answers. Ours: 784 inputs, 16 hidden neurons, 3 outputs. Every line is a connection with its own weight.',
      run: async () => { for (const l of [0, 1, 2]) await Promise.all(A.nodes[l].map((_, k) => A.flash(l, k, '#2a78d6', 380))); } },
    { title: 'Input layer', text: () => `The drawing is 28 × 28 = 784 pixels. Each pixel feeds one <b>input neuron</b> with its brightness, from 0 (black) to 1 (white). Shown here: the six brightest pixels of your shape, ${R.xin.map((v) => v.toFixed(2)).join(', ')}.`,
      run: async () => { await A.fillLayer(0, R.xin, { ms: 700 }); }, show: () => A.setLayer(0, R.xin) },
    { title: 'Channels and weights', text: () => `The connections are called <b>channels</b>, and each has a <b>weight</b>: green positive, red negative, thicker = stronger. Look at one hidden neuron (orange ring): the weights of its channels from these six pixels are ${R.W0[R.hk].map((v) => v.toFixed(2)).join(', ')}.`,
      run: async () => { A.dim(true, focus); A.ring(1, R.hk, '#eb6834'); await A.clock.wait(500); }, show: () => { A.dim(true, focus); A.ring(1, R.hk, '#eb6834'); A.edgeLabels(0, (i, j) => (j === R.hk ? { text: f2(R.W0[j][i]), color: R.W0[j][i] >= 0 ? '#1b8c5c' : '#d64045' } : null)); } },
    { title: 'Weighted sum and bias', text: () => `Each input is multiplied by its channel’s weight (the dot size shows the product), the products of all 784 inputs are added up, and the neuron’s own <b>bias</b> (${f2(net.b[0][R.hj])}) is added: <b>z = ${f2(R.z1[R.hj])}</b>.`,
      run: async () => { await A.pulse(0, 'fwd', { only: (i, j) => j === R.hk, size: (i, j) => Math.abs(R.W0[j][i] * R.xin[i]) * 2 + 0.2, ms: 1100 }); }, show: () => A.sub(1, R.hk, 'z = ' + f2(R.z1[R.hj]), '#2a78d6') },
    { title: 'Activation function', text: () => `The sum goes through the <b>activation function</b> σ, which squishes it into 0…1 and decides how strongly the neuron fires: σ(${f2(R.z1[R.hj])}) = <b>${R.a1[R.hj].toFixed(2)}</b>. ${R.a1[R.hj] > 0.5 ? 'High: this neuron is <b>activated</b>.' : 'Low: this neuron stays quiet.'}`,
      run: async () => { await A.fillLayer(1, R.hid.map((j, k) => (k === R.hk ? R.ah[k] : 0)), { ms: 700 }); }, show: () => { A.setNode(1, R.hk, R.ah[R.hk]); } },
    { title: 'Forward propagation', text: () => 'Every hidden neuron does the same at the same time, and the activated ones pass their values on to the outputs through their own weighted channels. This flow from input to output is <b>forward propagation</b>.',
      run: async () => { A.dim(false); A.clearEdgeLabels(); A.ring(1, R.hk, null); await A.pulse(0, 'fwd', { size: (i, j) => Math.abs(R.W0[j][i] * R.xin[i]) * 2 + 0.1 }); await A.fillLayer(1, R.ah); await A.pulse(1, 'fwd', { size: (i, j) => Math.abs(R.W1[j][i] * R.ah[i]) * 0.3 + 0.1 }); await A.fillLayer(2, Array.from(R.out)); },
      show: () => { A.dim(false); A.clearEdgeLabels(); A.ring(1, R.hk, null); A.setLayer(1, R.ah); A.setLayer(2, Array.from(R.out)); } },
    { title: 'The output: a probability', text: () => `Scaled to add up to 100%, the outputs read as probabilities: ${NAMES.map((n, o) => `${n} <b>${Math.round(R.p[o] * 100)}%</b>`).join(', ')}. The highest wins: the network says <b>${NAMES[R.top]}</b>.`,
      run: async () => { await A.flash(2, R.top, '#2a78d6', 700); }, show: () => { R.p.forEach((q, o) => A.sub(2, o, Math.round(q * 100) + '%', o === R.top ? '#111' : '#6b6b68')); A.ring(2, R.top, '#2a78d6'); } },
    { title: 'Compare with the real answer', text: () => `In training we know the truth: it is a <b>${NAMES[R.T]}</b>${forced == null && truth == null ? ' (we assume the network’s own answer; pick another above to see a mistake)' : ''}. Wanted (${want(R.T).join(', ')}), got (${Array.from(R.out).map((v) => v.toFixed(2)).join(', ')}): the <b>error</b> is ${R.err.toFixed(3)}. ${R.top === R.T ? 'Right, though not 100% sure.' : '<b>Wrong</b>: this must be corrected.'}`,
      run: async () => { await A.flash(2, R.T, R.top === R.T ? '#1b8c5c' : '#d64045', 800); }, show: () => { A.ring(2, R.T, R.top === R.T ? '#1b8c5c' : '#d64045'); A.sub(2, R.T, 'should be 1', R.top === R.T ? '#1b8c5c' : '#d64045'); } },
    { title: 'Backpropagation', text: () => `The error is sent <b>backwards</b> through the network (orange). Each weight learns how much it contributed to the error and gets a small correction in the direction that lowers it. Watch the lines change: after one correction the error falls from ${R.err.toFixed(3)} to <b>${R.err2.toFixed(3)}</b>.`,
      run: async () => { A.clearRings(); await A.pulse(1, 'back', { size: (i, j) => Math.min(1, Math.abs(R.g1[j][i]) * 20) + 0.2 }); await A.pulse(0, 'back', { size: (i, j) => Math.min(1, Math.abs(R.g0[j][i]) * 200) + 0.2 }); await A.morphWeights([R.W0n, R.W1n], { ms: 900 }); },
      show: () => { A.clearRings(); A.setWeights([R.W0n, R.W1n], { max: A.wmax }); } },
    { title: 'Repeat', text: () => 'Forward propagation, compare, backpropagation: repeated for thousands of examples, many times over. The weights slowly settle on values that recognise shapes. That is <b>training</b>. Try it below, then see every part in depth in Parts 1 and 2.' },
  ];
  function refresh() { R = computeR(); if (player) player.goto(player.k); }
  function stepDraw() { if (!ready) return; refresh(); }
  R = computeR();
  player = new UL.Player({ controls: 'st-ctl', narration: 'st-narr', progress: 'st-bar', clock: A.clock, steps: STEPS, base, pause: 1100, intro: 'Press <b>▶ Play</b> to watch your shape travel through the network, or <b>Next step</b> to go at your own pace.' });
  UL.seg('st-truth', (v) => { forced = v === 'auto' ? null : +v; refresh(); });

  /* ------------------------------------------------------------ train a fresh network */
  (function () {
    const P = new UL.Plot('#tn-chart', { w: 460, h: 240, pad: { l: 42, r: 10, t: 12, b: 32 } }), img = UL.pixPlot('#tn-img', 84);
    let fresh, seen, hist, seed = 20; const probe = TEST[4];
    const reset = () => { fresh = new UL.MLP([784, 16, 3], { seed: seed++ }); seen = 0; hist = [[0, acc(fresh)]]; draw(); };
    function train(n) { for (let i = 0; i < n; i++) { const [x, y] = TRAIN[(seen + i) % TRAIN.length]; fresh.forward(x); fresh.backward(want(y)); if ((i + 1) % 10 === 0) fresh.step(3, 10); } seen += n; hist.push([seen, acc(fresh)]); }
    function draw() {
      P.clear(); P.domain(0, Math.max(1000, seen), 0, 1); P.axes({ xlabel: 'shapes seen', fy: (v) => Math.round(v * 100) + '%' }); P.line(0, 1 / 3, Math.max(1000, seen), 1 / 3, { color: UL.FAINT, dash: [4, 4], w: 1 });
      P.clipped(() => P.poly(hist, { color: UL.COLORS[2], w: 2.4 })); hist.forEach((h) => P.marker(h[0], h[1], { r: 3, fill: UL.COLORS[2] }));
      UL.drawDigit(img.ctx, probe[0], 0, 0, 84); const o = Float64Array.from(fresh.forward(probe[0])), s = o.reduce((a, b) => a + b, 0) || 1;
      UL.outbars('tn-out', Array.from(o, (v) => v / s), probe[1], { labels: ['○', '□', '△'] });
      $('#tn-calc').innerHTML = `<span class="k">This example</span><span>a ${NAMES[probe[1]]}; the network says <b>${NAMES[UL.argmax(o)]}</b></span><span class="k">Error</span><span>${fresh.cost(want(probe[1])).toFixed(3)}</span>`;
      $('#tn-status').innerHTML = `shapes seen <b>${seen}</b> · accuracy <b>${Math.round(hist[hist.length - 1][1] * 100)}%</b>${seen === 0 ? ' · random weights: about 33%, a guess' : ''}`;
    }
    $('#tn-go').addEventListener('click', () => { train(100); draw(); });
    $('#tn-auto').addEventListener('click', () => { let n = 0; const go = () => { train(200); draw(); n++; if (n < 30 && hist[hist.length - 1][1] < 0.95) setTimeout(go, 60); }; go(); });
    $('#tn-reset').addEventListener('click', reset);
    reset();
  })();

  UL.quiz('qz', [
    { q: 'A 28 × 28 picture is fed to a network. How many neurons does the input layer need?', o: ['28', '56', '784', '3'], a: 2, why: 'One input neuron per pixel: 28 × 28 = 784.' },
    { q: 'What is added to the weighted sum before the activation function?', o: ['The error', 'The bias', 'The output', 'Another layer'], a: 1, why: 'Each hidden neuron adds its own bias to the sum of inputs × weights.' },
    { q: 'What decides whether a neuron is activated?', o: ['The input layer', 'The activation function', 'The test set', 'The number of layers'], a: 1, why: 'The activation function (sigmoid, ReLU, …) turns the sum into the neuron’s output.' },
    { q: 'Which statement does NOT hold true?', o: ['Activation functions decide how strongly a neuron fires', 'Both forward and backward propagation take place during training', 'The error is only used by the input layer', 'Most of the computing happens in the hidden layers'], a: 2, why: 'The error is measured at the output and sent back through every layer (backpropagation), adjusting weights everywhere.' },
    { q: 'The network says “square” but the drawing is a triangle. What happens during training?', o: ['Nothing', 'The error is propagated back and the weights are adjusted', 'A new layer is added', 'The picture is deleted'], a: 1, why: 'That is backpropagation: compare, send the error back, correct the weights a little.' },
  ]);
  UL.start();
})();
