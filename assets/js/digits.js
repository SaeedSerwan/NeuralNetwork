/* The digit network: draw, look inside, train live
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$;
  const pre = UL.pretrained();
  let mine = new UL.MLP([784, 16, 16, 10], { seed: 7 }), mineTrained = false;
  const onehot = (d) => { const y = new Float64Array(10); y[d] = 1; return y; };
  const model = (k) => (k === 'mine' ? mine : pre);
  let px = new Float32Array(784), drawModel = 'pre', hoverN = null;
  const listeners = [];

  /* ------------------------------------------------------------ 01 draw */
  const NP = new UL.Plot('#dr-net', { w: 560, h: 440, pad: { l: 0, r: 0, t: 0, b: 0 } }), small = UL.pixPlot('#dr-small', 112);
  const LX = [40, 250, 400, 530], foot = $('#dr-foot');
  const ys = (n, top = 40, bot = 420) => Array.from({ length: n }, (_, k) => top + ((k + 0.5) * (bot - top)) / n);
  let rev = 3; // how far the forward pass has travelled: 0 = only the picture, 3 = all layers
  function renderNet() {
    const net = model(drawModel), out = Float64Array.from(net.forward(px)), a1 = net.a[1], a2 = net.a[2], H1 = net.sizes[1], H2 = net.sizes[2];
    const c = NP.ctx, Y1 = ys(H1), Y2 = ys(H2), Y3 = ys(10, 60, 400);
    NP.clear();
    // input: the picture, as a column-free block
    UL.drawDigit(c, px, LX[0] - 30, 160, 112);
    c.strokeStyle = '#d6d6d2'; c.strokeRect(LX[0] - 30.5, 159.5, 113, 113);
    c.font = '500 11px "IBM Plex Sans", sans-serif'; c.fillStyle = '#111'; c.textAlign = 'center';
    c.fillText('784 inputs', LX[0] + 26, 150); c.fillText('hidden 1', LX[1], 22); c.fillText('hidden 2', LX[2], 22); c.fillText('output', LX[3], 22);
    // input → hidden 1: one fan per neuron, coloured by the sign of its weighted sum contribution
    const k1 = Math.max(0, Math.min(1, rev)), k2 = Math.max(0, Math.min(1, rev - 1)), k3 = Math.max(0, Math.min(1, rev - 2));
    if (rev < 3) { const fx = rev < 1 ? LX[0] + 83 + (LX[1] - LX[0] - 83) * k1 : rev < 2 ? LX[1] + (LX[2] - LX[1]) * k2 : LX[2] + (LX[3] - LX[2]) * k3; const gr = c.createLinearGradient(fx - 40, 0, fx + 6, 0); gr.addColorStop(0, 'rgba(42,120,214,0)'); gr.addColorStop(1, 'rgba(42,120,214,.28)'); c.fillStyle = gr; c.fillRect(fx - 40, 30, 46, 400); }
    for (let j = 0; j < H1 && k1 > 0; j++) { const z = net.z[1][j]; c.strokeStyle = z > 0 ? 'rgba(27,140,92,.25)' : 'rgba(214,64,69,.18)'; c.lineWidth = Math.min(3, 0.3 + Math.abs(z) * 0.15); c.beginPath(); c.moveTo(LX[0] + 83, 216); c.lineTo(LX[1] - 9, Y1[j]); c.stroke(); }
    const fan = (A, YA, XA, YB, XB, W, nIn, nOut) => {
      const items = []; for (let j = 0; j < nOut; j++) for (let i = 0; i < nIn; i++) items.push([W[j * nIn + i] * A[i], i, j]);
      items.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0]));
      items.slice(0, 70).forEach(([v, i, j], r) => { c.strokeStyle = v > 0 ? `rgba(27,140,92,${0.15 + 0.6 * (1 - r / 70)})` : `rgba(214,64,69,${0.12 + 0.5 * (1 - r / 70)})`; c.lineWidth = Math.min(3, 0.4 + Math.abs(v) * 0.9); c.beginPath(); c.moveTo(XA + 9, YA[i]); c.lineTo(XB - 9, YB[j]); c.stroke(); });
    };
    if (k1 >= 1) fan(a1, Y1, LX[1], Y2, LX[2], net.W[1], H1, H2); if (k2 >= 1) fan(a2, Y2, LX[2], Y3, LX[3], net.W[2], H2, 10);
    const node = (x, y, a, hi, r = 8.5) => { const g = Math.round(255 * (1 - a)); c.fillStyle = `rgb(${g},${g},${g})`; c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill(); c.lineWidth = hi ? 2.5 : 1; c.strokeStyle = hi ? '#eb6834' : '#111'; c.stroke(); };
    Y1.forEach((y, j) => node(LX[1], y, a1[j] * k1, hoverN && hoverN[0] === 1 && hoverN[1] === j, H1 > 16 ? 5 : 8.5));
    Y2.forEach((y, j) => node(LX[2], y, a2[j] * k2, hoverN && hoverN[0] === 2 && hoverN[1] === j, H2 > 16 ? 5 : 8.5));
    const top = UL.argmax(out);
    Y3.forEach((y, j) => { node(LX[3], y, out[j] * k3, j === top && k3 >= 1, 10); c.fillStyle = j === top ? '#111' : '#6b6b68'; c.font = (j === top ? '600 ' : '') + '12px "IBM Plex Mono", monospace'; c.textAlign = 'left'; c.fillText(String(j), LX[3] + 16, y + 4); });
    UL.drawDigit(small.ctx, px, 0, 0, 112);
    const empty = px.every((v) => v === 0);
    UL.outbars('dr-out', Array.from(out, (v) => v * k3));
    const sorted = Array.from(out).map((v, j) => [v, j]).sort((p, q) => q[0] - p[0]);
    $('#dr-ans').innerHTML = !empty && k3 < 1 ? '<span class="k">Answer</span><span>the signal is on its way…</span>' : empty ? '<span class="k">Answer</span><span>draw something</span>' : `<span class="k">Answer</span><span class="big-num">${top}</span><span class="k">Runner-up</span><span>${sorted[1][1]} (${sorted[1][0].toFixed(2)})</span>`;
    NP._Y = [Y1, Y2];
  }
  const pad = new UL.DrawPad('#dr-pad', { onchange: (p) => { px = p; renderNet(); listeners.forEach((f) => f()); } });
  const clockW = new UL.Clock();
  async function wave() { // replay the forward pass layer by layer, so students see the signal travel
    clockW.cancel(); try { await clockW.tween(1500, (t, raw) => { rev = raw * 3; renderNet(); }); } catch (e) { return; } rev = 3; renderNet();
  }
  pad.cv.addEventListener('mouseup', wave); pad.cv.addEventListener('touchend', wave);
  const setAndWave = (p) => { pad.setPixels(p); wave(); };
  $('#dr-clear').addEventListener('click', () => pad.clear());
  UL.seg('dr-model', (v) => { drawModel = v; if (v === 'mine' && !mineTrained) foot.innerHTML = '<b>Your network has not been trained yet</b>: its weights are random. Train it in § 03 first.'; renderNet(); });
  NP.cv.addEventListener('mousemove', (e) => {
    const [X, Y] = NP.pointer(e), net = model(drawModel); let hit = null;
    [1, 2].forEach((l) => { if (Math.abs(X - LX[l]) < 14) NP._Y[l - 1].forEach((y, j) => { if (Math.abs(Y - y) < 11) hit = [l, j]; }); });
    if (JSON.stringify(hit) === JSON.stringify(hoverN)) return; hoverN = hit; renderNet();
    if (!hit) { foot.textContent = 'Hover a neuron of hidden layer 1 to see its 784 weights as a picture.'; return; }
    const [l, j] = hit, nIn = net.sizes[l - 1];
    if (l === 1) {
      const w = net.W[0].subarray(j * 784, j * 784 + 784); let mx = 0; w.forEach((v) => { mx = Math.max(mx, Math.abs(v)); });
      foot.innerHTML = `<div style="display:flex;gap:1rem;align-items:center"><canvas id="dr-wimg" width="84" height="84" class="pix" style="width:84px;border-radius:4px;border:1px solid var(--line)"></canvas><span>Hidden layer 1, neuron ${j + 1}: weighted sum <b>${UL.fmt(net.z[1][j], 2)}</b> (bias ${UL.fmt(net.b[0][j], 2)}), activation <b>${net.a[1][j].toFixed(2)}</b>. Its 784 weights as a picture: green pixels push it up, red pixels push it down.</span></div>`;
      UL.drawDigit($('#dr-wimg').getContext('2d'), w, 0, 0, 84, { signed: true, scale: mx * 0.7 });
    } else {
      foot.innerHTML = `Hidden layer 2, neuron ${j + 1}: it reads the ${nIn} neurons of hidden layer 1. Weighted sum <b>${UL.fmt(net.z[2][j], 2)}</b>, activation <b>${net.a[2][j].toFixed(2)}</b>. Its strongest inputs: ${Array.from(net.W[1].subarray(j * nIn, j * nIn + nIn)).map((w, i) => [w, i]).sort((p, q) => Math.abs(q[0]) - Math.abs(p[0])).slice(0, 3).map(([w, i]) => `neuron ${i + 1} (${UL.fmt(w, 1)})`).join(', ')}.`;
    }
  });
  NP.cv.addEventListener('mouseleave', () => { hoverN = null; renderNet(); });

  /* ------------------------------------------------------------ 02 learned weights */
  function weightGrid(el, net, o = {}) {
    el = typeof el === 'string' ? document.getElementById(el) : el; const H = net.sizes[1];
    if (el.children.length !== H) { el.innerHTML = Array.from({ length: H }, (_, j) => `<figure><canvas width="56" height="56"></canvas><figcaption>${j + 1}</figcaption></figure>`).join(''); }
    const cvs = $$('canvas', el);
    for (let j = 0; j < H; j++) { const w = net.W[0].subarray(j * 784, j * 784 + 784); let mx = 1e-9; for (let i = 0; i < 784; i++) mx = Math.max(mx, Math.abs(w[i])); UL.drawDigit(cvs[j].getContext('2d'), w, 0, 0, 56, { signed: true, scale: mx * 0.75 }); }
    return cvs;
  }
  let lwModel = 'pre';
  function drawLearned() {
    const net = model(lwModel), cvs = weightGrid('lw-grid', net); net.forward(px);
    $$('#lw-grid figcaption').forEach((f, j) => { f.innerHTML = `${j + 1} · <b style="color:var(--ink)">${net.a[1][j].toFixed(2)}</b>`; });
    cvs.forEach((c) => { c.style.outline = ''; });
  }
  UL.seg('lw-model', (v) => { lwModel = v; drawLearned(); });
  listeners.push(drawLearned);

  /* ------------------------------------------------------------ 03 train */
  const status = $('#tr-status'), CP = new UL.Plot('#tr-cost', { w: 460, h: 240, pad: { l: 44, r: 10, t: 12, b: 34 } }), AP = new UL.Plot('#tr-acc', { w: 460, h: 240, pad: { l: 44, r: 10, t: 12, b: 34 } });
  const eta = UL.range('tr-eta', () => {}, (v) => v.toFixed(1));
  let batch = 32, D = null, hist = [], seen = 0, order = [], ptr = 0, seed = 7;
  const test = (net, n = 1000) => { let c = 0, ok = 0; for (let k = 0; k < n; k++) { const i = D.nTrain + k; net.forward(D.get(i)); c += net.cost(onehot(D.y[i])); if (UL.argmax(net.a[net.L]) === D.y[i]) ok++; } return [c / n, ok / n]; };
  const shuffle = () => { const r = UL.rng(seed * 31 + hist.length); order = Array.from({ length: D.nTrain }, (_, i) => i); for (let i = order.length - 1; i > 0; i--) { const j = r.int(i + 1); [order[i], order[j]] = [order[j], order[i]]; } ptr = 0; };
  function plots() {
    const xs = hist.map((h) => h[0]), xmax = Math.max(5000, ...xs);
    const draw = (P, k, ymax, lab, fy) => { P.clear(); P.domain(0, xmax, 0, ymax); P.axes({ xlabel: 'training digits seen', fx: (v) => (v >= 1000 ? v / 1000 + 'k' : v), fy }); if (hist.length) P.clipped(() => P.poly(hist.map((h) => [h[0], h[k]]), { color: k === 1 ? UL.COLORS[1] : UL.COLORS[2], w: 2.2 })); };
    draw(CP, 1, Math.max(1, ...hist.map((h) => Math.min(3.5, h[1]))) * 1.05); draw(AP, 2, 1, '', (v) => Math.round(v * 100) + '%');
    const last = hist[hist.length - 1] || [0, 0, 0];
    UL.stats('tr-stats', [{ k: 'Digits seen', v: seen.toLocaleString('en-US') }, { k: 'Epochs (passes over 5,000)', v: (seen / D.nTrain).toFixed(1) }, { k: 'Test cost', v: last[1].toFixed(3), hi: 1 }, { k: 'Test accuracy', v: (last[2] * 100).toFixed(1) + '%', hi: 1 }, { k: 'Parameters', v: mine.nParams.toLocaleString('en-US') }]);
    weightGrid('tr-w', mine);
  }
  function resetMine() {
    const H = +$('#tr-hid').value; seed++; mine = new UL.MLP([784, H, H, 10], { seed }); mineTrained = false; seen = 0; hist = []; shuffle();
    const [c, a] = test(mine); hist.push([0, c, a]); plots(); if (drawModel === 'mine') renderNet(); if (lwModel === 'mine') drawLearned(); if (miModel === 'mine') mistakes();
  }
  let lastEval = 0;
  const loop = UL.loop(() => {
    const t0 = performance.now();
    while (performance.now() - t0 < 14) {
      for (let k = 0; k < batch; k++) { if (ptr >= order.length) shuffle(); const i = order[ptr++]; mine.forward(D.get(i)); mine.backward(onehot(D.y[i])); }
      mine.step(eta.v, batch); seen += batch;
      if (seen - lastEval >= 1000) break;
    }
    if (seen - lastEval >= 1000) { lastEval = seen; const [c, a] = test(mine); hist.push([seen, c, a]); mineTrained = true; plots(); if (drawModel === 'mine') renderNet(); if (lwModel === 'mine') drawLearned(); }
    status.innerHTML = `training… <b>${seen.toLocaleString('en-US')}</b> digits`;
    if (seen >= 150000) { stop(); status.innerHTML = 'finished 30 epochs'; return false; }
  });
  const stop = () => { loop.stop(); $('#tr-go').textContent = 'Train'; if (miModel === 'mine') mistakes(); };
  $('#tr-go').addEventListener('click', () => { if (!D) return; if (loop.on) { stop(); status.textContent = 'paused'; } else { loop.start(); $('#tr-go').textContent = 'Pause'; } });
  $('#tr-reset').addEventListener('click', () => { if (!D) return; stop(); lastEval = 0; resetMine(); status.textContent = 'new random weights'; });
  $('#tr-hid').addEventListener('change', () => { if (!D) return; stop(); lastEval = 0; resetMine(); });
  UL.seg('tr-batch', (v) => { batch = +v; });

  /* ------------------------------------------------------------ 04 mistakes */
  let miModel = 'pre';
  function mistakes() {
    if (!D) return; const net = model(miModel), conf = Array.from({ length: 10 }, () => new Array(10).fill(0)), wrong = [];
    for (let k = 0; k < D.nTest; k++) { const i = D.nTrain + k, p = net.predict(D.get(i)); conf[D.y[i]][p]++; if (p !== D.y[i]) wrong.push([i, p]); }
    const el = $('#mi-grid'); el.innerHTML = '';
    wrong.slice(0, 48).forEach(([i, p]) => { const f = document.createElement('figure'); f.style.margin = 0; f.style.textAlign = 'center'; f.style.font = '.7rem var(--mono)'; const c = document.createElement('canvas'); c.width = 28; c.height = 28; c.style.width = '42px'; c.style.height = '42px'; UL.drawDigit(c.getContext('2d'), D.get(i), 0, 0, 28); c.title = `said ${p}, was ${D.y[i]}`; c.addEventListener('click', () => { setAndWave(D.get(i)); document.getElementById('draw').scrollIntoView(); }); f.appendChild(c); f.insertAdjacentHTML('beforeend', `<figcaption><b class="neg">${p}</b> · ${D.y[i]}</figcaption>`); el.appendChild(f); });
    $('#mi-status').innerHTML = `<b>${wrong.length}</b> wrong out of ${D.nTest} test digits · accuracy <b>${(100 - (wrong.length * 100) / D.nTest).toFixed(1)}%</b>`;
    $('#mi-conf').innerHTML = `<tr><th>true ↓ · said →</th>${[...Array(10).keys()].map((j) => `<th>${j}</th>`).join('')}</tr>` + conf.map((row, i) => `<tr><th>${i}</th>${row.map((v, j) => `<td class="${i === j ? 'good' : v >= 3 ? 'bad' : ''}">${v || ''}</td>`).join('')}</tr>`).join('');
  }
  UL.seg('mi-model', (v) => { miModel = v; mistakes(); });

  /* ------------------------------------------------------------ 05 out of distribution */
  (function () {
    const P = UL.pixPlot('#od-img', 224); let r = UL.rng(9);
    const make = {
      noise: () => Float32Array.from({ length: 784 }, () => r()),
      blank: () => { const a = new Float32Array(784); for (let k = 0; k < 6; k++) a[r.int(784)] = 0.4 + 0.6 * r(); return a; },
      bars: () => { const a = new Float32Array(784), ph = r.int(4), v = r() < 0.5; for (let i = 0; i < 784; i++) { const q = v ? i % 28 : Math.floor(i / 28); a[i] = (q + ph) % 6 < 2 ? 1 : 0; } return a; },
      block: () => { const a = new Float32Array(784), x0 = 4 + r.int(10), y0 = 4 + r.int(10), w = 6 + r.int(10); for (let y = y0; y < Math.min(28, y0 + w); y++) for (let x = x0; x < Math.min(28, x0 + w); x++) a[y * 28 + x] = 1; return a; },
      inv: () => { const S = UL.samples(), src = S.get(40 + r.int(40)); return Float32Array.from(src, (v) => 1 - v); },
    };
    function run(k) {
      const a = make[k](), out = Float64Array.from(pre.forward(a)), top = UL.argmax(out);
      UL.drawDigit(P.ctx, a, 0, 0, 224); UL.outbars('od-out', out);
      $('#od-ans').innerHTML = `<span class="k">Answer</span><span class="big-num">${top}</span><span class="k">Confidence</span><span class="hl">${out[top].toFixed(2)}</span><span class="k">Honest answer</span><span>“this is not a digit”, which it cannot give</span>`;
    }
    $$('[data-o]').forEach((b) => b.addEventListener('click', () => run(b.dataset.o))); run('noise');
  })();

  /* ------------------------------------------------------------ start */
  pad.setPixels(UL.samples().get(UL.samples().ofDigit(5, 1)));
  status.textContent = 'loading 6,000 digits…';
  UL.digits().then((d) => {
    D = d; status.textContent = 'ready: press Train'; resetMine(); mistakes();
    $('#dr-rand').addEventListener('click', () => setAndWave(D.get(D.nTrain + Math.floor(Math.random() * D.nTest))));
  }).catch(() => { status.textContent = 'Could not load the digit data.'; });
  drawLearned();

  UL.quiz('qz', [
    { q: 'You draw a 7 and output neuron 7 has activation 0.93 while all others are below 0.1. What is the network’s answer?', o: ['0.93', '7', 'It depends on the cost', 'Nothing yet: it needs training'], a: 1, why: 'The answer is the index of the brightest output neuron.' },
    { q: 'The first-layer weight pictures mostly look like noise. What does that tell you?', o: ['The network is broken', 'The network found a working solution that is not the tidy edges-and-loops one we hoped for', 'The learning rate was 0', 'The pictures are drawn wrongly'], a: 1, why: 'About 95% correct with messy features: it optimises the cost, not our idea of neatness.' },
    { q: 'With η = 20 the test cost jumps around and accuracy stays poor. Why?', o: ['Too few hidden neurons', 'The steps are too large and overshoot', 'The data is imbalanced', 'The batch is too small'], a: 1, why: 'A learning rate that is too large overshoots the valley, as with the ball in Part 2.' },
    { q: 'Why does the network still name a digit for random noise or a white block?', o: ['Noise looks like a 0', 'It was only ever trained on digits and has no way to say “not a digit”', 'The bias of neuron 0 is large', 'It is a bug'], a: 1, why: 'Ten outputs always come out; one is always the largest. Outside its training data its confidence means nothing.' },
    { q: 'Why do we measure accuracy on 1,000 test digits that are never used for training?', o: ['They are easier', 'To see how well the network handles new data, not how well it remembers', 'To make training faster', 'Because MNIST says so'], a: 1, why: 'A network can do well on its training data simply by memorising it.' },
  ]);
  UL.start();
})();
