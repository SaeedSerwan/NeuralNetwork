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
    if (ep < 8) setTimeout(chunk, 0); else { ready = true; st.innerHTML = `trained on 3,000 shapes · accuracy on 300 new shapes <b>${Math.round(acc(net) * 100)}%</b>`; predict(); }
  }
  order.sort(() => Math.random() - 0.5); setTimeout(chunk, 50);

  /* ------------------------------------------------------------ step by step */
  let k = 0, forced = null; const svgEl = $('#st-svg'), calc = $('#st-calc');
  const STEPS = ['The layers', 'Input layer', 'Channels and weights', 'Weighted sum and bias', 'Activation function', 'Forward propagation', 'The output: a probability', 'Compare with the real answer', 'Backpropagation', 'Repeat'];
  function stepDraw() {
    if (!svgEl) return;
    const out = Float64Array.from(net.forward(px)), a1 = Float64Array.from(net.a[1]), z1 = Float64Array.from(net.z[1]), sum = out.reduce((a, b) => a + b, 0) || 1, p = Array.from(out, (v) => v / sum), top = UL.argmax(out);
    const ins = Array.from(px).map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]); const shownIn = ins.slice(0, 5).map((x) => x[1]).concat([0, 27, 783]);
    const hShown = Array.from(a1).map((v, j) => [v * Math.abs(net.W[1][top * 16 + j]), j]).sort((x, y) => y[0] - x[0]).slice(0, 6).map((x) => x[1]).sort((x, y) => x - y);
    const hj = hShown.reduce((b, j) => (a1[j] * net.W[1][top * 16 + j] > a1[b] * net.W[1][top * 16 + b] ? j : b), hShown[0]);
    const T = forced != null ? forced : truth != null ? truth : top, err = out.reduce((s, v, j) => s + (v - (j === T ? 1 : 0)) ** 2, 0);
    // one training step on a copy, to show backpropagation
    const cp = clone(net); cp.forward(px); cp.backward(want(T)); const gIn = shownIn.map((i) => cp.gW[0][hj * 784 + i]), gOut = [0, 1, 2].map((o) => cp.gW[1][o * 16 + hj]); cp.step(3, 1); const after = Float64Array.from(cp.forward(px)), err2 = after.reduce((s, v, j) => s + (v - (j === T ? 1 : 0)) ** 2, 0);
    const X = [50, 300, 520], yIn = shownIn.map((_, i) => 40 + i * 38), yH = hShown.map((_, i) => 55 + i * 46), yO = [110, 175, 240];
    const on = (s) => k >= s, hiIn = k >= 1, hiW = k === 2 || k === 3 || k === 8, hiH = k >= 3, hiO = k >= 5, back = k === 8;
    let s = '';
    shownIn.forEach((i, a) => hShown.forEach((j, b) => { const w = net.W[0][j * 784 + i], focus = j === hj && (k === 2 || k === 3 || k === 8); s += `<line x1="${X[0] + 14}" y1="${yIn[a]}" x2="${X[1] - 14}" y2="${yH[b]}" stroke="${!hiW && !focus ? '#e2e2de' : w >= 0 ? '#1b8c5c' : '#d64045'}" stroke-opacity="${focus ? 0.9 : hiW ? 0.25 : 1}" stroke-width="${focus ? 1 + Math.min(4, Math.abs(w) * 6) : 0.8}"/>`; }));
    hShown.forEach((j, b) => yO.forEach((y, o) => { const w = net.W[1][o * 16 + j], act = k >= 5 && a1[j] > 0.5; s += `<line x1="${X[1] + 14}" y1="${yH[b]}" x2="${X[2] - 16}" y2="${y}" stroke="${!act ? '#e2e2de' : w >= 0 ? '#1b8c5c' : '#d64045'}" stroke-width="${act ? 1 + Math.min(4, Math.abs(w) * 0.5) : 0.8}" stroke-opacity="${act ? 0.75 : 1}"/>`; }));
    if (back) { s += `<path d="M ${X[2] - 20} 290 C 420 320, 360 320, ${X[1] + 10} 300" stroke="#eb6834" stroke-width="2.5" fill="none" marker-end="url(#ah)"/><path d="M ${X[1] - 10} 300 C 220 320, 140 320, ${X[0] + 20} 330" stroke="#eb6834" stroke-width="2.5" fill="none" marker-end="url(#ah)"/><text x="300" y="342" text-anchor="middle" style="font:500 11px var(--mono);fill:#eb6834">error flows back</text>`; }
    const node = (x, y, v, lab, hl, r = 13) => { const g = Math.round(255 * (1 - Math.max(0, Math.min(1, v)))); s += `<circle cx="${x}" cy="${y}" r="${r}" fill="${v == null ? '#fff' : `rgb(${g},${g},${g})`}" stroke="${hl ? '#eb6834' : '#111'}" stroke-width="${hl ? 2.5 : 1}"/>`; if (lab) s += `<text x="${x + r + 6}" y="${y + 4}" style="font:11px var(--mono);fill:#3d3d3b">${lab}</text>`; };
    shownIn.forEach((i, a) => { s += `<circle cx="${X[0]}" cy="${yIn[a]}" r="12" fill="rgb(${Math.round(255 * (1 - px[i]))},${Math.round(255 * (1 - px[i]))},${Math.round(255 * (1 - px[i]))})" stroke="#111"/>`; if (hiIn) s += `<text x="${X[0] - 18}" y="${yIn[a] + 4}" text-anchor="end" style="font:10px var(--mono);fill:#6b6b68">${px[i].toFixed(2)}</text>`; });
    hShown.forEach((j, b) => node(X[1], yH[b], hiH ? (k >= 4 ? a1[j] : null) : null, k === 3 && j === hj ? `z = ${f2(z1[j])}` : k >= 4 ? a1[j].toFixed(2) : '', j === hj && k >= 2 && k <= 4));
    yO.forEach((y, o) => node(X[2], y, hiO ? out[o] : null, ['○ circle', '□ square', '△ triangle'][o] + (k >= 6 ? `  ${Math.round(p[o] * 100)}%` : ''), k >= 6 && o === top, 15));
    s += `<text x="${X[0]}" y="18" text-anchor="middle" style="font:500 11.5px var(--font);fill:#111">input · 784</text><text x="${X[1]}" y="18" text-anchor="middle" style="font:500 11.5px var(--font);fill:#111">hidden · 16</text><text x="${X[2]}" y="18" text-anchor="middle" style="font:500 11.5px var(--font);fill:#111">output · 3</text><text x="${X[0]}" y="${yIn[7] + 30}" text-anchor="middle" style="font:10px var(--mono);fill:#6b6b68">+ 776 more</text>`;
    svgEl.innerHTML = `<svg class="netsvg" viewBox="0 0 660 ${back ? 350 : 340}"><defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#eb6834"/></marker></defs>${s}</svg>`;
    const w8 = shownIn.map((i) => net.W[0][hj * 784 + i]);
    const TXT = [
      'A network is made of layers of neurons. The <b>input layer</b> receives the data, the <b>hidden layers</b> do the computing, the <b>output layer</b> gives the answer. Our network: 784 inputs, 16 hidden neurons, 3 outputs.',
      `The drawing is 28 × 28 = 784 pixels. Each pixel feeds one <b>input neuron</b> with its brightness: 0 for black, 1 for white. The first five drawn here are the brightest pixels of your shape: ${shownIn.slice(0, 3).map((i) => px[i].toFixed(2)).join(', ')} …`,
      `Each input neuron is connected to every hidden neuron. These connections are called <b>channels</b>, and every channel has a number, its <b>weight</b> (green +, red −). The highlighted hidden neuron’s weights from the 8 drawn inputs: ${w8.map(f2).join(', ')}.`,
      `Each input is multiplied by its channel’s weight, and the products are added up. Then the neuron’s own <b>bias</b> is added. For the highlighted neuron: weighted sum of all 784 inputs + bias (${f2(net.b[0][hj])}) = <b>z = ${f2(z1[hj])}</b>.`,
      `The sum goes through an <b>activation function</b> (here the sigmoid σ). It decides whether the neuron is <b>activated</b>: σ(${f2(z1[hj])}) = <b>${a1[hj].toFixed(2)}</b>. ${a1[hj] > 0.5 ? 'High: this neuron fires and passes its signal on.' : 'Low: this neuron stays quiet.'} Dark circles are active neurons.`,
      `Activated neurons pass their values on to the next layer through their own channels and weights. This flow from input to output is <b>forward propagation</b>. Bright lines: signals from active hidden neurons.`,
      `Each output neuron ends with a value; scaled to add up to 100% they read as probabilities: ${NAMES.map((n, o) => `${n} ${Math.round(p[o] * 100)}%`).join(', ')}. The neuron with the highest probability is the answer: <b>${NAMES[top]}</b>.`,
      `During training we know the real answer: here a <b>${NAMES[T]}</b>${forced == null && truth == null ? ' (we assume the network’s own answer; choose another with the buttons above)' : ''}. Compare: wanted (${want(T).join(', ')}), got (${Array.from(out).map((v) => v.toFixed(2)).join(', ')}). The <b>error</b> = ${err.toFixed(3)}. ${top === T ? 'Correct, but not yet 100% sure.' : 'Wrong: this error must be corrected.'}`,
      `The error is sent <b>backwards</b> through the network: <b>backpropagation</b>. Every weight gets a small correction in the direction that lowers the error. Corrections for the highlighted neuron’s channels to the outputs: ${gOut.map((g) => UL.fmt(-3 * g, 4)).join(', ')}. After one correction the error falls from ${err.toFixed(3)} to <b>${err2.toFixed(3)}</b>.`,
      'Forward propagation, compare, backpropagation: repeat for thousands of examples, many times over, and the weights settle on values that recognise shapes. That is training. Try it in § 03, then see every part in depth in Parts 1 and 2.',
    ];
    calc.innerHTML = `<div><b>${k + 1} · ${STEPS[k]}</b></div><div style="font-family:var(--font);font-size:.92rem;line-height:1.65">${TXT[k]}</div>`;
    $('#st-status').innerHTML = `step <b>${k + 1}</b> of ${STEPS.length}`;
  }
  UL.seg('st-truth', (v) => { forced = v === 'auto' ? null : +v; stepDraw(); });
  $('#st-next').addEventListener('click', () => { k = (k + 1) % STEPS.length; stepDraw(); });
  $('#st-back').addEventListener('click', () => { k = (k + STEPS.length - 1) % STEPS.length; stepDraw(); });

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
