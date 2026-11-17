/* Part 3 · Types of neural networks: CNN filters, RNN memory, a 1-D GAN, attention
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$, S = UL.samples();
  const f2 = (v) => UL.fmt(v, 2);
  const picks = [S.ofDigit(4), S.ofDigit(7), S.ofDigit(0), S.ofDigit(2), S.ofDigit(8), 6, S.ofDigit(1)];

  /* ------------------------------------------------------------ shuffled pixels */
  (function () {
    const A = UL.pixPlot('#sh-a', 196), B = UL.pixPlot('#sh-b', 196), r = UL.rng(42), perm = Array.from({ length: 784 }, (_, i) => i);
    for (let i = 783; i > 0; i--) { const j = r.int(i + 1); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const show = (i) => { const a = S.get(i); UL.drawDigit(A.ctx, a, 0, 0, 196); UL.drawDigit(B.ctx, Float32Array.from(perm, (p) => a[p]), 0, 0, 196); };
    UL.thumbs('sh-thumbs', picks, show, { select: 0 }); show(picks[0]);
  })();

  /* ------------------------------------------------------------ convolution */
  const KER = { vert: [-1, 0, 1, -2, 0, 2, -1, 0, 1], horz: [-1, -2, -1, 0, 0, 0, 1, 2, 1], diag: [0, 1, 2, -1, 0, 1, -2, -1, 0], blur: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9], sharp: [0, -1, 0, -1, 5, -1, 0, -1, 0] };
  const conv = (x, k, relu) => { const y = new Float32Array(26 * 26); for (let i = 0; i < 26; i++) for (let j = 0; j < 26; j++) { let s = 0; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) s += k[u * 3 + v] * x[(i + u) * 28 + j + v]; y[i * 26 + j] = relu ? Math.max(0, s) : s; } return y; };
  (function () {
    const PI = UL.pixPlot('#cv-in', 280), PO = UL.pixPlot('#cv-out', 260), calc = $('#cv-calc'), relu = $('#cv-relu'), grid = $('#cv-grid');
    let k = KER.vert.slice(), idx = picks[0], pos = -1, y = null;
    grid.innerHTML = k.map((v, i) => `<input data-i="${i}" value="${+v.toFixed(2)}" aria-label="filter weight ${i + 1}">`).join('');
    const syncGrid = () => $$('input', grid).forEach((el, i) => { el.value = +k[i].toFixed(2); });
    $$('input', grid).forEach((el) => el.addEventListener('input', () => { const v = parseFloat(el.value); if (!isNaN(v)) { k[+el.dataset.i] = v; draw(); } }));
    function draw() {
      const x = S.get(idx); y = conv(x, k, relu.checked); let mx = 1e-9; y.forEach((v) => { mx = Math.max(mx, Math.abs(v)); });
      UL.drawDigit(PI.ctx, x, 0, 0, 280);
      const show = pos < 0 ? y : y.map((v, i) => (i <= pos ? v : 0));
      PO.clear(); UL.drawDigit(PO.ctx, show, 0, 0, 260, { signed: true, scale: mx * 0.8 });
      if (pos >= 0) {
        const i = Math.floor(pos / 26), j = pos % 26, s = 10;
        PI.ctx.strokeStyle = '#eb6834'; PI.ctx.lineWidth = 2.5; PI.ctx.strokeRect(j * s, i * s, 3 * s, 3 * s);
        PO.ctx.strokeStyle = '#111'; PO.ctx.lineWidth = 2; PO.ctx.strokeRect(j * 10, i * 10, 10, 10);
        const P = [], Pr = []; let sum = 0; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) { const xv = x[(i + u) * 28 + j + v], kv = k[u * 3 + v]; P.push(xv); Pr.push(xv * kv); sum += xv * kv; }
        const grid = (vals, kind) => `<div class="mgrid">${vals.map((v) => { const bg = kind === 'px' ? `rgb(${Math.round(v * 255)},${Math.round(v * 255)},${Math.round(v * 255)})` : UL.div(v, kind === 'k' ? Math.max(...k.map(Math.abs)) : Math.max(0.5, ...Pr.map(Math.abs))); const fg = kind === 'px' ? (v > 0.5 ? '#000' : '#fff') : '#111'; return `<span style="background:${bg};color:${fg}">${kind === 'px' ? v.toFixed(1) : UL.fmt(v, kind === 'k' ? 1 : 2)}</span>`; }).join('')}</div>`;
        const out = relu.checked ? Math.max(0, sum) : sum;
        calc.innerHTML = `<div><b>Position (${i + 1}, ${j + 1})</b></div><div class="convf">${grid(P, 'px')}<span class="op">⊙</span>${grid(k, 'k')}<span class="op">=</span>${grid(Pr, 'p')}</div><div style="font-size:.86rem">pixels under the filter · filter weights · products</div><div>add the 9 products: <b>${f2(sum)}</b>${relu.checked ? ` → ReLU → <b>${f2(out)}</b>` : ''}</div><div style="font-family:var(--font);color:var(--muted);font-size:.84rem">That one number becomes the outlined pixel of the feature map.</div>`;
      } else calc.innerHTML = `<div><b>The filter</b></div><div style="font-family:var(--font)">9 weights, the same at all 26 × 26 = 676 positions. Each output number is the sum of 9 products: filter weight × pixel under it.</div><div style="font-family:var(--font);color:var(--muted);font-size:.84rem">Hover the feature map, or press Play.</div>`;
    }
    UL.seg('cv-k', (v) => { k = KER[v].slice(); syncGrid(); draw(); });
    relu.addEventListener('change', draw);
    UL.thumbs('cv-thumbs', picks, (i) => { idx = i; draw(); }, { select: 0 });
    PO.cv.addEventListener('mousemove', (e) => { if (loop.on) return; const [X, Y] = PO.pointer(e); pos = Math.min(675, Math.floor(Y / 10) * 26 + Math.floor(X / 10)); draw(); });
    PO.cv.addEventListener('mouseleave', () => { if (!loop.on) { pos = -1; draw(); } });
    let t = 0, slowEnd = 0; const loop = UL.loop((dt) => { t += dt; if (t > (pos < slowEnd ? 0.4 : 0.02)) { t = 0; pos += pos < slowEnd ? 1 : 3; if (pos > 675) { pos = -1; draw(); $('#cv-play').textContent = 'Play'; return false; } draw(); } });
    $('#cv-play').addEventListener('click', () => { if (loop.on) { loop.stop(); $('#cv-play').textContent = 'Play'; } else { const x0 = S.get(idx); let r0 = 0; while (r0 < 25 && ![...Array(28).keys()].some((c) => x0[(r0 + 2) * 28 + c] > 0.3)) r0++; pos = r0 * 26 + 6; slowEnd = pos + 14; loop.start(); $('#cv-play').textContent = 'Pause'; } });
    draw();
  })();

  /* ------------------------------------------------------------ conv + relu + pool pipeline */
  (function () {
    const el = $('#pl-rows'), names = [['vertical edge', 'vert'], ['horizontal edge', 'horz'], ['diagonal', 'diag'], ['blur', 'blur']];
    el.innerHTML = names.map(([n]) => `<div style="display:grid;grid-template-columns:120px repeat(3, 110px);gap:1rem;align-items:center;font-size:.84rem;color:var(--muted)"><span>${n}</span><canvas width="28" height="28" class="pix" style="width:110px;border-radius:4px"></canvas><canvas width="26" height="26" class="pix" style="width:110px;border-radius:4px;border:1px solid var(--line)"></canvas><canvas width="13" height="13" class="pix" style="width:110px;border-radius:4px;border:1px solid var(--line)"></canvas></div>`).join('') + `<div style="display:grid;grid-template-columns:120px repeat(3, 110px);gap:1rem;font:.72rem var(--mono);color:var(--faint);text-align:center"><span></span><span>input 28 × 28</span><span>conv + ReLU 26 × 26</span><span>max-pool 13 × 13</span></div>`;
    function draw(i) {
      const x = S.get(i);
      $$('#pl-rows > div', document).slice(0, 4).forEach((row, r) => {
        const cv = $$('canvas', row), y = conv(x, KER[names[r][1]], true); let mx = 1e-9; y.forEach((v) => { mx = Math.max(mx, v); });
        const p = new Float32Array(169); for (let a = 0; a < 13; a++) for (let b = 0; b < 13; b++) p[a * 13 + b] = Math.max(y[2 * a * 26 + 2 * b], y[2 * a * 26 + 2 * b + 1], y[(2 * a + 1) * 26 + 2 * b], y[(2 * a + 1) * 26 + 2 * b + 1]);
        UL.drawDigit(cv[0].getContext('2d'), x, 0, 0, 28); UL.drawDigit(cv[1].getContext('2d'), y, 0, 0, 26, { signed: true, scale: mx * 0.8 }); UL.drawDigit(cv[2].getContext('2d'), p, 0, 0, 13, { signed: true, scale: mx * 0.8 });
      });
    }
    UL.thumbs('pl-thumbs', picks, draw, { select: 0 }); draw(picks[0]);
  })();

  /* ------------------------------------------------------------ RNN memory */
  (function () {
    const words = ['This', 'morning', 'I', 'took', 'my', 'cat', 'for', 'a', '___'];
    const H = 8, E = 6, r = UL.rng(17);
    const emb = words.map(() => Array.from({ length: E }, () => r.normal()));
    const Wh = Array.from({ length: H }, () => Array.from({ length: H }, () => r.normal() * 0.38)), Wx = Array.from({ length: H }, () => Array.from({ length: E }, () => r.normal() * 0.45));
    const Wf = Array.from({ length: H }, () => Array.from({ length: E }, () => r.normal() * 0.2));
    const mv = (M, v) => M.map((row) => row.reduce((s, w, i) => s + w * v[i], 0));
    function run(type, first) { // returns the hidden states, optionally with a different first word
      let h = new Array(H).fill(0), c = new Array(H).fill(0); const out = [];
      words.forEach((_, t) => {
        const x = t === 0 && first ? first : emb[t], cand = mv(Wh, h).map((v, i) => Math.tanh(v + mv(Wx, x)[i]));
        if (type === 'rnn') h = cand;
        else { const f = mv(Wf, x).map((v) => 1 / (1 + Math.exp(-(v + 3.2)))), ig = 0.5; c = c.map((ci, i) => f[i] * ci + ig * cand[i]); h = c.map((v) => Math.tanh(v)); }
        out.push(h.slice());
      });
      return out;
    }
    const alt = Array.from({ length: E }, () => r.normal());
    let type = 'rnn', t = 0;
    const HP = new UL.Plot('#rn-heat', { w: 420, h: 220, pad: { l: 8, r: 8, t: 8, b: 26 } }), MP = new UL.Plot('#rn-mem', { w: 420, h: 220, pad: { l: 40, r: 10, t: 10, b: 30 } });
    function draw() {
      const hs = run(type), hs2 = run(type, alt), diff = hs.map((h, k) => Math.sqrt(h.reduce((s, v, i) => s + (v - hs2[k][i]) ** 2, 0)));
      $('#rn-tokens').innerHTML = words.map((w, k) => `<span class="${k === t - 1 ? 'on' : k < t ? 'done' : ''}">${w}</span>`).join('');
      HP.clear(); const cw = (HP.W - 16) / words.length, ch = (HP.H - 34) / H;
      for (let k = 0; k < t; k++) for (let i = 0; i < H; i++) { HP.ctx.fillStyle = UL.div(hs[k][i], 1); HP.ctx.fillRect(8 + k * cw + 1, 8 + i * ch + 1, cw - 2, ch - 2); }
      words.forEach((w, k) => HP.textPx(w.length > 6 ? w.slice(0, 5) + '…' : w, 8 + (k + 0.5) * cw, HP.H - 12, { font: '10px "IBM Plex Mono", monospace', color: k < t ? '#111' : '#a3a3a0' }));
      const d0 = diff[0] || 1; UL.chart(MP, [{ data: diff.slice(0, Math.max(1, t)).map((d) => d / d0), color: type === 'rnn' ? UL.COLORS[1] : UL.COLORS[2], w: 2.4 }], { xmax: 9, ymin: 0, ymax: 1.05, xlabel: 'words read', fy: (v) => Math.round(v * 100) + '%' });
      const last = t ? diff[t - 1] / d0 : 1;
      $('#rn-foot').innerHTML = t === 0 ? 'Press <b>Read next word</b>. The right chart changes the first word (“This”) to a different word and measures how much the memory changes: that is how much of the first word is still remembered.' : `After “${words[t - 1]}”: <b>${Math.round(last * 100)}%</b> of the first word’s effect is left in the ${type === 'rnn' ? 'simple RNN’s' : 'LSTM’s'} memory. ${type === 'rnn' ? 'Each step multiplies the old memory by W_h and squishes it, so old information fades.' : 'The forget gate stays close to 1, so the cell state carries the old information forward almost unchanged.'}`;
    }
    $('#rn-next').addEventListener('click', () => { if (t < words.length) t++; draw(); });
    $('#rn-all').addEventListener('click', () => { t = words.length; draw(); });
    $('#rn-reset').addEventListener('click', () => { t = 0; draw(); });
    UL.seg('rn-type', (v) => { type = v; draw(); });
    draw();
  })();

  /* ------------------------------------------------------------ 1-D GAN */
  (function () {
    const P = new UL.Plot('#gn-plot', { w: 560, h: 320, pad: { l: 36, r: 12, t: 12, b: 32 } }), calc = $('#gn-calc'), st = $('#gn-status');
    let data = 'one', mu, ls, a, b, c, rounds, r;
    const sig = UL.sigmoid, D = (x) => sig(a * x * x + b * x + c);
    const real = () => (data === 'one' ? 1.6 + 0.5 * r.normal() : (r() < 0.5 ? -1.8 : 1.8) + 0.45 * r.normal());
    const reset = () => { r = UL.rng(5); mu = -2.5; ls = Math.log(0.4); a = 0; b = 0; c = 0; rounds = 0; draw(); };
    function round() {
      const n = 64, lrD = 0.05, lrG = 0.03;
      for (let k = 0; k < 3; k++) { // detective: raise D on real, lower it on fakes
        let ga = 0, gb = 0, gc = 0;
        for (let i = 0; i < n; i++) { const x = real(), d = D(x); ga += (1 - d) * x * x; gb += (1 - d) * x; gc += 1 - d; const z = r.normal(), f = mu + Math.exp(ls) * z, df = D(f); ga -= df * f * f; gb -= df * f; gc -= df; }
        a += (lrD * ga) / n; b += (lrD * gb) / n; c += (lrD * gc) / n; a = Math.max(-3, Math.min(3, a));
      }
      let gm = 0, gs = 0; // forger: make D(fake) large (maximise log D)
      for (let i = 0; i < n; i++) { const z = r.normal(), s = Math.exp(ls), f = mu + s * z, d = D(f), g = (1 - d) * (2 * a * f + b); gm += g; gs += g * z * s; }
      mu += (lrG * gm) / n * 4; ls += (lrG * gs) / n * 2; ls = Math.max(Math.log(0.05), Math.min(Math.log(3), ls)); rounds++;
    }
    function hist(sample, nb = 40, lo = -5, hi = 5) { const h = new Array(nb).fill(0); sample.forEach((x) => { const k = Math.floor(((x - lo) / (hi - lo)) * nb); if (k >= 0 && k < nb) h[k]++; }); return h; }
    function draw() {
      const rr = UL.rng(99), R = [], F = [], N = 3000;
      for (let i = 0; i < N; i++) { R.push(data === 'one' ? 1.6 + 0.5 * rr.normal() : (rr() < 0.5 ? -1.8 : 1.8) + 0.45 * rr.normal()); F.push(mu + Math.exp(ls) * rr.normal()); }
      const hr = hist(R), hf = hist(F), mx = Math.max(...hr, ...hf), bw = 10 / 40;
      P.clear(); P.domain(-5, 5, 0, 1.05); P.axes({ xlabel: 'x', ny: 4 });
      const c2 = P.ctx;
      hr.forEach((v, k) => { c2.fillStyle = 'rgba(120,120,115,.35)'; const x0 = P.sx(-5 + k * bw), x1 = P.sx(-5 + (k + 1) * bw); c2.fillRect(x0, P.sy((v / mx) * 0.95), x1 - x0 - 1, P.sy(0) - P.sy((v / mx) * 0.95)); });
      hf.forEach((v, k) => { c2.fillStyle = 'rgba(42,120,214,.55)'; const x0 = P.sx(-5 + k * bw), x1 = P.sx(-5 + (k + 1) * bw); c2.fillRect(x0 + 2, P.sy((v / mx) * 0.95), x1 - x0 - 5, P.sy(0) - P.sy((v / mx) * 0.95)); });
      const pts = []; for (let x = -5; x <= 5; x += 0.05) pts.push([x, D(x)]); P.clipped(() => P.poly(pts, { color: UL.COLORS[1], w: 2.4 }));
      P.line(-5, 0.5, 5, 0.5, { color: UL.FAINT, dash: [3, 4], w: 1 });
      let dr = 0, dfk = 0; for (let i = 0; i < 400; i++) { dr += D(R[i]); dfk += D(F[i]); }
      calc.innerHTML = `<span class="k">Rounds</span><span class="hl">${rounds}</span><span class="k">Forger makes</span><span>x = ${f2(mu)} + ${f2(Math.exp(ls))} · noise</span><span class="k">D on real data</span><span>${(dr / 400).toFixed(2)}</span><span class="k">D on fakes</span><span>${(dfk / 400).toFixed(2)}</span><span class="k">Reading</span><span style="font-family:var(--font)">${dfk / 400 > 0.4 ? 'The fakes now pass about as often as real data.' : 'The detective still catches most fakes.'}</span>`;
      st.innerHTML = `round <b>${rounds}</b>`;
    }
    let t = 0; const loop = UL.loop((dt) => { t += dt; if (t > 0.05) { t = 0; for (let k = 0; k < 3; k++) round(); draw(); if (rounds > 1500) { $('#gn-play').textContent = 'Play'; return false; } } });
    $('#gn-play').addEventListener('click', () => { if (loop.on) { loop.stop(); $('#gn-play').textContent = 'Play'; } else { loop.start(); $('#gn-play').textContent = 'Pause'; } });
    $('#gn-step').addEventListener('click', () => { round(); draw(); });
    $('#gn-reset').addEventListener('click', () => { loop.stop(); $('#gn-play').textContent = 'Play'; reset(); });
    UL.seg('gn-data', (v) => { data = v; loop.stop(); $('#gn-play').textContent = 'Play'; reset(); });
    reset();
  })();

  /* ------------------------------------------------------------ attention */
  (function () {
    const S2 = [
      { w: ['The', 'animal', 'didn’t', 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'tired'], it: { 1: 0.62, 5: 0.12, 10: 0.14, 7: 0.05 }, cap: '“tired” describes an animal, so “it” attends mostly to <b>animal</b>.' },
      { w: ['The', 'animal', 'didn’t', 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'wide'], it: { 1: 0.12, 5: 0.6, 10: 0.16, 7: 0.05 }, cap: '“wide” describes a street, so now “it” attends mostly to <b>street</b>. One changed word at the end changes what “it” means.' },
    ];
    let s = 0;
    const att = (k, sent) => { // hand-made attention row for word k
      const n = sent.w.length, row = new Array(n).fill(0.02);
      if (k === 7) Object.entries(sent.it).forEach(([j, v]) => { row[+j] = v; });
      else { row[k] = 0.5; if (k > 0) row[k - 1] = 0.2; if (k < n - 1) row[k + 1] = 0.15; }
      const z = row.reduce((a, b) => a + b, 0); return row.map((v) => v / z);
    };
    function draw(k = 7) {
      const sent = S2[s], row = att(k, sent);
      $('#at-tokens').innerHTML = sent.w.map((w, j) => `<span data-j="${j}" style="background:rgba(42,120,214,${Math.min(0.9, row[j] * 1.5)});color:${row[j] > 0.35 ? '#fff' : '#111'};${j === k ? 'outline:2px solid #eb6834;' : ''}cursor:pointer">${w}</span>`).join('');
      $$('#at-tokens span').forEach((sp) => sp.addEventListener('mouseenter', () => draw(+sp.dataset.j)));
      $('#at-cap').innerHTML = k === 7 ? sent.cap : `“${sent.w[k]}” attends mostly to itself and its neighbours. Hover <b>it</b>.`;
    }
    UL.seg('at-s', (v) => { s = +v; draw(); }); draw();
  })();

  /* ------------------------------------------------------------ exercise 3 */
  (function () {
    const Q = [['A bank predicts loan default from 20 numbers on an application form.', 'MLP', 'a short, fixed list of numbers'], ['A phone unlocks when it recognises its owner’s face.', 'CNN', 'an image: local patterns such as eyes and edges'], ['A device forecasts the next hour’s electricity demand from past readings.', 'RNN / LSTM', 'a sequence over time, small and fast'], ['A game studio needs thousands of realistic fake faces.', 'GAN', 'creates new, realistic images'], ['A chatbot answers questions about the course handbook.', 'transformer', 'long text: attention over the question and the handbook']];
    const opts = ['—', 'MLP', 'CNN', 'RNN / LSTM', 'GAN', 'transformer'];
    $('#ex3-q').innerHTML = Q.map((q, i) => `<div>${i + 1}. ${q[0]}</div><div><select data-i="${i}">${opts.map((o) => `<option>${o}</option>`).join('')}</select></div><div class="res" id="ex3-r${i}"></div>`).join('');
    $('#ex3-check').addEventListener('click', () => {
      let ok = 0; Q.forEach((q, i) => { const v = $(`select[data-i="${i}"]`).value, good = v === q[1]; if (good) ok++; $('#ex3-r' + i).innerHTML = good ? `<span class="pos">✓ ${q[2]}</span>` : `<span class="neg">✗ ${q[1]}: ${q[2]}</span>`; });
      $('#ex3-score').innerHTML = `<b>${ok}</b> of 5 · The shape of the data decides the network.`;
    });
  })();

  UL.quiz('qz', [
    { q: 'Why is a CNN better than a dense network for photos?', o: ['It has more weights', 'It looks at small neighbourhoods with shared filters, so it keeps the grid structure and needs far fewer weights', 'It has memory', 'It uses two networks'], a: 1, why: 'A 3 × 3 filter has 9 weights and finds its pattern anywhere in the image.' },
    { q: 'What does the hidden state of an RNN hold?', o: ['The final answer', 'A memory of the sequence read so far', 'The filter weights', 'Random noise'], a: 1, why: 'h_t is updated at every step from h_{t−1} and the new input.' },
    { q: 'Why do simple RNNs struggle with long sentences?', o: ['They are too fast', 'The influence of early words fades step by step (vanishing gradient)', 'They cannot read words', 'They need images'], a: 1, why: 'Watch the memory chart in § 03. LSTMs and transformers fix this in different ways.' },
    { q: 'In a GAN, what does the discriminator do?', o: ['Generates images', 'Says whether an example is real or fake', 'Compresses data', 'Chooses the learning rate'], a: 1, why: 'It is a classifier with one output; its feedback is how the generator learns.' },
    { q: 'Which network type is behind ChatGPT and Claude?', o: ['CNN', 'GAN', 'Transformer', 'Perceptron'], a: 2, why: 'Transformers use attention: every word looks at every other word directly.' },
  ]);
  UL.start();
})();
