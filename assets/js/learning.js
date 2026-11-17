/* Part 2 · How a network learns: cost, gradient descent, backpropagation
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, $ = UL.$, $$ = UL.$$, S = UL.samples(), trained = UL.pretrained();
  const f2 = (v) => UL.fmt(v, 2), f3 = (v) => UL.fmt(v, 3);
  const onehot = (d) => Array.from({ length: 10 }, (_, j) => (j === d ? 1 : 0));

  /* ------------------------------------------------------------ 01 random vs trained */
  (function () {
    let seed = 11, rnd = new UL.MLP([784, 16, 16, 10], { seed }), idx = 0;
    const costLine = (net, d) => { const o = net.forward(S.get(idx)), c = net.cost(onehot(d)); return { o: Float64Array.from(o), c }; };
    function draw() {
      const d = S.label(idx), A = costLine(rnd, d), B = costLine(trained, d);
      UL.outbars('rd-a', A.o, d); UL.outbars('rd-b', B.o, d);
      const row = (r) => `<span class="k">Answer</span><span class="hl">${UL.argmax(r.o)} ${UL.argmax(r.o) === d ? '✓' : '✗'}</span><span class="k">Cost</span><span class="hl" style="font-size:1.15rem">${r.c.toFixed(2)}</span>`;
      $('#rd-ca').innerHTML = row(A); $('#rd-cb').innerHTML = row(B);
    }
    UL.thumbs('rd-thumbs', [0, 5, S.ofDigit(7), S.ofDigit(1), S.ofDigit(8), S.ofDigit(0), S.ofDigit(4)], (i) => { idx = i; draw(); }, { select: 0 });
    $('#rd-new').addEventListener('click', () => { rnd = new UL.MLP([784, 16, 16, 10], { seed: ++seed }); draw(); });
    draw();
  })();

  /* ------------------------------------------------------------ 03 gradient descent in 1-D */
  (function () {
    const C = (w) => 0.05 * w ** 4 - 0.5 * w ** 2 + 0.3 * w + 2.2, dC = (w) => 0.2 * w ** 3 - w + 0.3;
    const P = new UL.Plot('#g1-plot', { w: 560, h: 340, pad: { l: 36, r: 14, t: 14, b: 34 } }), calc = $('#g1-calc'), st = $('#g1-status');
    const eta = UL.range('g1-eta', () => draw(), (v) => v.toFixed(2));
    let w = 3.4, trail = [w], n = 0;
    function draw(prev) {
      P.clear(); P.domain(-4, 4, 0, 5); P.axes({ xlabel: 'weight w', ylabel: 'cost C' });
      const pts = []; for (let x = -4; x <= 4; x += 0.02) pts.push([x, C(x)]); P.clipped(() => P.poly(pts, { color: UL.INK, w: 2.2 }));
      P.clipped(() => trail.forEach((t, i) => P.marker(t, C(t), { r: 3, fill: UL.COLORS[0], alpha: 0.25 + (0.5 * i) / trail.length, ring: false })));
      const g = dC(w), cw = Math.max(-4, Math.min(4, w));
      if (Math.abs(w) <= 4) { P.clipped(() => P.line(w - 0.7, C(w) - 0.7 * g, w + 0.7, C(w) + 0.7 * g, { color: UL.COLORS[1], w: 1.6 })); P.marker(w, C(w), { r: 8, fill: UL.COLORS[0] }); }
      const step = -eta.v * g;
      if (Math.abs(w) <= 4 && Math.abs(step) > 0.02) P.clipped(() => P.arrow(w, C(w) + 0.35, Math.max(-4, Math.min(4, w + step)), C(w) + 0.35, { color: UL.COLORS[2], w: 1.8, head: 8 }));
      calc.innerHTML = `<span class="k">w</span><span class="hl">${f3(w)}</span><span class="k">cost C(w)</span><span>${Math.abs(w) > 50 ? '∞' : f3(C(w))}</span><span class="k">slope ∂C/∂w</span><span class="${g > 0 ? 'neg' : 'pos'}">${Math.abs(g) > 1e4 ? 'huge' : f3(g)}</span>` +
        `<span class="k">step −η · slope</span><span>${Math.abs(step) > 1e4 ? 'huge' : f3(step)} → ${g > 0 ? 'move left' : 'move right'}</span><span class="k">next w</span><span>${Math.abs(w + step) > 1e4 ? 'off the chart' : f3(w + step)}</span>` +
        (Math.abs(cw - w) > 0 ? '<span class="k">!</span><span class="neg">The ball left the picture: η is too large and the steps overshoot.</span>' : '');
      st.innerHTML = `steps <b>${n}</b>`;
    }
    const doStep = () => { if (Math.abs(w) > 1e6) return; w = w - eta.v * dC(w); trail.push(w); n++; if (trail.length > 60) trail.shift(); };
    let t = 0; const loop = UL.loop((dt) => { t += dt; if (t > 0.25) { t = 0; doStep(); draw(); if (Math.abs(dC(w)) < 1e-4 || Math.abs(w) > 1e6 || n > 300) { $('#g1-run').textContent = 'Run'; return false; } } });
    $('#g1-step').addEventListener('click', () => { doStep(); draw(); });
    $('#g1-run').addEventListener('click', () => { if (loop.on) { loop.stop(); $('#g1-run').textContent = 'Run'; } else { loop.start(); $('#g1-run').textContent = 'Pause'; } });
    P.cv.addEventListener('click', (e) => { const [X] = P.pointer(e); w = Math.max(-3.95, Math.min(3.95, P.ix(X))); trail = [w]; n = 0; draw(); });
    draw();
  })();

  /* ------------------------------------------------------------ 04 gradient descent in 2-D */
  (function () {
    const F = (x, y) => 3 - 2.2 * Math.exp(-((x - 1.5) ** 2 + (y - 1) ** 2) / 1.6) - 1.5 * Math.exp(-((x + 1.7) ** 2 + (y + 1.3) ** 2) / 0.9) + 0.06 * (x * x + y * y) + 0.25 * Math.exp(-((x + 0.2) ** 2 + (y - 0.1) ** 2) / 0.4);
    const G = (x, y) => { const h = 1e-4; return [(F(x + h, y) - F(x - h, y)) / (2 * h), (F(x, y + h) - F(x, y - h)) / (2 * h)]; };
    const P = new UL.Plot('#g2-plot', { w: 520, h: 440, pad: { l: 30, r: 10, t: 10, b: 28 } }), calc = $('#g2-calc');
    const eta = UL.range('g2-eta', () => {}, (v) => v.toFixed(2)), mom = $('#g2-mom'), noise = $('#g2-noise'), r = UL.rng(4);
    P.domain(-3.5, 3.5, -3, 3);
    // background, computed once
    const bg = document.createElement('canvas'); bg.width = P.cv.width; bg.height = P.cv.height;
    (function () {
      const c = bg.getContext('2d'); c.scale(P.dpr, P.dpr); const st = 4;
      for (let X = P.pad.l; X < P.W - P.pad.r; X += st) for (let Y = P.pad.t; Y < P.H - P.pad.b; Y += st) {
        const v = F(P.ix(X + 2), P.iy(Y + 2)), t = Math.max(0, Math.min(1, (v - 0.9) / 2.4)), k = Math.round(40 + 215 * t);
        c.fillStyle = `rgb(${Math.round(k * 0.92)},${Math.round(k * 0.95)},${k})`; c.fillRect(X, Y, st, st);
        const lv = v * 4, lv2 = F(P.ix(X + 6), P.iy(Y + 2)) * 4, lv3 = F(P.ix(X + 2), P.iy(Y + 6)) * 4;
        if (Math.floor(lv) !== Math.floor(lv2) || Math.floor(lv) !== Math.floor(lv3)) { c.fillStyle = 'rgba(255,255,255,.35)'; c.fillRect(X, Y, 1.5, 1.5); }
      }
    })();
    let balls = [];
    function draw() {
      P.ctx.setTransform(1, 0, 0, 1, 0, 0); P.ctx.drawImage(bg, 0, 0); P.ctx.setTransform(P.dpr, 0, 0, P.dpr, 0, 0); P.axes({ xlabel: 'weight w₁' });
      balls.forEach((b, k) => {
        const col = UL.COLORS[(k + 1) % 8];
        P.clipped(() => P.poly(b.path, { color: col, w: 2 }));
        const [x, y] = b.path[b.path.length - 1]; P.marker(x, y, { r: 6, fill: col });
        const [gx, gy] = G(x, y), L = Math.hypot(gx, gy) || 1;
        if (L > 0.01) P.clipped(() => P.arrow(x, y, x - (gx / L) * 0.7, y - (gy / L) * 0.7, { color: '#fff', w: 2, head: 7 }));
      });
      const b = balls[balls.length - 1];
      if (!b) { calc.innerHTML = '<span class="k">Start</span><span>click the landscape</span><span class="k">Two valleys</span><span>top right is wider, bottom left is deeper</span>'; return; }
      const [x, y] = b.path[b.path.length - 1], [gx, gy] = G(x, y);
      calc.innerHTML = `<span class="k">Weights</span><span>w₁ = ${f2(x)}, w₂ = ${f2(y)}</span><span class="k">Cost</span><span class="hl">${f3(F(x, y))}</span><span class="k">Gradient</span><span>(${f2(gx)}, ${f2(gy)})</span><span class="k">Step</span><span>−η × gradient = (${f2(-eta.v * gx)}, ${f2(-eta.v * gy)})</span><span class="k">Steps taken</span><span>${b.path.length - 1}</span>`;
    }
    const step = () => balls.forEach((b) => {
      const [x, y] = b.path[b.path.length - 1]; let [gx, gy] = G(x, y);
      if (noise.checked) { gx += r.normal() * 0.5; gy += r.normal() * 0.5; }
      const m = mom.checked ? 0.85 : 0; b.v = [m * b.v[0] - eta.v * gx, m * b.v[1] - eta.v * gy];
      const nx = Math.max(-3.5, Math.min(3.5, x + b.v[0])), ny = Math.max(-3, Math.min(3, y + b.v[1])); b.path.push([nx, ny]); if (b.path.length > 400) b.path.shift();
    });
    let t = 0, steps = 0; const loop = UL.loop((dt) => { t += dt; if (t > 0.06) { t = 0; step(); steps++; draw(); if (steps > 250) { $('#g2-run').textContent = 'Run'; return false; } } });
    P.cv.addEventListener('click', (e) => { const [X, Y] = P.pointer(e); balls.push({ path: [[P.ix(X), P.iy(Y)]], v: [0, 0] }); if (balls.length > 6) balls.shift(); steps = 0; draw(); if (!loop.on) { loop.start(); $('#g2-run').textContent = 'Pause'; } });
    $('#g2-run').addEventListener('click', () => { if (loop.on) { loop.stop(); $('#g2-run').textContent = 'Run'; } else { if (!balls.length) balls.push({ path: [[-0.4, 2.6]], v: [0, 0] }); steps = 0; loop.start(); $('#g2-run').textContent = 'Pause'; } });
    $('#g2-clear').addEventListener('click', () => { balls = []; draw(); });
    balls.push({ path: [[-0.4, 2.6]], v: [0, 0] }); draw();
  })();

  /* ------------------------------------------------------------ 05 gradient per knob */
  (function () {
    const P = new UL.Plot('#kn-plot', { w: 560, h: 300, pad: { l: 44, r: 10, t: 16, b: 30 } }), calc = $('#kn-calc');
    let seed = 2, bseed = 1;
    function run() {
      const net = new UL.MLP([784, 16, 16, 10], { seed }), r = UL.rng(bseed);
      for (let k = 0; k < 32; k++) { const i = r.int(S.n); net.forward(S.get(i)); net.backward(onehot(S.label(i))); }
      const g = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => net.gW[1][i * 16 + ((i * 5) % 16)] / 32), big = Math.max(...g.map(Math.abs));
      UL.bars(P, g.map((v, i) => ({ v, label: 'w' + (i + 1), color: v > 0 ? '#d64045' : '#1b8c5c' })), { fv: (v) => UL.fmt(v, 4), ylabel: 'gradient' });
      const ord = g.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]), hi = ord[0], lo = ord[7];
      calc.innerHTML = `<span class="k">Biggest</span><span>w${hi[1] + 1}: the cost cares most about this knob</span><span class="k">Smallest</span><span>w${lo[1] + 1}: hardly matters right now</span><span class="k">Ratio</span><span class="hl">w${hi[1] + 1} matters about ${Math.round(hi[0] / Math.max(1e-9, ord[5][0]))}× more than w${ord[5][1] + 1}</span><span class="k">Red, above 0</span><span>nudge this weight <b>down</b></span><span class="k">Green, below 0</span><span>nudge this weight <b>up</b></span>`;
      return big;
    }
    $('#kn-new').addEventListener('click', () => { seed++; run(); }); $('#kn-batch').addEventListener('click', () => { bseed++; run(); }); run();
  })();

  /* ------------------------------------------------------------ 06 backpropagation by hand */
  (function () {
    const sig = UL.sigmoid, x = [1.0, 0.5], y = 1, eta = 1;
    let W1, b1, W2, b2;
    const init = () => { W1 = [[0.4, -0.6], [0.7, 0.2]]; b1 = [0.1, -0.3]; W2 = [0.5, -0.8]; b2 = 0.2; };
    init();
    function compute() {
      const z1 = W1.map((r, j) => r[0] * x[0] + r[1] * x[1] + b1[j]), a1 = z1.map(sig), z2 = W2[0] * a1[0] + W2[1] * a1[1] + b2, a2 = sig(z2), C = (a2 - y) ** 2;
      const dCda2 = 2 * (a2 - y), da2dz2 = a2 * (1 - a2), d2 = dCda2 * da2dz2, gW2 = a1.map((a) => d2 * a), gb2 = d2;
      const d1 = a1.map((a, j) => d2 * W2[j] * a * (1 - a)), gW1 = d1.map((d) => x.map((xi) => d * xi)), gb1 = d1;
      return { z1, a1, z2, a2, C, dCda2, da2dz2, d2, gW2, gb2, d1, gW1, gb1 };
    }
    let base = compute(), k = 0;
    const NX = [70, 280, 490], NY = [[90, 250], [90, 250], [170]], el = $('#bp-svg'), calc = $('#bp-calc'), st = $('#bp-status');
    const STEPS = [
      { t: 'The network', show: [], txt: () => 'Inputs x = (1.0, 0.5). The weights are written on the connections, the biases under the neurons. We want the output to be <b>1</b>.' },
      { t: 'Forward: hidden neuron 1', show: ['h0'], txt: (v) => `z = 0.4·1.0 + (−0.6)·0.5 + 0.1 = <b>${f3(v.z1[0])}</b><br>a = σ(z) = <b>${f3(v.a1[0])}</b>` },
      { t: 'Forward: hidden neuron 2', show: ['h0', 'h1'], txt: (v) => `z = 0.7·1.0 + 0.2·0.5 − 0.3 = <b>${f3(v.z1[1])}</b><br>a = σ(z) = <b>${f3(v.a1[1])}</b>` },
      { t: 'Forward: output neuron', show: ['h0', 'h1', 'o'], txt: (v) => `z = 0.5·${f3(v.a1[0])} + (−0.8)·${f3(v.a1[1])} + 0.2 = <b>${f3(v.z2)}</b><br>a = σ(z) = <b>${f3(v.a2)}</b>` },
      { t: 'The cost', show: ['h0', 'h1', 'o', 'C'], txt: (v) => `C = (a − y)² = (${f3(v.a2)} − 1)² = <b>${f3(v.C)}</b>. The output is too low.` },
      { t: 'Backward: ∂C/∂a at the output', show: ['h0', 'h1', 'o', 'C', 'dC'], back: true, txt: (v) => `∂C/∂a = 2(a − y) = <b>${f3(v.dCda2)}</b>. Negative: raising the output lowers the cost.` },
      { t: 'Backward: through the sigmoid', show: ['h0', 'h1', 'o', 'C', 'dC', 'd2'], back: true, txt: (v) => `∂a/∂z = a(1 − a) = ${f3(v.da2dz2)}<br>δ = ∂C/∂z = ${f3(v.dCda2)} × ${f3(v.da2dz2)} = <b>${UL.fmt(v.d2, 4)}</b>` },
      { t: 'Backward: gradients of the output weights', show: ['h0', 'h1', 'o', 'C', 'dC', 'd2', 'gW2'], back: true, txt: (v) => `∂z/∂w = the activation the weight multiplies, so<br>∂C/∂w₁ = ${f3(v.a1[0])} × ${UL.fmt(v.d2, 4)} = <b>${UL.fmt(v.gW2[0], 4)}</b><br>∂C/∂w₂ = ${f3(v.a1[1])} × ${UL.fmt(v.d2, 4)} = <b>${UL.fmt(v.gW2[1], 4)}</b><br>∂C/∂b = δ = <b>${UL.fmt(v.gb2, 4)}</b>` },
      { t: 'Backward: pass the blame to the hidden neurons', show: ['h0', 'h1', 'o', 'C', 'dC', 'd2', 'gW2', 'd1'], back: true, txt: (v) => `Each hidden neuron gets δ × its weight to the output, times its own sigmoid slope:<br>δ₁ = ${UL.fmt(v.d2, 4)} × 0.5 × ${f3(v.a1[0] * (1 - v.a1[0]))} = <b>${UL.fmt(v.d1[0], 4)}</b><br>δ₂ = ${UL.fmt(v.d2, 4)} × (−0.8) × ${f3(v.a1[1] * (1 - v.a1[1]))} = <b>${UL.fmt(v.d1[1], 4)}</b>` },
      { t: 'Backward: gradients of the first-layer weights', show: ['h0', 'h1', 'o', 'C', 'dC', 'd2', 'gW2', 'd1', 'gW1'], back: true, txt: (v) => `∂C/∂w = input × δ:<br>neuron 1: (${UL.fmt(v.gW1[0][0], 4)}, ${UL.fmt(v.gW1[0][1], 4)})<br>neuron 2: (${UL.fmt(v.gW1[1][0], 4)}, ${UL.fmt(v.gW1[1][1], 4)})` },
      { t: 'Update: one gradient descent step (η = 1)', show: ['upd'], txt: (v) => `Every weight: w ← w − η · ∂C/∂w.<br>Output weights: 0.5 → <b>${f3(0.5 - eta * v.gW2[0])}</b>, −0.8 → <b>${f3(-0.8 - eta * v.gW2[1])}</b>.<br>Run the forward pass again: the cost falls from ${f3(base.C)} to <b>${f3(after().C)}</b>.` },
    ];
    function after() { const s = [W1.map((r) => r.slice()), b1.slice(), W2.slice(), b2]; W1 = W1.map((r, j) => r.map((w, i) => w - eta * base.gW1[j][i])); b1 = b1.map((b, j) => b - eta * base.gb1[j]); W2 = W2.map((w, j) => w - eta * base.gW2[j]); b2 = b2 - eta * base.gb2; const v = compute(); [W1, b1, W2, b2] = s; return v; }
    function draw() {
      const S2 = STEPS[k], sh = new Set(S2.show), upd = sh.has('upd'), v = upd ? after() : base;
      const show = (key) => upd || sh.has(key), FW = '#2a78d6', BW = '#eb6834';
      const ww = upd ? [W1.map((r, j) => r.map((w, i) => w - eta * base.gW1[j][i])), W2.map((w, j) => w - eta * base.gW2[j])] : [W1, W2];
      let s = '';
      const edge = (x1, y1, x2, y2, w, g, gshow, on) => { const mx = x1 + (x2 - x1) * 0.62, my = y1 + (y2 - y1) * 0.62; s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${on ? '#111' : '#c9c9c4'}" stroke-width="${1 + Math.min(4, Math.abs(w) * 3)}" stroke-opacity=".55"/>`; s += `<text x="${mx}" y="${my - 6}" text-anchor="middle" style="font:11px var(--mono);fill:${w >= 0 ? '#1b8c5c' : '#d64045'}">w=${f2(w)}</text>`; if (gshow) s += `<text x="${mx}" y="${my + 12}" text-anchor="middle" style="font:500 10.5px var(--mono);fill:${BW}">∂C/∂w=${UL.fmt(g, 3)}</text>`; };
      for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) edge(NX[0] + 22, NY[0][i], NX[1] - 22, NY[1][j], ww[0][j][i], base.gW1[j][i], sh.has('gW1'), show('h' + j));
      for (let j = 0; j < 2; j++) edge(NX[1] + 22, NY[1][j], NX[2] - 22, NY[2][0], ww[1][j], base.gW2[j], sh.has('gW2'), show('o'));
      const node = (x0, y0, label, val, on, sub, back) => { s += `<circle cx="${x0}" cy="${y0}" r="22" fill="${on ? '#e8f0fb' : '#fff'}" stroke="${on ? FW : '#111'}" stroke-width="1.5"/><text x="${x0}" y="${y0 + 4}" text-anchor="middle" style="font:500 12px var(--mono);fill:#111">${val}</text><text x="${x0}" y="${y0 - 30}" text-anchor="middle" style="font:11px var(--font);fill:#6b6b68">${label}</text>`; if (sub) s += `<text x="${x0}" y="${y0 + 38}" text-anchor="middle" style="font:10.5px var(--mono);fill:#6b6b68">${sub}</text>`; if (back) s += `<text x="${x0}" y="${y0 + 52}" text-anchor="middle" style="font:500 10.5px var(--mono);fill:${BW}">${back}</text>`; };
      node(NX[0], NY[0][0], 'x₁', '1.0', false); node(NX[0], NY[0][1], 'x₂', '0.5', false);
      for (let j = 0; j < 2; j++) node(NX[1], NY[1][j], 'hidden ' + (j + 1), show('h' + j) ? f3(v.a1[j]) : '?', show('h' + j), 'b=' + f2(upd ? b1[j] - eta * base.gb1[j] : b1[j]), sh.has('d1') ? 'δ=' + UL.fmt(base.d1[j], 3) : '');
      node(NX[2], NY[2][0], 'output', show('o') ? f3(v.a2) : '?', show('o'), 'b=' + f2(upd ? b2 - eta * base.gb2 : b2), sh.has('d2') ? 'δ=' + UL.fmt(base.d2, 3) : '');
      s += `<rect x="570" y="140" width="96" height="60" rx="8" fill="${show('C') ? '#fff4d6' : '#fff'}" stroke="#111"/><text x="618" y="163" text-anchor="middle" style="font:11px var(--font);fill:#6b6b68">cost (y = 1)</text><text x="618" y="185" text-anchor="middle" style="font:500 13px var(--mono);fill:#111">${show('C') ? f3(v.C) : '?'}</text>`;
      s += `<line x1="512" y1="170" x2="570" y2="170" stroke="#111"/>`;
      if (sh.has('dC')) s += `<text x="618" y="222" text-anchor="middle" style="font:500 10.5px var(--mono);fill:${BW}">∂C/∂a=${f3(base.dCda2)}</text>`;
      el.innerHTML = `<svg class="netsvg" viewBox="0 0 680 320">${s}</svg>`;
      calc.innerHTML = `<div><b>${S2.t}</b></div><div style="line-height:1.9">${S2.txt(base)}</div>`;
      st.innerHTML = `step <b>${k}</b> of ${STEPS.length - 1} · ${S2.back ? '<span style="color:#eb6834">backward</span>' : k === 0 ? '' : upd ? 'update' : '<span style="color:#2a78d6">forward</span>'}`;
    }
    $('#bp-next').addEventListener('click', () => { if (k < STEPS.length - 1) k++; else { // apply the update and start a new round
      W1 = W1.map((r, j) => r.map((w, i) => w - eta * base.gW1[j][i])); b1 = b1.map((b, j) => b - eta * base.gb1[j]); W2 = W2.map((w, j) => w - eta * base.gW2[j]); b2 -= eta * base.gb2; base = compute(); k = 1; } draw(); });
    $('#bp-back').addEventListener('click', () => { if (k > 0) k--; draw(); });
    $('#bp-reset').addEventListener('click', () => { init(); base = compute(); k = 0; draw(); });
    draw();
  })();

  /* ------------------------------------------------------------ 07 training numbers */
  UL.stats('tr-stats', [{ k: 'Cost on the test set', v: '3.05 → 0.08', hi: 1 }, { k: 'Test accuracy', v: '9.7% → 94.9%', hi: 1 }, { k: 'Mini-batch', v: '32 digits' }, { k: 'Epochs', v: '30' }, { k: 'Learning rate η', v: '3' }]);

  /* ------------------------------------------------------------ 08 exercise 2 */
  (function () {
    const vals = [0.1, 0.6, 0, 0, 0, 0, 0, 0.3, 0, 0]; let want = 1;
    const el = $('#cc-in'), out = $('#cc-out');
    el.innerHTML = vals.map((v, j) => `<label class="ctl">${j} <input class="num" type="number" min="0" max="1" step="0.1" value="${v}" data-j="${j}" style="width:4.2rem"></label>`).join('') + `<label class="ctl">correct digit <select id="cc-want">${[...Array(10).keys()].map((d) => `<option ${d === 1 ? 'selected' : ''}>${d}</option>`).join('')}</select></label>`;
    function upd() {
      const terms = vals.map((a, j) => [j, a, j === want ? 1 : 0]).filter(([, a, y]) => a !== y), c = vals.reduce((s, a, j) => s + (a - (j === want ? 1 : 0)) ** 2, 0);
      out.innerHTML = `<span class="k">Terms that are not 0</span><span>${terms.map(([j, a, y]) => `(${a} − ${y})²`).join(' + ') || '—'}</span><span class="k">Cost</span><span class="hl" style="font-size:1.1rem">${c.toFixed(2)}</span>`;
    }
    $$('input', el).forEach((i) => i.addEventListener('input', () => { vals[+i.dataset.j] = +i.value || 0; upd(); }));
    $('#cc-want').addEventListener('change', (e) => { want = +e.target.value; upd(); }); upd();
    UL.exercises('ex2-list', [
      { q: 'A network sees a “1”. Its outputs are 0.1 for “0”, 0.6 for “1”, 0.3 for “7” and 0 for the others. What is the cost of this example?', a: '0.1² + (0.6 − 1)² + 0.3² = 0.01 + 0.16 + 0.09 = <b>0.26</b>.' },
      { q: 'w = 0.5, gradient = −2, η = 0.1. What is the new w? Did w go up or down, and why?', a: '0.5 − 0.1 · (−2) = <b>0.7</b>: up, because the slope is negative, so the cost falls when w increases.' },
      { q: 'How many inputs does the cost function of the 784 → 16 → 16 → 10 network have? How many outputs?', a: '<b>13,002 inputs</b> (every weight and bias) and <b>1 output</b> (the cost).' },
      { q: 'A ball stops in a valley that is not the lowest. What is this called? Name one thing you could try.', a: 'A <b>local minimum</b>. Try a different random start, a larger learning rate, or momentum (as in Adam). Try it in § 04.' },
      { q: 'Accuracy before training is about 10%. Why exactly 10%?', a: 'Random weights guess one of 10 digits more or less at random: right 1 time in 10.' },
    ]);
  })();

  UL.quiz('qz', [
    { q: 'The cost of one example is 0. What does that mean?', o: ['The network is untrained', 'Every output is exactly what we wanted', 'The gradient is very large', 'All weights are 0'], a: 1, why: 'Each squared difference is 0 only when the output equals the wanted value.' },
    { q: 'The slope ∂C/∂w of a weight is positive. Gradient descent will…', o: ['increase w', 'decrease w', 'leave w unchanged', 'set w to 0'], a: 1, why: 'A positive slope means the cost rises when w rises, so w ← w − η · (positive) makes w smaller.' },
    { q: 'What happens with a learning rate that is much too large?', o: ['Training is slow but safe', 'The steps overshoot, the cost jumps around or grows', 'The network gets more layers', 'Nothing'], a: 1, why: 'Try η = 1.5 in § 03: the ball jumps over the valley and can fly off the curve.' },
    { q: 'What does backpropagation compute?', o: ['The output of the network', 'The gradient of the cost with respect to every weight and bias', 'The learning rate', 'The test accuracy'], a: 1, why: 'Backpropagation computes the gradient; gradient descent then uses it to update the weights.' },
    { q: 'Why train on mini-batches of 32 digits instead of all 60,000 at once?', o: ['It is more accurate', 'Each step is much cheaper, so we take many more steps; on average they still go downhill', 'It avoids the cost function', 'MNIST requires it'], a: 1, why: 'The mini-batch gradient is a noisy but cheap estimate of the full gradient.' },
    { q: 'In ∂C/∂w = ∂z/∂w · ∂a/∂z · ∂C/∂a, what is ∂z/∂w?', o: ['The learning rate', 'The input activation that the weight multiplies', 'The cost', 'Always 1'], a: 1, why: 'z = w·a + …, so changing w by a little changes z by a times that little.' },
  ]);
  UL.start();
})();
