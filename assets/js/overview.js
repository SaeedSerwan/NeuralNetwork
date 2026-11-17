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

  /* ------------------------------------------------------------ the loop with one weight */
  (function () {
    const X = [1, 2, 3, 4, 5], Y = [2.3, 3.8, 6.4, 7.7, 10.2], eta = 0.01;
    const P = new UL.Plot('#lp-plot', { w: 520, h: 320, pad: { l: 36, r: 12, t: 12, b: 32 } }), calc = $('#lp-calc'), steps = $$('#lp-steps > div');
    let w, phase, round;
    const cost = (w) => X.reduce((s, x, i) => s + (w * x - Y[i]) ** 2, 0), grad = (w) => X.reduce((s, x, i) => s + 2 * (w * x - Y[i]) * x, 0);
    const reset = () => { w = 0.4; phase = 0; round = 0; draw(); };
    function draw() {
      P.clear(); P.domain(0, 5.6, 0, 12); P.axes({ xlabel: 'input x' });
      if (phase >= 1) X.forEach((x, i) => { if (phase >= 2) P.line(x, Y[i], x, w * x, { color: '#d64045', w: 2 }); });
      if (phase >= 1) P.poly([[0, 0], [5.6, 5.6 * w]], { color: UL.COLORS[0], w: 2.4 });
      X.forEach((x, i) => P.marker(x, Y[i], { r: 5.5, fill: UL.INK }));
      steps.forEach((s, k) => s.classList.toggle('on', k === phase - 1));
      const g = grad(w), C = cost(w);
      const T = [
        `<div><b>Round ${round + 1}</b>. The weight is <b>w = ${w.toFixed(3)}</b>. Black dots: five examples (x, right answer). Press <b>Next step</b>.</div>`,
        `<div><b>1 · Guess.</b> The model predicts w · x for every example: the blue line. For x = 3 it says ${(w * 3).toFixed(2)}; the right answer is 6.4.</div>`,
        `<div><b>2 · Measure.</b> Square the red gaps and add them up: cost = <b>${C.toFixed(2)}</b>. One number for how wrong the model is.</div>`,
        `<div><b>3 · Find the direction.</b> The slope of the cost is ∂C/∂w = <b>${g.toFixed(2)}</b>. ${g < 0 ? 'Negative: making w bigger lowers the cost.' : 'Positive: making w smaller lowers the cost.'}</div>`,
        `<div><b>4 · Nudge.</b> w ← w − η · slope = ${w.toFixed(3)} − ${eta} × ${g.toFixed(2)} = <b>${(w - eta * g).toFixed(3)}</b>. Then the loop starts again.</div>`,
      ];
      calc.innerHTML = T[phase];
      $('#lp-status').innerHTML = `round <b>${round + 1}</b> · w = <b>${w.toFixed(3)}</b> · cost <b>${C.toFixed(2)}</b>`;
    }
    $('#lp-next').addEventListener('click', () => { if (phase === 4) { w -= eta * grad(w); round++; phase = 1; } else phase++; draw(); });
    $('#lp-run').addEventListener('click', () => { for (let k = 0; k < 20; k++) { w -= eta * grad(w); round++; } phase = 2; draw(); });
    $('#lp-reset').addEventListener('click', reset);
    reset();
  })();

  UL.quiz('qz', [
    { q: 'Why do we not write if-statements to read handwritten digits?', o: ['Computers cannot compare numbers', 'Nobody can write exact rules that cover every way of writing a 3; learning from examples works better', 'It would be too fast', 'Digits are not numbers'], a: 1, why: 'No two 3s have the same pixels. The rules are too many and too fuzzy to write by hand.' },
    { q: 'In an artificial neuron, what plays the role of the synapse strength?', o: ['The bias', 'The weight', 'The activation function', 'The layer'], a: 1, why: 'Each connection has a weight, like the strength of a synapse; learning changes the weights.' },
    { q: 'Put the learning loop in order.', o: ['Nudge, guess, measure, direction', 'Guess, measure the cost, find the gradient, nudge the weights', 'Measure, nudge, guess, direction', 'Guess, nudge, measure, direction'], a: 1, why: 'Forward pass → cost → gradient (backpropagation) → gradient descent step. Repeat.' },
  ]);
  UL.start();
})();
