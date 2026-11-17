/* Neural Network Lab — animation engine
   An SVG network whose signals visibly travel: forward pulses (blue) carry activations along the weights,
   backward pulses (orange) carry the error back, neurons fill up as values arrive, weights thicken or thin
   as they are updated. A Player runs a list of narrated steps with Play / Pause / Step / Back and a speed control.
   Ideas adapted from NeuroViz (step-aligned forward, backward and update animation), CNN Explainer
   (overview + detail with a live formula) and TensorFlow Playground (weights as coloured lines).
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, NS = 'http://www.w3.org/2000/svg';
  const FW = '#2a78d6', BW = '#eb6834', POS = '#1b8c5c', NEG = '#d64045';
  const mk = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

  /* ------------------------------------------------------------ a clock that can pause and change speed */
  class Clock {
    constructor() { this.speed = 1; this.paused = false; this.gen = 0; }
    // run fn(t) for t: 0 → 1 over ms milliseconds (scaled by speed, frozen while paused)
    tween(ms, fn) {
      const gen = this.gen;
      return new Promise((res, rej) => {
        let t = 0, last = performance.now();
        const frame = (now) => {
          if (gen !== this.gen) { rej(new Error('cancelled')); return; }
          const dt = now - last; last = now; if (!this.paused) t += (dt * this.speed) / ms;
          fn(ease(Math.min(1, t)), Math.min(1, t)); if (t >= 1) res(); else requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
      });
    }
    wait(ms) { return this.tween(ms, () => {}); }
    cancel() { this.gen++; }
  }
  UL.Clock = Clock;

  /* ------------------------------------------------------------ the network picture */
  class NetAnim {
    // layers: [{n, label, names: [..] (optional), shown (optional)}]
    constructor(el, layers, o = {}) {
      this.el = typeof el === 'string' ? document.getElementById(el) : el; this.layers = layers; this.o = o;
      this.clock = o.clock || new Clock();
      const W = (this.W = o.w || 640), H = (this.H = o.h || 340), R = (this.R = o.r || 16), top = o.top || 38, bot = o.bottom || 26;
      const xs = o.xs || layers.map((_, i) => (o.left || 70) + (i * (W - (o.left || 70) - (o.right || 110))) / (layers.length - 1));
      this.pos = layers.map((L, i) => { const n = L.n, gap = Math.min(L.gap || o.gap || 2.6 * R + 10, (H - top - bot) / n); const y0 = (top + H - bot) / 2 - ((n - 1) * gap) / 2; return Array.from({ length: n }, (_, k) => [xs[i], y0 + k * gap]); });
      this.svg = mk('svg', { viewBox: `0 0 ${W} ${H}`, class: 'netsvg anim' });
      const defs = mk('defs', {}, this.svg);
      const glow = mk('filter', { id: 'glow' + (NetAnim.n = (NetAnim.n || 0) + 1), x: '-50%', y: '-50%', width: '200%', height: '200%' }, defs);
      mk('feGaussianBlur', { stdDeviation: 2.5, result: 'b' }, glow); const m = mk('feMerge', {}, glow); mk('feMergeNode', { in: 'b' }, m); mk('feMergeNode', { in: 'SourceGraphic' }, m);
      this.glowId = glow.id;
      this.gEdges = mk('g', {}, this.svg); this.gEdgeLab = mk('g', {}, this.svg); this.gPulse = mk('g', {}, this.svg); this.gNodes = mk('g', {}, this.svg); this.gLab = mk('g', {}, this.svg);
      this.edges = []; this.edgeLab = [];
      for (let l = 0; l < layers.length - 1; l++) {
        this.edges.push(this.pos[l + 1].map((q) => this.pos[l].map((p) => mk('line', { x1: p[0] + R, y1: p[1], x2: q[0] - R, y2: q[1], stroke: '#d6d6d2', 'stroke-width': 1, 'stroke-linecap': 'round' }, this.gEdges))));
        this.edgeLab.push(this.pos[l + 1].map((q, j) => this.pos[l].map((p, i) => { let f = Array.isArray(o.labelAt) ? o.labelAt[l] : o.labelAt != null ? o.labelAt : 'stagger'; const nOut = this.pos[l + 1].length; if (f === 'stagger') f = 0.24 + 0.05 * j; const t = mk('text', { x: p[0] + R + (q[0] - p[0] - 2 * R) * f, y: p[1] + (q[1] - p[1]) * f - 4, 'text-anchor': 'middle', class: 'elab' }, this.gEdgeLab); return t; })));
      }
      this.nodes = this.pos.map((P, l) => P.map(([x, y], k) => {
        const g = mk('g', { transform: `translate(${x},${y})`, class: 'nd' }, this.gNodes);
        const ring = mk('circle', { r: R, fill: '#fff', stroke: '#111', 'stroke-width': 1.3 }, g);
        const fill = mk('circle', { r: 0, fill: '#111' }, g);
        const txt = mk('text', { y: 4, 'text-anchor': 'middle', class: 'nval' }, g); txt.style.fontSize = Math.min(13, R * 0.62) + 'px';
        const sub = mk('text', { y: R + 14, 'text-anchor': 'middle', class: 'nsub' }, g);
        const side = mk('text', { x: l === layers.length - 1 ? R + 8 : -R - 8, y: 4, 'text-anchor': l === layers.length - 1 ? 'start' : 'end', class: 'nside' }, g);
        if (layers[l].names) side.textContent = layers[l].names[k];
        return { g, ring, fill, txt, sub, side, v: 0 };
      }));
      layers.forEach((L, l) => { if (L.label) { const t = mk('text', { x: this.pos[l][0][0], y: 16, 'text-anchor': 'middle', class: 'llab' }, this.gLab); t.textContent = L.label; } });
      this.el.innerHTML = ''; this.el.appendChild(this.svg);
      this.w = null; this.vals = layers.map((L) => new Array(L.n).fill(null));
    }
    /* --- static state */
    setNode(l, k, v, o = {}) { // v in 0…1 (or null = empty); text shown inside
      const nd = this.nodes[l][k]; nd.v = v;
      const a = v == null ? 0 : Math.max(0, Math.min(1, Math.abs(v)));
      nd.fill.setAttribute('r', this.R - 1.5); nd.fill.setAttribute('fill-opacity', 0.08 + 0.92 * a); nd.fill.setAttribute('opacity', v == null ? 0 : 1); nd.fill.setAttribute('fill', o.color || (v != null && v < 0 ? NEG : '#1d2433'));
      nd.txt.textContent = o.text != null ? o.text : v == null ? '' : (+v).toFixed(2);
      nd.txt.style.fill = a > 0.45 ? '#fff' : '#111';
      if (o.sub != null) nd.sub.textContent = o.sub;
    }
    setLayer(l, vals, o = {}) { vals.forEach((v, k) => this.setNode(l, k, v, { text: o.texts ? o.texts[k] : undefined, sub: o.subs ? o.subs[k] : undefined })); }
    ring(l, k, color, w = 2.6) { const r = this.nodes[l][k].ring; r.setAttribute('stroke', color || '#111'); r.setAttribute('stroke-width', color ? w : 1.3); }
    clearRings() { this.nodes.forEach((L) => L.forEach((n) => { n.ring.setAttribute('stroke', '#111'); n.ring.setAttribute('stroke-width', 1.3); })); }
    sub(l, k, t, color) { const s = this.nodes[l][k].sub; s.textContent = t || ''; s.style.fill = color || '#6b6b68'; }
    setWeights(W, o = {}) { // W[l][j][i]: weight from neuron i of layer l to neuron j of layer l + 1
      this.w = W; let mx = o.max || 1e-9; if (!o.max) W.forEach((M) => M.forEach((r) => r.forEach((w) => { mx = Math.max(mx, Math.abs(w)); })));
      this.wmax = mx;
      W.forEach((M, l) => M.forEach((row, j) => row.forEach((w, i) => { const e = this.edges[l][j][i]; e.setAttribute('stroke', w >= 0 ? POS : NEG); e.setAttribute('stroke-opacity', 0.35 + 0.45 * Math.min(1, Math.abs(w) / mx)); e.setAttribute('stroke-width', 0.8 + 5 * Math.min(1, Math.abs(w) / mx)); })));
    }
    edgeLabels(l, fn) { this.edgeLab[l].forEach((row, j) => row.forEach((t, i) => { const r = fn(i, j); t.textContent = r == null ? '' : r.text != null ? r.text : r; t.style.fill = (r && r.color) || '#3d3d3b'; })); }
    clearEdgeLabels() { this.edgeLab.forEach((L) => L.forEach((r) => r.forEach((t) => { t.textContent = ''; }))); }
    dim(on, keep) { // fade every edge except those for which keep(l, i, j) is true
      this.edges.forEach((L, l) => L.forEach((row, j) => row.forEach((e, i) => { e.style.opacity = on && !(keep && keep(l, i, j)) ? 0.12 : 1; })));
    }
    /* --- animation */
    async pulse(l, dir, o = {}) { // dots travel along every edge between layer l and l + 1; dir: 'fwd' or 'back'
      const R = this.R, dots = [], color = o.color || (dir === 'back' ? BW : FW), ms = o.ms || 900;
      this.pos[l + 1].forEach((q, j) => this.pos[l].forEach((p, i) => {
        if (o.only && !o.only(i, j)) return;
        const s = o.size ? o.size(i, j) : 1; if (s <= 0.02) return;
        const from = dir === 'back' ? [q[0] - R, q[1]] : [p[0] + R, p[1]], to = dir === 'back' ? [p[0] + R, p[1]] : [q[0] - R, q[1]];
        const c = mk('circle', { r: 2.2 + 4.2 * Math.min(1, s), fill: typeof color === 'function' ? color(i, j) : color, filter: `url(#${this.glowId})` }, this.gPulse);
        dots.push([c, from, to, (i + j) * 0.03]);
      }));
      try {
        await this.clock.tween(ms, (t, raw) => dots.forEach(([c, a, b, d]) => { const u = Math.max(0, Math.min(1, (raw - d) / (1 - d))), e = ease(u); c.setAttribute('cx', a[0] + (b[0] - a[0]) * e); c.setAttribute('cy', a[1] + (b[1] - a[1]) * e); c.setAttribute('opacity', u >= 1 ? 0 : 1); }));
      } finally { dots.forEach(([c]) => c.remove()); }
    }
    async fillLayer(l, vals, o = {}) { // neurons fill up from their old value to the new one, numbers count
      const old = this.nodes[l].map((n) => n.v || 0);
      await this.clock.tween(o.ms || 450, (t) => vals.forEach((v, k) => this.setNode(l, k, old[k] + (v - old[k]) * t, { text: o.texts ? (t > 0.95 ? o.texts[k] : '') : undefined })));
      vals.forEach((v, k) => this.setNode(l, k, v, { text: o.texts ? o.texts[k] : undefined }));
    }
    async morphWeights(W2, o = {}) { // weights change smoothly to new values (after a gradient step)
      const W1 = this.w.map((M) => M.map((r) => r.slice()));
      await this.clock.tween(o.ms || 600, (t) => this.setWeights(W1.map((M, l) => M.map((r, j) => r.map((w, i) => w + (W2[l][j][i] - w) * t))), { max: this.wmax }));
      this.setWeights(W2, { max: this.wmax });
    }
    async flash(l, k, color, ms = 500) { const r = this.nodes[l][k].ring; await this.clock.tween(ms, (t) => { r.setAttribute('stroke', color); r.setAttribute('stroke-width', 1.3 + 4 * Math.sin(t * Math.PI)); }); r.setAttribute('stroke-width', 1.3); r.setAttribute('stroke', '#111'); }
  }
  UL.NetAnim = NetAnim;

  /* ------------------------------------------------------------ narrated steps with controls */
  class Player {
    // steps: [{title, text: () => html, run: async () => {}, show: () => {}}]; show() draws the finished state instantly
    constructor(o) {
      this.o = o; this.steps = o.steps; this.k = -1; this.busy = false; this.auto = false; this.clock = o.clock;
      const c = typeof o.controls === 'string' ? document.getElementById(o.controls) : o.controls;
      c.insertAdjacentHTML('afterbegin', `<button class="btn" data-a="play">▶ Play</button><button class="btn ghost small" data-a="step">Next step</button><button class="btn ghost small" data-a="back">Back</button><button class="btn ghost small" data-a="reset">Restart</button><label class="ctl">speed <input type="range" min="0.25" max="3" step="0.25" value="1" data-a="speed"><output>1×</output></label>`);
      this.bPlay = c.querySelector('[data-a=play]');
      this.bPlay.addEventListener('click', () => (this.auto ? this.pause() : this.play()));
      c.querySelector('[data-a=step]').addEventListener('click', () => { this.pause(); this.next(); });
      c.querySelector('[data-a=back]').addEventListener('click', () => { this.pause(); this.goto(this.k - 1); });
      c.querySelector('[data-a=reset]').addEventListener('click', () => { this.pause(); if (o.onReset) o.onReset(); this.goto(-1); });
      const sp = c.querySelector('[data-a=speed]'); sp.addEventListener('input', () => { this.clock.speed = +sp.value; sp.nextElementSibling.textContent = sp.value + '×'; });
      this.narr = typeof o.narration === 'string' ? document.getElementById(o.narration) : o.narration;
      this.bar = o.progress ? (typeof o.progress === 'string' ? document.getElementById(o.progress) : o.progress) : null;
      if (this.bar) { this.bar.innerHTML = this.steps.map((s, i) => `<span data-i="${i}" title="${s.title}">${i + 1}</span>`).join(''); this.bar.querySelectorAll('span').forEach((s) => s.addEventListener('click', () => { this.pause(); this.goto(+s.dataset.i); })); }
      this.goto(-1);
    }
    narrate() {
      const s = this.steps[this.k];
      this.narr.innerHTML = this.k < 0 ? (this.o.intro || 'Press <b>▶ Play</b> to watch, or <b>Next step</b> to go one step at a time.') : `<div class="nhead"><span class="nstep">${this.k + 1} / ${this.steps.length}</span> ${s.title}</div><div class="nbody">${typeof s.text === 'function' ? s.text() : s.text}</div>`;
      if (this.bar) this.bar.querySelectorAll('span').forEach((e, i) => { e.className = i < this.k ? 'done' : i === this.k ? 'on' : ''; });
    }
    goto(k) { // jump without animation
      this.clock.cancel(); this.busy = false; k = Math.max(-1, Math.min(this.steps.length - 1, k));
      if (this.o.base) this.o.base(); for (let i = 0; i <= k; i++) if (this.steps[i].show) this.steps[i].show();
      this.k = k; this.narrate();
    }
    async next() {
      if (this.busy) return; if (this.k >= this.steps.length - 1) { if (this.o.loop) { if (this.o.onLoop) this.o.onLoop(); this.goto(-1); } else { this.pause(); return; } }
      this.k++; this.narrate(); const s = this.steps[this.k]; this.busy = true;
      try { if (s.run) await s.run(); if (s.show) s.show(); } catch (e) { if (e.message !== 'cancelled') throw e; return; } finally { this.busy = false; }
      if (this.auto) { try { await this.clock.wait(this.o.pause || 700); } catch (e) { return; } if (this.auto) this.next(); }
    }
    play() { this.auto = true; this.bPlay.textContent = '❚❚ Pause'; this.clock.paused = false; if (!this.busy) this.next(); }
    pause() { this.auto = false; this.bPlay.textContent = '▶ Play'; }
  }
  UL.Player = Player;
  UL.ANIM = { FW, BW, POS, NEG };
})();
