/*!
 * Neural Network Lab — shared library
 * Canvas plotting, page UI and small neural networks that train in the browser.
 * Everything runs in the browser; no dependencies.
 * Author: Eng. Saeed Serwan Abdulsattar
 */
(function () {
  'use strict';
  const UL = {};
  /* ================================================================ random */
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    const r = function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    r.normal = function () {
      let u = 0, v = 0;
      while (u === 0) u = r();
      while (v === 0) v = r();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    r.int = (n) => Math.floor(r() * n);
    return r;
  }
  UL.rng = rng;

  /* ================================================================ colours */
  UL.COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
  UL.SHAPES = ['circle', 'square', 'triangle', 'diamond', 'cross', 'star', 'tri-down', 'plus'];
  UL.INK = '#111111'; UL.MUTED = '#6b6b68'; UL.FAINT = '#a3a3a0'; UL.LINE = '#e7e7e4'; UL.GREY = '#c9c9c4';
  UL.color = (k) => (k < 0 ? UL.GREY : UL.COLORS[k % UL.COLORS.length]);
  UL.shape = (k) => (k < 0 ? 'circle' : UL.SHAPES[k % UL.SHAPES.length]);
  UL.rgba = function (hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };
  UL.mix = function (weights) { // blend palette colours by weights (soft clustering)
    let r = 0, g = 0, b = 0, s = 0;
    weights.forEach((w, k) => {
      const n = parseInt(UL.color(k).slice(1), 16);
      r += w * ((n >> 16) & 255); g += w * ((n >> 8) & 255); b += w * (n & 255); s += w;
    });
    s = s || 1;
    return `rgb(${Math.round(r / s)},${Math.round(g / s)},${Math.round(b / s)})`;
  };


  /* ================================================================ plotting */
  class Plot {
    constructor(el, o = {}) {
      this.el = typeof el === 'string' ? document.querySelector(el) : el;
      this.W = o.w || 600; this.H = o.h || 420;
      this.pad = Object.assign({ l: 14, r: 14, t: 14, b: 14 }, o.pad || {});
      this.cv = document.createElement('canvas');
      this.dpr = Math.min(2, Math.max(1.5, window.devicePixelRatio || 1));
      this.cv.width = Math.round(this.W * this.dpr); this.cv.height = Math.round(this.H * this.dpr);
      this.cv.style.aspectRatio = `${this.W} / ${this.H}`;
      this.cv.setAttribute('role', 'img');
      if (o.label) this.cv.setAttribute('aria-label', o.label);
      this.el.appendChild(this.cv);
      this.ctx = this.cv.getContext('2d');
      this.ctx.scale(this.dpr, this.dpr);
      this.dom = [0, 1, 0, 1];
    }
    domain(x0, x1, y0, y1, equal) {
      if (equal) { // same scale on both axes so circles look round
        const pw = this.W - this.pad.l - this.pad.r, ph = this.H - this.pad.t - this.pad.b;
        const sx = pw / (x1 - x0), sy = ph / (y1 - y0), s = Math.min(sx, sy);
        const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        x0 = cx - pw / s / 2; x1 = cx + pw / s / 2; y0 = cy - ph / s / 2; y1 = cy + ph / s / 2;
      }
      this.dom = [x0, x1, y0, y1]; return this;
    }
    fit(pts, margin = 0.08, equal = true, fx = (p) => p.x, fy = (p) => p.y) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      pts.forEach((p) => { const x = fx(p), y = fy(p); if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; });
      if (x1 - x0 < 1e-9) { x0 -= 1; x1 += 1; } if (y1 - y0 < 1e-9) { y0 -= 1; y1 += 1; }
      const mx = (x1 - x0) * margin, my = (y1 - y0) * margin;
      return this.domain(x0 - mx, x1 + mx, y0 - my, y1 + my, equal);
    }
    sx(x) { const [a, b] = this.dom; return this.pad.l + ((x - a) / (b - a)) * (this.W - this.pad.l - this.pad.r); }
    sy(y) { const [, , c, d] = this.dom; return this.H - this.pad.b - ((y - c) / (d - c)) * (this.H - this.pad.t - this.pad.b); }
    ix(px) { const [a, b] = this.dom; return a + ((px - this.pad.l) / (this.W - this.pad.l - this.pad.r)) * (b - a); }
    iy(py) { const [, , c, d] = this.dom; return c + ((this.H - this.pad.b - py) / (this.H - this.pad.t - this.pad.b)) * (d - c); }
    clear() { this.ctx.clearRect(0, 0, this.W, this.H); return this; }
    clipped(fn) { const c = this.ctx; c.save(); c.beginPath(); c.rect(this.pad.l, this.pad.t, this.W - this.pad.l - this.pad.r, this.H - this.pad.t - this.pad.b); c.clip(); fn(); c.restore(); }
    frame() {
      const c = this.ctx; c.strokeStyle = UL.LINE; c.lineWidth = 1;
      c.strokeRect(this.pad.l + 0.5, this.pad.t + 0.5, this.W - this.pad.l - this.pad.r - 1, this.H - this.pad.t - this.pad.b - 1);
      return this;
    }
    axes(o = {}) { // light grid + tick labels
      const c = this.ctx, [x0, x1, y0, y1] = this.dom;
      c.font = '10.5px "IBM Plex Mono", monospace'; c.fillStyle = UL.MUTED; c.lineWidth = 1;
      const ticks = (a, b, n) => { const step = niceStep((b - a) / n); const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10)); return out; };
      if (o.x !== false) ticks(x0, x1, o.nx || 6).forEach((v) => {
        const X = Math.round(this.sx(v)) + 0.5; c.strokeStyle = '#f0f0ed'; c.beginPath(); c.moveTo(X, this.pad.t); c.lineTo(X, this.H - this.pad.b); c.stroke();
        c.textAlign = 'center'; c.textBaseline = 'top'; c.fillText(o.fx ? o.fx(v) : fmtTick(v), X, this.H - this.pad.b + 6);
      });
      if (o.y !== false) ticks(y0, y1, o.ny || 5).forEach((v) => {
        const Y = Math.round(this.sy(v)) + 0.5; c.strokeStyle = '#f0f0ed'; c.beginPath(); c.moveTo(this.pad.l, Y); c.lineTo(this.W - this.pad.r, Y); c.stroke();
        c.textAlign = 'right'; c.textBaseline = 'middle'; c.fillText(o.fy ? o.fy(v) : fmtTick(v), this.pad.l - 6, Y);
      });
      c.strokeStyle = UL.GREY; c.beginPath(); c.moveTo(this.pad.l, this.H - this.pad.b + 0.5); c.lineTo(this.W - this.pad.r, this.H - this.pad.b + 0.5); c.stroke();
      if (o.xlabel) { c.fillStyle = UL.INK; c.font = '11.5px "IBM Plex Sans", sans-serif'; c.textAlign = 'center'; c.textBaseline = 'bottom'; c.fillText(o.xlabel, (this.pad.l + this.W - this.pad.r) / 2, this.H - 2); }
      if (o.ylabel) { c.save(); c.fillStyle = UL.INK; c.font = '11.5px "IBM Plex Sans", sans-serif'; c.translate(10, (this.pad.t + this.H - this.pad.b) / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(o.ylabel, 0, 0); c.restore(); }
      return this;
    }
    marker(x, y, o = {}) { drawMarker(this.ctx, this.sx(x), this.sy(y), o); }
    markerPx(X, Y, o = {}) { drawMarker(this.ctx, X, Y, o); }
    line(x0, y0, x1, y1, o = {}) {
      const c = this.ctx; c.save(); c.strokeStyle = o.color || UL.INK; c.lineWidth = o.w || 1.5; c.globalAlpha = o.alpha == null ? 1 : o.alpha;
      if (o.dash) c.setLineDash(o.dash);
      c.beginPath(); c.moveTo(this.sx(x0), this.sy(y0)); c.lineTo(this.sx(x1), this.sy(y1)); c.stroke(); c.restore();
    }
    poly(pts, o = {}) {
      const c = this.ctx; if (!pts.length) return; c.save(); c.strokeStyle = o.color || UL.INK; c.lineWidth = o.w || 2; c.lineJoin = 'round'; c.lineCap = 'round';
      if (o.dash) c.setLineDash(o.dash); c.globalAlpha = o.alpha == null ? 1 : o.alpha;
      c.beginPath(); pts.forEach((p, i) => (i ? c.lineTo(this.sx(p[0]), this.sy(p[1])) : c.moveTo(this.sx(p[0]), this.sy(p[1]))));
      if (o.fill) { c.fillStyle = o.fill; c.fill(); } if (o.w !== 0) c.stroke(); c.restore();
    }
    circle(x, y, R, o = {}) { // R in data units
      const c = this.ctx, rx = Math.abs(this.sx(x + R) - this.sx(x)); c.save(); c.beginPath(); c.arc(this.sx(x), this.sy(y), rx, 0, 2 * Math.PI);
      if (o.fill) { c.fillStyle = o.fill; c.fill(); } if (o.color) { c.strokeStyle = o.color; c.lineWidth = o.w || 1; if (o.dash) c.setLineDash(o.dash); c.stroke(); } c.restore();
    }
    ellipse(x, y, rx, ry, ang, o = {}) {
      const c = this.ctx, kx = Math.abs(this.sx(1) - this.sx(0)), ky = Math.abs(this.sy(1) - this.sy(0));
      c.save(); c.translate(this.sx(x), this.sy(y)); c.scale(kx, -ky); c.rotate(ang);
      c.beginPath(); c.ellipse(0, 0, rx, ry, 0, 0, 2 * Math.PI); c.restore();
      c.save(); if (o.fill) { c.fillStyle = o.fill; c.fill(); } c.strokeStyle = o.color || UL.INK; c.lineWidth = o.w || 1.5; if (o.dash) c.setLineDash(o.dash); c.stroke(); c.restore();
    }
    arrow(x0, y0, x1, y1, o = {}) {
      const c = this.ctx, X0 = this.sx(x0), Y0 = this.sy(y0), X1 = this.sx(x1), Y1 = this.sy(y1), a = Math.atan2(Y1 - Y0, X1 - X0), h = o.head || 9;
      c.save(); c.strokeStyle = c.fillStyle = o.color || UL.INK; c.lineWidth = o.w || 2.2; c.lineCap = 'round';
      c.beginPath(); c.moveTo(X0, Y0); c.lineTo(X1 - Math.cos(a) * h * 0.6, Y1 - Math.sin(a) * h * 0.6); c.stroke();
      c.beginPath(); c.moveTo(X1, Y1); c.lineTo(X1 - h * Math.cos(a - 0.4), Y1 - h * Math.sin(a - 0.4)); c.lineTo(X1 - h * Math.cos(a + 0.4), Y1 - h * Math.sin(a + 0.4)); c.closePath(); c.fill();
      if (o.label) { c.font = '500 12px "IBM Plex Sans", sans-serif'; c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText(o.label, X1 + 7 * Math.cos(a) + 3, Y1 + 9 * Math.sin(a)); }
      c.restore();
    }
    text(s, x, y, o = {}) { this.textPx(s, this.sx(x), this.sy(y), o); }
    textPx(s, X, Y, o = {}) {
      const c = this.ctx; c.save(); c.font = o.font || '11px "IBM Plex Mono", monospace'; c.fillStyle = o.color || UL.INK;
      c.textAlign = o.align || 'center'; c.textBaseline = o.base || 'middle';
      if (o.halo) { c.lineWidth = 3.5; c.strokeStyle = '#fff'; c.lineJoin = 'round'; c.strokeText(s, X, Y); }
      c.fillText(s, X, Y); c.restore();
    }
    regions(fn, step = 6, alpha = 0.1) { // colour every patch by fn(x, y) -> cluster index
      const c = this.ctx; step = Math.round(step);
      for (let X = this.pad.l; X < this.W - this.pad.r; X += step) for (let Y = this.pad.t; Y < this.H - this.pad.b; Y += step) {
        const k = fn(this.ix(X + step / 2), this.iy(Y + step / 2));
        if (k == null || k < 0) continue;
        c.fillStyle = UL.rgba(UL.color(k), alpha); c.fillRect(X, Y, Math.min(step, this.W - this.pad.r - X), Math.min(step, this.H - this.pad.b - Y));
      }
    }
    pointer(e) { const r = this.cv.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * this.W, ((e.clientY - r.top) / r.height) * this.H]; }
    nearest(pts, e, maxPx = 12, fx = (p) => p.x, fy = (p) => p.y) {
      const [X, Y] = this.pointer(e); let best = -1, bd = maxPx * maxPx;
      pts.forEach((p, i) => { const dx = this.sx(fx(p)) - X, dy = this.sy(fy(p)) - Y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } });
      return best;
    }
  }
  function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }
  function fmtTick(v) { return Math.abs(v) >= 1000 ? v.toLocaleString('en-US') : String(+v.toFixed(3)); }
  function drawMarker(c, X, Y, o) {
    const r = o.r || 4.2, shape = o.shape || 'circle';
    c.save(); c.globalAlpha = o.alpha == null ? 1 : o.alpha;
    c.beginPath();
    if (shape === 'circle') c.arc(X, Y, r, 0, 2 * Math.PI);
    else if (shape === 'square') c.rect(X - r * 0.88, Y - r * 0.88, r * 1.76, r * 1.76);
    else if (shape === 'triangle') { c.moveTo(X, Y - r * 1.15); c.lineTo(X + r * 1.05, Y + r * 0.75); c.lineTo(X - r * 1.05, Y + r * 0.75); c.closePath(); }
    else if (shape === 'tri-down') { c.moveTo(X, Y + r * 1.15); c.lineTo(X + r * 1.05, Y - r * 0.75); c.lineTo(X - r * 1.05, Y - r * 0.75); c.closePath(); }
    else if (shape === 'diamond') { c.moveTo(X, Y - r * 1.25); c.lineTo(X + r * 1.05, Y); c.lineTo(X, Y + r * 1.25); c.lineTo(X - r * 1.05, Y); c.closePath(); }
    else if (shape === 'star') { for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + (i * Math.PI) / 5, R = i % 2 ? r * 0.55 : r * 1.3; c.lineTo(X + R * Math.cos(a), Y + R * Math.sin(a)); } c.closePath(); }
    else if (shape === 'cross' || shape === 'plus' || shape === 'x') {
      const k = r * 1.05; c.lineWidth = o.lw || 2.4; c.strokeStyle = o.fill || UL.INK; c.lineCap = 'round';
      if (shape === 'plus') { c.moveTo(X - k, Y); c.lineTo(X + k, Y); c.moveTo(X, Y - k); c.lineTo(X, Y + k); }
      else { c.moveTo(X - k, Y - k); c.lineTo(X + k, Y + k); c.moveTo(X + k, Y - k); c.lineTo(X - k, Y + k); }
      if (o.ring !== false) { c.save(); c.lineWidth = (o.lw || 2.4) + 2.5; c.strokeStyle = '#fff'; c.stroke(); c.restore(); }
      c.stroke(); c.restore(); return;
    }
    if (o.fill) { c.fillStyle = o.fill; c.fill(); }
    if (o.ring !== false) { c.lineWidth = o.ringW || 1.2; c.strokeStyle = o.ringColor || '#fff'; c.stroke(); }
    if (o.stroke) { c.lineWidth = o.lw || 1.5; c.strokeStyle = o.stroke; c.stroke(); }
    c.restore();
  }
  UL.Plot = Plot;
  UL.drawMarker = drawMarker;

  /* ================================================================ UI */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  UL.$ = $; UL.$$ = $$;
  UL.range = function (id, cb, fmt = (v) => v) {
    const el = typeof id === 'string' ? document.getElementById(id) : id, out = el.parentElement.querySelector('output, .v');
    const upd = () => { const v = +el.value; if (out) out.textContent = fmt(v); cb && cb(v); };
    el.addEventListener('input', upd); if (out) out.textContent = fmt(+el.value);
    return { el, get v() { return +el.value; }, set(v) { el.value = v; if (out) out.textContent = fmt(+el.value); } };
  };
  UL.seg = function (id, cb) {
    const el = typeof id === 'string' ? document.getElementById(id) : id, btns = $$('button', el);
    const api = {
      value: (btns.find((b) => b.classList.contains('on')) || btns[0]).dataset.v,
      set(v, fire = true) { api.value = v; btns.forEach((b) => b.classList.toggle('on', b.dataset.v === v)); if (fire && cb) cb(v); },
    };
    btns.forEach((b) => b.addEventListener('click', () => api.set(b.dataset.v)));
    btns.forEach((b) => b.classList.toggle('on', b.dataset.v === api.value));
    return api;
  };
  UL.tip = {
    el: null,
    show(e, html) {
      if (!this.el) { this.el = document.createElement('div'); this.el.className = 'tooltip'; document.body.appendChild(this.el); }
      this.el.innerHTML = html; this.el.hidden = false;
      const x = e.clientX, y = e.clientY, w = this.el.offsetWidth;
      this.el.style.left = (x + w + 30 > window.innerWidth ? x - w - 24 : x) + 'px'; this.el.style.top = y + 'px';
    },
    hide() { if (this.el) this.el.hidden = true; },
  };
  UL.sw = (k) => `<span class="sw" style="background:${UL.color(k)}"></span>`;
  UL.legend = function (el, items) {
    el.innerHTML = items.map((it) => `<span><i class="${it.cls || ''}" style="background:${it.color}"></i>${it.label}</span>`).join('');
  };
  UL.copyText = async function (t) {
    try { await navigator.clipboard.writeText(t); } catch (e) {
      const ta = document.createElement('textarea'); ta.value = t; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    }
  };
  UL.tocSpy = function () {
    const links = $$('.toc a'); if (!links.length) return;
    const secs = links.map((a) => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
    const on = () => {
      let cur = secs[0];
      secs.forEach((s) => { if (s.getBoundingClientRect().top < window.innerHeight * 0.3) cur = s; });
      links.forEach((a) => a.classList.toggle('active', cur && a.getAttribute('href') === '#' + cur.id));
    };
    window.addEventListener('scroll', on, { passive: true }); on();
  };
  UL.math = function () {
    if (window.renderMathInElement) renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }, { left: '\\[', right: '\\]', display: true }],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option', 'canvas'],
      throwOnError: false, strict: 'ignore', trust: (c) => c.command === '\\htmlData',
      preProcess: (m) => (UL.texTips ? UL.texTips(m) : m),
    });
    if (UL.tips) UL.tips();
  };
  UL.tex = function (el, src, display = false) {
    if (window.katex) katex.render(src, el, { throwOnError: false, displayMode: display }); else el.textContent = src;
  };
  UL.quiz = function (rootId, qs) {
    const root = document.getElementById(rootId); if (!root) return;
    const score = document.getElementById(rootId + '-score');
    let answered = 0, right = 0;
    root.innerHTML = qs.map((q, i) => `<div class="q" data-i="${i}"><p class="q-text"><span>${q.q}</span></p><div class="q-opts">${q.o.map((o, j) => `<button data-j="${j}">${o}</button>`).join('')}</div><p class="q-why">${q.why}</p></div>`).join('');
    $$('.q', root).forEach((qd) => {
      const q = qs[+qd.dataset.i];
      $$('button', qd).forEach((b) => b.addEventListener('click', () => {
        const j = +b.dataset.j; answered++; if (j === q.a) right++;
        $$('button', qd).forEach((x) => { x.disabled = true; if (+x.dataset.j === q.a) x.classList.add('right'); });
        if (j !== q.a) b.classList.add('wrong');
        qd.classList.add('done');
        if (score) score.textContent = `Score: ${right} / ${answered} answered (of ${qs.length})`;
      }));
    });
    UL.math();
  };
  UL.onVisible = function (el, fn) { // start animations only when on screen
    if (!('IntersectionObserver' in window)) { fn(true); return; }
    new IntersectionObserver((es) => es.forEach((e) => fn(e.isIntersecting)), { threshold: 0.05 }).observe(el);
  };
  UL.start = function () { // safe to call from several scripts on one page
    clearTimeout(UL._st); UL._st = setTimeout(() => { if (UL._started) return; UL._started = true; UL.tocSpy(); UL.math(); }, 0);
  };
  UL.fmt = (v, d = 2) => (Math.abs(v) < 0.5 * 10 ** -d ? '0' : v.toFixed(d)).replace('-', '−');

  /* ---------------------------------------------------- simple line charts */
  UL.chart = function (plot, series, o = {}) {
    const xs = series.flatMap((s) => s.data.map((_, i) => i + (o.x0 == null ? 1 : o.x0)));
    const ys = series.flatMap((s) => s.data).filter((v) => isFinite(v));
    const xmax = o.xmax || Math.max(2, ...xs), ymin = o.ymin != null ? o.ymin : Math.min(...ys, o.hline != null ? o.hline : Infinity), ymax = o.ymax != null ? o.ymax : Math.max(...ys);
    const T = (v) => (o.log ? Math.log10(Math.max(1, v)) : v);
    plot.clear(); plot.domain(o.xmin || 0, xmax, T(ymin), T(ymax) + (T(ymax) - T(ymin) || 1) * 0.04);
    plot.axes({ xlabel: o.xlabel, ylabel: o.ylabel, nx: o.nx || 6, ny: o.ny || 5, fy: o.log ? (v) => (Number.isInteger(v) ? String(10 ** v) : '') : o.fy });
    if (o.band) { const c = plot.ctx; c.fillStyle = o.band.color || 'rgba(237,161,0,.15)'; c.fillRect(plot.sx(o.band.x0), plot.pad.t, plot.sx(o.band.x1) - plot.sx(o.band.x0), plot.H - plot.pad.t - plot.pad.b); if (o.band.label) plot.textPx(o.band.label, (plot.sx(o.band.x0) + plot.sx(o.band.x1)) / 2, plot.pad.t + 12, { color: '#a0522d', font: '10.5px "IBM Plex Sans", sans-serif' }); }
    if (o.hline != null) { plot.line(o.xmin || 0, T(o.hline), xmax, T(o.hline), { color: UL.FAINT, dash: [4, 4], w: 1 }); if (o.hlabel) plot.textPx(o.hlabel, plot.W - plot.pad.r - 4, plot.sy(T(o.hline)) - 8, { align: 'right', color: UL.MUTED, font: '10px "IBM Plex Mono", monospace' }); }
    plot.clipped(() => series.forEach((s) => {
      if (s.dots) s.data.forEach((v, i) => plot.marker(i + (o.x0 == null ? 1 : o.x0), T(v), { r: s.r || 2, fill: s.color, ring: false, alpha: s.alpha }));
      else plot.poly(s.data.map((v, i) => [i + (o.x0 == null ? 1 : o.x0), T(v)]), { color: s.color, w: s.w || 2, alpha: s.alpha, dash: s.dash });
    }));
    return plot;
  };
  UL.bars = function (plot, vals, o = {}) { // simple bar chart, vals: [{v, color, label}]
    const n = vals.length, ymin = o.ymin != null ? o.ymin : Math.min(0, ...vals.map((x) => x.v)), ymax = o.ymax != null ? o.ymax : Math.max(0, ...vals.map((x) => x.v));
    plot.clear(); plot.domain(-0.6, n - 0.4, ymin, ymax + (ymax - ymin) * 0.08); plot.axes({ x: false, ylabel: o.ylabel, ny: o.ny || 4, fy: o.fy });
    const c = plot.ctx, bw = (plot.sx(1) - plot.sx(0)) * (o.width || 0.62);
    vals.forEach((b, i) => {
      const X = plot.sx(i), Y0 = plot.sy(0), Y1 = plot.sy(b.v); c.fillStyle = b.color || UL.COLORS[0]; c.fillRect(X - bw / 2, Math.min(Y0, Y1), bw, Math.abs(Y1 - Y0));
      if (b.label != null) plot.textPx(b.label, X, plot.H - plot.pad.b + 10, { color: UL.MUTED, font: '11px "IBM Plex Sans", sans-serif', base: 'top' });
      if (o.values !== false) plot.textPx(o.fv ? o.fv(b.v) : UL.fmt(b.v), X, b.v >= 0 ? Y1 - 9 : Y1 + 10, { font: '500 11px "IBM Plex Mono", monospace' });
    });
    if (ymin < 0) plot.line(-0.6, 0, n - 0.4, 0, { color: UL.INK, w: 1 });
    return plot;
  };

  /* ---------------------------------------------------- small helpers for labs */
  UL.loop = function (fn) { // requestAnimationFrame loop with start/stop; fn(dt) returns false to stop
    let id = null, last = 0;
    const tick = (t) => { const dt = last ? (t - last) / 1000 : 0; last = t; if (fn(dt) === false) { id = null; last = 0; return; } id = requestAnimationFrame(tick); };
    return { start() { if (id == null) { last = 0; id = requestAnimationFrame(tick); } }, stop() { if (id != null) cancelAnimationFrame(id); id = null; last = 0; }, get on() { return id != null; } };
  };
  UL.stats = function (el, items) { // [{k, v, hi}]
    (typeof el === 'string' ? document.getElementById(el) : el).innerHTML = items.map((it) => `<div class="stat${it.hi ? ' hi' : ''}"><span class="k">${it.k}</span><span class="v">${it.v}</span></div>`).join('');
  };

  /* ================================================================ neural networks */
  // Activation functions: f(z), and the slope written with z and a = f(z).
  const ACT = {
    sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a), name: 'sigmoid' },
    relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0), name: 'ReLU' },
    tanh: { f: (z) => Math.tanh(z), d: (z, a) => 1 - a * a, name: 'tanh' },
    linear: { f: (z) => z, d: () => 1, name: 'none (linear)' },
  };
  UL.ACT = ACT;
  UL.sigmoid = ACT.sigmoid.f;

  /* A small fully connected network. Weights W[l] are stored row by row: W[l][j * nIn + i] is the weight
     from neuron i of layer l to neuron j of layer l + 1 (row j of the matrix W in a = σ(W a + b)). */
  class MLP {
    constructor(sizes, o = {}) {
      this.sizes = sizes.slice(); this.act = o.act || 'sigmoid'; this.out = o.out || 'sigmoid'; this.loss = o.loss || 'mse';
      this.L = sizes.length - 1; this.W = []; this.b = []; this.gW = []; this.gb = []; this.vW = []; this.vb = [];
      const r = rng(o.seed || 1);
      for (let l = 0; l < this.L; l++) {
        const nIn = sizes[l], nOut = sizes[l + 1], s = o.scale != null ? o.scale : (this.act === 'relu' ? Math.sqrt(2 / nIn) : 1 / Math.sqrt(nIn));
        const W = new Float64Array(nIn * nOut); for (let k = 0; k < W.length; k++) W[k] = r.normal() * s;
        this.W.push(W); this.b.push(new Float64Array(nOut));
        this.gW.push(new Float64Array(nIn * nOut)); this.gb.push(new Float64Array(nOut));
        this.vW.push(new Float64Array(nIn * nOut)); this.vb.push(new Float64Array(nOut));
      }
      this.a = sizes.map((n) => new Float64Array(n)); this.z = sizes.map((n) => new Float64Array(n)); this.d = sizes.map((n) => new Float64Array(n));
    }
    get nParams() { return this.W.reduce((s, w) => s + w.length, 0) + this.b.reduce((s, b) => s + b.length, 0); }
    actOf(l) { return ACT[l === this.L ? this.out : this.act]; }
    forward(x) { // x: array-like of length sizes[0]; returns the output layer (kept in this.a)
      const a0 = this.a[0]; for (let i = 0; i < a0.length; i++) a0[i] = x[i];
      for (let l = 0; l < this.L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], b = this.b[l], ain = this.a[l], z = this.z[l + 1], a = this.a[l + 1], f = this.actOf(l + 1).f;
        for (let j = 0; j < nOut; j++) { let s = b[j]; const o = j * nIn; for (let i = 0; i < nIn; i++) s += W[o + i] * ain[i]; z[j] = s; a[j] = f(s); }
      }
      return this.a[this.L];
    }
    cost(y) { // y: wanted output (array). mse: Σ (a − y)²; bce: cross-entropy for one sigmoid output
      const a = this.a[this.L]; let c = 0;
      if (this.loss === 'bce') { for (let j = 0; j < a.length; j++) { const p = Math.min(1 - 1e-7, Math.max(1e-7, a[j])); c -= y[j] * Math.log(p) + (1 - y[j]) * Math.log(1 - p); } }
      else for (let j = 0; j < a.length; j++) c += (a[j] - y[j]) ** 2;
      return c;
    }
    zeroGrad() { this.gW.forEach((g) => g.fill(0)); this.gb.forEach((g) => g.fill(0)); }
    backward(y, wgt = 1) { // adds dCost/dparameter for the last forward pass to gW, gb (backpropagation)
      const L = this.L, aL = this.a[L], dL = this.d[L], oa = this.actOf(L);
      for (let j = 0; j < aL.length; j++) dL[j] = (this.loss === 'bce' && this.out === 'sigmoid') ? (aL[j] - y[j]) : 2 * (aL[j] - y[j]) * oa.d(this.z[L][j], aL[j]);
      for (let l = L - 1; l >= 0; l--) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], gW = this.gW[l], gb = this.gb[l], ain = this.a[l], dout = this.d[l + 1];
        for (let j = 0; j < nOut; j++) { const dj = dout[j] * wgt; if (dj === 0) continue; gb[j] += dj; const o = j * nIn; for (let i = 0; i < nIn; i++) gW[o + i] += dj * ain[i]; }
        if (l > 0) {
          const din = this.d[l], fa = this.actOf(l);
          for (let i = 0; i < nIn; i++) { let s = 0; for (let j = 0; j < nOut; j++) s += W[j * nIn + i] * dout[j]; din[i] = s * fa.d(this.z[l][i], ain[i]); }
        }
      }
    }
    step(lr, n = 1, o = {}) { // gradient descent step with the average gradient of n examples (optional momentum, L2)
      const mom = o.momentum || 0, l2 = o.l2 || 0;
      for (let l = 0; l < this.L; l++) {
        const W = this.W[l], gW = this.gW[l], vW = this.vW[l], b = this.b[l], gb = this.gb[l], vb = this.vb[l];
        for (let k = 0; k < W.length; k++) { const g = gW[k] / n + l2 * W[k]; vW[k] = mom * vW[k] - lr * g; W[k] += vW[k]; }
        for (let k = 0; k < b.length; k++) { vb[k] = mom * vb[k] - lr * gb[k] / n; b[k] += vb[k]; }
      }
      this.zeroGrad();
    }
    predict(x) { const o = this.forward(x); let k = 0; for (let j = 1; j < o.length; j++) if (o[j] > o[k]) k = j; return k; }
  }
  UL.MLP = MLP;

  UL.pretrained = function () { // the lecture's 784-16-16-10 network, trained offline (see tools/)
    const P = window.NN_PRETRAINED; if (!P) return null;
    const bin = atob(P.b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const f = new Float32Array(buf), net = new MLP(P.sizes); let k = 0;
    net.W.forEach((W) => { for (let i = 0; i < W.length; i++) W[i] = f[k++]; });
    net.b.forEach((b) => { for (let i = 0; i < b.length; i++) b[i] = f[k++]; });
    return net;
  };



  UL.pixPlot = function (el, size = 280) { const P = new Plot(el, { w: size, h: size, pad: { l: 0, r: 0, t: 0, b: 0 } }); P.cv.classList.add('pix'); return P; };
  UL.exercises = function (id, items) { // [{q, a}] → numbered list with "Show answer" buttons
    const el = document.getElementById(id); if (!el) return;
    el.innerHTML = items.map((it, i) => `<li><p style="margin:0">${it.q}</p><button class="btn ghost small" data-i="${i}">Show answer</button><div class="reveal" hidden>${it.a}</div></li>`).join('');
    $$('button', el).forEach((b) => b.addEventListener('click', () => { const r = b.nextElementSibling; r.hidden = !r.hidden; b.textContent = r.hidden ? 'Show answer' : 'Hide answer'; }));
  };
  UL.thumbs = function (el, idxs, onpick, o = {}) { // little digit buttons from the shipped samples
    const S = UL.samples(); el = typeof el === 'string' ? document.getElementById(el) : el; if (!S || !el) return;
    el.innerHTML = ''; const cvs = [];
    idxs.forEach((i, k) => {
      const c = document.createElement('canvas'); c.width = 28; c.height = 28; c.title = 'digit ' + S.label(i); c.setAttribute('role', 'button'); c.tabIndex = 0;
      UL.drawDigit(c.getContext('2d'), S.get(i), 0, 0, 28); el.appendChild(c); cvs.push(c);
      const go = () => { cvs.forEach((x) => x.classList.remove('on')); c.classList.add('on'); onpick(i, k); };
      c.addEventListener('click', go); c.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
    });
    if (o.select != null && cvs[o.select]) cvs[o.select].classList.add('on');
    return { clear() { cvs.forEach((x) => x.classList.remove('on')); } };
  };
  /* A layered network diagram in SVG. layers: [{n, shown, label}]. Returns the svg and node positions. */
  UL.netSVG = function (el, layers, o = {}) {
    el = typeof el === 'string' ? document.getElementById(el) : el;
    const W = o.w || 640, H = o.h || 360, top = 34, bot = 18, R = o.r || 7.5, xs = layers.map((_, i) => 40 + (i * (W - 80)) / (layers.length - 1));
    const pos = layers.map((L, i) => { const m = L.shown || L.n, gap = Math.min(2 * R + 5, (H - top - bot) / m); const y0 = (top + H - bot) / 2 - ((m - 1) * gap) / 2; return Array.from({ length: m }, (_, k) => [xs[i], y0 + k * gap]); });
    let edges = '';
    if (o.edges !== false) for (let l = 0; l < layers.length - 1; l++) pos[l].forEach((p, i) => pos[l + 1].forEach((q, j) => { edges += `<line class="e" data-l="${l}" data-i="${i}" data-j="${j}" x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#d6d6d2" stroke-width="0.5"/>`; }));
    let nodes = '';
    pos.forEach((P, l) => P.forEach((p, k) => { nodes += `<circle class="nd" data-l="${l}" data-k="${k}" cx="${p[0]}" cy="${p[1]}" r="${R}" fill="#fff" stroke="#111" stroke-width="1"/>`; }));
    let labels = '';
    layers.forEach((L, l) => {
      labels += `<text x="${xs[l]}" y="16" text-anchor="middle" style="font:500 11.5px var(--font);fill:#111">${L.label || ''}</text>`;
      if (L.shown && L.shown < L.n) labels += `<text x="${xs[l]}" y="${H - 4}" text-anchor="middle">+ ${L.n - L.shown} more</text>`;
      if (o.digits && l === layers.length - 1) P_digits(l);
    });
    function P_digits(l) { pos[l].forEach((p, k) => { labels += `<text x="${p[0] + R + 6}" y="${p[1] + 4}" style="font:11px var(--mono);fill:#6b6b68">${k}</text>`; }); }
    el.innerHTML = `<svg class="netsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label || 'Network diagram'}">${edges}${nodes}${labels}</svg>`;
    const svg = el.querySelector('svg');
    return { svg, pos, nodes: (l) => $$(`circle.nd[data-l="${l}"]`, svg), edges: () => $$('line.e', svg) };
  };


  UL.outbars = function (el, outs, want = -1, o = {}) { // 10 output neurons as bars; want = the correct digit (orange tick at 1)
    el = typeof el === 'string' ? document.getElementById(el) : el; const top = UL.argmax(outs);
    el.innerHTML = Array.from(outs).map((v, j) => `<div class="r${j === top ? ' top' : ''}${j === want ? ' want' : ''}"><span class="d">${o.labels ? o.labels[j] : j}</span><span class="bar"><i style="width:${Math.max(0, Math.min(1, v)) * 100}%"></i>${want >= 0 ? `<b style="left:calc(${j === want ? 100 : 0}% - ${j === want ? 2 : 0}px)"></b>` : ''}</span><span>${v.toFixed(2)}</span></div>`).join('');
  };

  let sampleCache = null;
  UL.samples = function () { // 80 test digits shipped in assets/js/data/samples.js: 40 threes, then 4 of each digit
    if (sampleCache) return sampleCache;
    const S = window.NN_SAMPLES; if (!S) return null;
    const bin = atob(S.b64), X = new Float32Array(bin.length);
    for (let i = 0; i < bin.length; i++) X[i] = bin.charCodeAt(i) / 255;
    sampleCache = { n: S.n, X, label: (i) => +S.labels[i], get: (i) => X.subarray(i * 784, i * 784 + 784), ofDigit: (d, k = 0) => 40 + d * 4 + k };
    return sampleCache;
  };

  let digitCache = null;
  UL.digits = function () { // → Promise<{X: Float32Array (N × 784, 0…1), y: Uint8Array, nTrain, nTest}>
    if (digitCache) return digitCache;
    const D = window.NN_DIGITS;
    digitCache = new Promise((res, rej) => {
      if (!D) { rej(new Error('digit data missing')); return; }
      const img = new Image();
      img.onload = () => {
        const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
        const c = cv.getContext('2d'); c.drawImage(img, 0, 0); const px = c.getImageData(0, 0, img.width, img.height).data;
        const N = D.train + D.test, X = new Float32Array(N * 784), y = new Uint8Array(N);
        for (let n = 0; n < N; n++) {
          const r0 = Math.floor(n / D.cols) * 28, c0 = (n % D.cols) * 28; y[n] = +D.labels[n];
          for (let r = 0; r < 28; r++) for (let q = 0; q < 28; q++) X[n * 784 + r * 28 + q] = px[((r0 + r) * img.width + c0 + q) * 4] / 255;
        }
        res({ X, y, nTrain: D.train, nTest: D.test, get: (n) => X.subarray(n * 784, n * 784 + 784) });
      };
      img.onerror = rej; img.src = D.png;
    });
    return digitCache;
  };

  UL.drawDigit = function (ctx, px, x, y, size, o = {}) { // a 28 × 28 (or n × n) picture, white strokes on black
    const n = Math.round(Math.sqrt(px.length)), s = size / n;
    for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
      const v = px[r * n + q]; let col;
      if (o.signed) col = UL.div(v, o.scale || 1);
      else { const g = Math.round(255 * Math.max(0, Math.min(1, v))); col = o.invert ? `rgb(${255 - g},${255 - g},${255 - g})` : `rgb(${g},${g},${g})`; }
      ctx.fillStyle = col; ctx.fillRect(x + q * s, y + r * s, Math.ceil(s), Math.ceil(s));
    }
  };
  UL.div = function (v, scale = 1) { // diverging colour: red (−) · white (0) · green (+), as on the lecture slides
    const t = Math.max(-1, Math.min(1, v / scale)), k = Math.abs(t) ** 0.8;
    const to = t >= 0 ? [27, 140, 92] : [214, 64, 69];
    const c = [255, 255, 255].map((w, i) => Math.round(w + (to[i] - w) * k));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  UL.argmax = (a) => { let k = 0; for (let j = 1; j < a.length; j++) if (a[j] > a[k]) k = j; return k; };

  /* A drawing pad that turns strokes into a 28 × 28 MNIST-style picture:
     crop to the drawing, scale it into a 20 × 20 box, and centre its centre of mass in the 28 × 28 frame. */
  UL.DrawPad = class {
    constructor(el, o = {}) {
      this.el = typeof el === 'string' ? document.querySelector(el) : el; this.N = o.size || 280; this.onchange = o.onchange || (() => {});
      this.cv = document.createElement('canvas'); this.cv.width = this.N; this.cv.height = this.N; this.cv.className = 'drawpad';
      this.el.appendChild(this.cv); this.c = this.cv.getContext('2d'); this.clear(false); this.down = false; this.last = null;
      const pos = (e) => { const r = this.cv.getBoundingClientRect(), t = e.touches ? e.touches[0] : e; return [((t.clientX - r.left) / r.width) * this.N, ((t.clientY - r.top) / r.height) * this.N]; };
      const start = (e) => { e.preventDefault(); this.down = true; this.last = pos(e); this.dot(this.last); };
      const move = (e) => { if (!this.down) return; e.preventDefault(); const p = pos(e); this.stroke(this.last, p); this.last = p; this.fire(); };
      const end = () => { if (this.down) { this.down = false; this.fire(true); } };
      this.cv.addEventListener('mousedown', start); window.addEventListener('mousemove', move); window.addEventListener('mouseup', end);
      this.cv.addEventListener('touchstart', start, { passive: false }); this.cv.addEventListener('touchmove', move, { passive: false }); this.cv.addEventListener('touchend', end);
      this.pending = false;
    }
    fire() { if (this.pending) return; this.pending = true; requestAnimationFrame(() => { this.pending = false; this.onchange(this.pixels()); }); }
    clear(fire = true) { this.c.fillStyle = '#000'; this.c.fillRect(0, 0, this.N, this.N); this.empty = true; if (fire) this.onchange(this.pixels()); }
    dot(p) { this.stroke(p, [p[0] + 0.1, p[1] + 0.1]); }
    stroke(a, b) { const c = this.c; c.strokeStyle = '#fff'; c.lineWidth = this.N / 13; c.lineCap = 'round'; c.lineJoin = 'round'; c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); this.empty = false; }
    setPixels(px) { // show a 28 × 28 picture on the pad
      const s = this.N / 28; for (let r = 0; r < 28; r++) for (let q = 0; q < 28; q++) { const g = Math.round(255 * px[r * 28 + q]); this.c.fillStyle = `rgb(${g},${g},${g})`; this.c.fillRect(q * s, r * s, Math.ceil(s), Math.ceil(s)); }
      this.empty = false; this.onchange(this.pixels());
    }
    pixels() {
      const out = new Float32Array(784); if (this.empty) return out;
      const N = this.N, d = this.c.getImageData(0, 0, N, N).data; let x0 = N, x1 = -1, y0 = N, y1 = -1;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (d[(y * N + x) * 4] > 20) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      if (x1 < 0) return out;
      const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h);
      const tmp = document.createElement('canvas'); tmp.width = 28; tmp.height = 28; const t = tmp.getContext('2d');
      t.fillStyle = '#000'; t.fillRect(0, 0, 28, 28); t.imageSmoothingEnabled = true; t.imageSmoothingQuality = 'high';
      t.drawImage(this.cv, x0, y0, w, h, (28 - w * s) / 2, (28 - h * s) / 2, w * s, h * s);
      const p = t.getImageData(0, 0, 28, 28).data; let m = 0, mx = 0, my = 0;
      for (let i = 0; i < 784; i++) { const v = p[i * 4] / 255; out[i] = v; m += v; mx += v * (i % 28); my += v * Math.floor(i / 28); }
      if (m > 0) { // shift the centre of mass to the middle, as in MNIST
        const dx = Math.round(13.5 - mx / m), dy = Math.round(13.5 - my / m), sh = new Float32Array(784);
        for (let r = 0; r < 28; r++) for (let q = 0; q < 28; q++) { const rr = r - dy, qq = q - dx; if (rr >= 0 && rr < 28 && qq >= 0 && qq < 28) sh[r * 28 + q] = out[rr * 28 + qq]; }
        return sh;
      }
      return out;
    }
  };

  window.UL = UL;
})();
