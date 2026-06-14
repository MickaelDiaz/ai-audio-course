/* ============================================================
   Audio AI Atlas — moteur de visualisation & utilitaires
   Vanilla JS, zéro dépendance. Tout est exposé sur window.Atlas.
   ============================================================ */
(function () {
  'use strict';

  /* ---------- Palette — « Studio Mono » : coque noir chaud, accent corail ---------- */
  const palette = {
    bg: '#0c0c0e',      // fond page — noir chaud
    stage: '#0e0e11',   // fond canvas
    panel: '#161619',   // boîtes / nodes
    panel2: '#1f1f23',
    grid: 'rgba(255,255,255,0.07)',
    text: '#f4f2ec',    // blanc cassé chaud
    dim: '#9b988f',
    faint: '#605d57',
    accent: '#ff5c49',  // accent de marque — corail (UI + identité)
    voice: '#ff5c49',   // voix — corail (= marque ; constant dans toute l'app)
    rest: '#fbbf24',    // musique/effets — ambre
    mix: '#a78bfa',     // mix — violet
    blue: '#60a5fa',
    pink: '#f472b6',
    red: '#f87171',
    green: '#34d399',
    orange: '#fb923c',
    yellow: '#facc15',
    teal: '#38bdf8',    // (héritage) repointé vers un cyan franc — plus aucun teal-vert
    violet: '#a78bfa',
  };

  const TAU = Math.PI * 2;

  /* ---------- Maths ---------- */
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  const ease = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3); // easeOutCubic

  /* Pseudo-aléatoire déterministe (stable d'une frame à l'autre) */
  function hash(n) {
    const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  /* Bruit 1D lisse (value noise) */
  function noise1(x) {
    const i = Math.floor(x), f = x - i;
    return lerp(hash(i), hash(i + 1), smoothstep(f));
  }

  /* ---------- FFT (radix-2, in-place) ---------- */
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -TAU / len, wr = Math.cos(ang), wi = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < half; k++) {
          const a = i + k, b = a + half;
          const vr = re[b] * cr - im[b] * ci;
          const vi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - vr; im[b] = im[a] - vi;
          re[a] += vr; im[a] += vi;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  const _hannCache = {};
  function hann(N) {
    if (!_hannCache[N]) {
      const w = new Float32Array(N);
      for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos(TAU * i / N);
      _hannCache[N] = w;
    }
    return _hannCache[N];
  }

  const _scratch = {};
  function scratch(N) {
    if (!_scratch[N]) _scratch[N] = { re: new Float32Array(N), im: new Float32Array(N) };
    return _scratch[N];
  }

  /* Magnitudes du spectre d'un signal réel (longueur N puissance de 2).
     Renvoie Float32Array de N/2 magnitudes normalisées (~0..1 pour un signal ±1). */
  function rfftMag(signal, applyHann = true) {
    const N = signal.length;
    const { re, im } = scratch(N);
    const w = applyHann ? hann(N) : null;
    for (let i = 0; i < N; i++) { re[i] = w ? signal[i] * w[i] : signal[i]; im[i] = 0; }
    fft(re, im);
    const out = new Float32Array(N / 2);
    const norm = 4 / N; // ~compense la fenêtre de Hann
    for (let i = 0; i < N / 2; i++) out[i] = Math.hypot(re[i], im[i]) * norm;
    return out;
  }
  /* Spectre complexe complet : { re, im } (références vers scratch — copier si besoin) */
  function rfft(signal, applyHann = true) {
    const N = signal.length;
    const { re, im } = scratch(N);
    const w = applyHann ? hann(N) : null;
    for (let i = 0; i < N; i++) { re[i] = w ? signal[i] * w[i] : signal[i]; im[i] = 0; }
    fft(re, im);
    return { re, im };
  }

  /* ---------- Colormaps (LUT 256) ---------- */
  function buildLUT(anchors) {
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255 * (anchors.length - 1);
      const k = Math.min(anchors.length - 2, Math.floor(t));
      const f = t - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(lerp(anchors[k][c], anchors[k + 1][c], f));
    }
    return lut;
  }
  const LUT_MAGMA = buildLUT([[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]]);
  const LUT_VIRIDIS = buildLUT([[68,1,84],[70,50,127],[54,92,141],[39,127,142],[31,161,135],[74,194,109],[159,218,58],[253,231,37]]);

  function magmaRGB(t) { const i = (clamp(t, 0, 1) * 255) | 0, o = i * 3; return [LUT_MAGMA[o], LUT_MAGMA[o + 1], LUT_MAGMA[o + 2]]; }
  function viridisRGB(t) { const i = (clamp(t, 0, 1) * 255) | 0, o = i * 3; return [LUT_VIRIDIS[o], LUT_VIRIDIS[o + 1], LUT_VIRIDIS[o + 2]]; }
  function magma(t) { const c = magmaRGB(t); return `rgb(${c[0]},${c[1]},${c[2]})`; }
  function viridis(t) { const c = viridisRGB(t); return `rgb(${c[0]},${c[1]},${c[2]})`; }

  /* ---------- Générateurs de signaux (déterministes, continus en t) ---------- */
  /* Pseudo-parole : harmoniques f0 variable + formants + plosives + fricatives.
     t en secondes absolues → cohérent entre frames / buffers. */
  function speechSample(t) {
    const SYL = 0.19;
    const syl = Math.floor(t / SYL);
    const r = hash(syl);
    const inSyl = (t / SYL) % 1;
    const active = r < 0.82 ? 1 : 0; // pauses
    const env = Math.pow(Math.sin(Math.PI * Math.min(inSyl / 0.9, 1)), 0.6) * active;
    const f0 = 118 + 26 * Math.sin(t * 2.3) + 30 * (hash(syl + 0.5) - 0.5);
    let v = 0;
    const ph = TAU * f0 * t;
    for (let h = 1; h <= 9; h++) {
      const amp = (0.55 + 0.45 * Math.sin(t * 2.7 + h * 1.7)) / Math.pow(h, 1.05);
      v += Math.sin(ph * h + h * h * 0.31) * amp;
    }
    v *= 0.22 * env;
    // plosive (burst large bande 10–20 ms) en début de syllabe
    if (active && inSyl < 0.045 && hash(syl + 0.7) > 0.45) {
      v += (hash(t * 99991) - 0.5) * 1.5 * (1 - inSyl / 0.045);
    }
    // fricative en fin de syllabe
    if (active && r > 0.58 && inSyl > 0.72) {
      v += (hash(t * 77773) - 0.5) * 0.4 * env;
    }
    return v;
  }
  /* Pseudo-musique : accords arpégés + basse + hi-hat */
  function musicSample(t) {
    const chords = [[220, 277.18, 329.63], [196, 246.94, 293.66], [174.61, 220, 261.63], [164.81, 207.65, 246.94]];
    const chord = chords[Math.floor(t / 2) % 4];
    let v = 0;
    for (let i = 0; i < chord.length; i++) {
      const f = chord[i];
      v += Math.sin(TAU * f * t) * 0.16 + Math.sin(TAU * f * 2 * t + i) * 0.05;
    }
    const bt = t % 0.5;
    v += Math.sin(TAU * (chord[0] / 2) * t) * 0.34 * Math.exp(-bt / 0.16);
    const ht = (t + 0.25) % 0.25;
    v += (hash(t * 131313) - 0.5) * 0.5 * Math.exp(-ht / 0.018);
    return v;
  }
  function noiseSample(t) { return (hash(t * 48000.123) - 0.5) * 0.8; }

  const gen = {
    speech: speechSample,
    music: musicSample,
    noise: noiseSample,
    sine: (t, f = 440) => Math.sin(TAU * f * t),
    chirp: (t, f0 = 100, f1 = 4000, T = 2) => {
      const tt = t % T;
      return Math.sin(TAU * (f0 * tt + (f1 - f0) * tt * tt / (2 * T)));
    },
    /* Remplit un Float32Array de N échantillons de fn, démarrant au temps t0, sample rate sr */
    buffer: (fn, N, sr, t0 = 0) => {
      const out = new Float32Array(N);
      for (let i = 0; i < N; i++) out[i] = fn(t0 + i / sr);
      return out;
    },
  };

  /* ---------- Formatage ---------- */
  const fmt = {
    hz: (f) => f >= 1000 ? (f / 1000).toFixed(f >= 10000 ? 1 : 2).replace(/\.?0+$/, '') + ' kHz' : Math.round(f) + ' Hz',
    ms: (s) => (s >= 0.1 ? s.toFixed(2) + ' s' : (s * 1000).toFixed(s * 1000 < 10 ? 1 : 0) + ' ms'),
    db: (d) => (d > 0 ? '+' : '') + d.toFixed(1) + ' dB',
    pct: (p) => Math.round(p * 100) + ' %',
    k: (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + ' G' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' M' : n >= 1e3 ? (n / 1e3).toFixed(1) + ' k' : '' + Math.round(n),
  };

  /* ---------- Aides au dessin ---------- */
  const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif";
  const MONO = "ui-monospace, 'Cascadia Code', Consolas, monospace";

  function text(ctx, str, x, y, o = {}) {
    ctx.font = `${o.bold ? '600 ' : ''}${o.size || 12}px ${o.mono ? MONO : FONT}`;
    ctx.fillStyle = o.color || palette.text;
    ctx.textAlign = o.align || 'left';
    ctx.textBaseline = o.baseline || 'alphabetic';
    ctx.fillText(str, x, y);
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* Boîte de diagramme bloc (titre + sous-titre), avec halo si active */
  function node(ctx, x, y, w, h, o = {}) {
    const color = o.color || palette.blue;
    ctx.save();
    if (o.active) { ctx.shadowColor = color; ctx.shadowBlur = 14; }
    roundRect(ctx, x, y, w, h, o.r ?? 8);
    ctx.fillStyle = o.fill || palette.panel;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = color;
    ctx.globalAlpha = o.active ? 1 : 0.55;
    ctx.lineWidth = o.active ? 1.6 : 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (o.title) text(ctx, o.title, x + w / 2, y + h / 2 + (o.sub ? -3 : 0), { align: 'center', baseline: 'middle', size: o.size || 12, bold: true, color: o.active ? palette.text : palette.dim });
    if (o.sub) text(ctx, o.sub, x + w / 2, y + h / 2 + 11, { align: 'center', baseline: 'middle', size: (o.size || 12) - 3, color: palette.faint, mono: true });
    ctx.restore();
  }

  /* Flèche avec tête */
  function arrow(ctx, x1, y1, x2, y2, o = {}) {
    const color = o.color || palette.dim;
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color;
    ctx.lineWidth = o.lw || 1.4;
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.globalAlpha = o.alpha ?? 1;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([]);
    const a = Math.atan2(y2 - y1, x2 - x1), s = o.head || 6;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - s * Math.cos(a - 0.45), y2 - s * Math.sin(a - 0.45));
    ctx.lineTo(x2 - s * Math.cos(a + 0.45), y2 - s * Math.sin(a + 0.45));
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  /* Forme d'onde centrée dans un rectangle */
  function wave(ctx, data, x, y, w, h, o = {}) {
    ctx.save();
    ctx.strokeStyle = o.color || palette.blue;
    ctx.lineWidth = o.lw || 1.5;
    ctx.globalAlpha = o.alpha ?? 1;
    ctx.beginPath();
    const n = data.length, cy = y + h / 2, sc = (o.scale ?? 1) * h / 2;
    for (let i = 0; i < n; i++) {
      const px = x + (i / (n - 1)) * w;
      const py = cy - clamp(data[i] * sc, -h / 2, h / 2);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  /* Barres de spectre (depuis le bas) */
  function bars(ctx, data, x, y, w, h, o = {}) {
    ctx.save();
    const n = data.length, bw = w / n;
    for (let i = 0; i < n; i++) {
      const v = clamp(data[i] * (o.scale ?? 1), 0, 1);
      const bh = v * h;
      ctx.fillStyle = o.colorFn ? o.colorFn(i / (n - 1), v) : (o.color || palette.blue);
      ctx.globalAlpha = o.alpha ?? 0.9;
      ctx.fillRect(x + i * bw, y + h - bh, Math.max(bw - (o.gap ?? 1), 0.5), bh);
    }
    ctx.restore();
  }

  /* Cadre discret avec label optionnel en coin */
  function frame(ctx, x, y, w, h, label) {
    ctx.save();
    ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
    roundRect(ctx, x, y, w, h, 6); ctx.stroke();
    if (label) text(ctx, label, x + 8, y - 6, { size: 11, color: palette.dim });
    ctx.restore();
  }

  /* Pastille / étiquette colorée */
  function chip(ctx, str, x, y, o = {}) {
    ctx.font = `600 ${o.size || 10}px ${FONT}`;
    const w = ctx.measureText(str).width + 14, h = (o.size || 10) + 8;
    const color = o.color || palette.blue;
    ctx.save();
    roundRect(ctx, x, y - h / 2, w, h, h / 2);
    ctx.fillStyle = color + '22'; ctx.fill();
    ctx.strokeStyle = color + '88'; ctx.lineWidth = 1; ctx.stroke();
    text(ctx, str, x + w / 2, y, { align: 'center', baseline: 'middle', size: o.size || 10, bold: true, color });
    ctx.restore();
    return w;
  }

  /* Point lumineux (pulse de données) */
  function glowDot(ctx, x, y, r, color) {
    ctx.save();
    ctx.shadowColor = color; ctx.shadowBlur = r * 3;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
    ctx.restore();
  }

  /* ---------- Scroller : spectrogramme défilant (offscreen) ----------
     values[0] = grave (bas de l'image). push() une colonne par appel. */
  class Scroller {
    constructor(wPx, hPx, cmapRGB) {
      this.w = Math.max(2, wPx | 0); this.h = Math.max(2, hPx | 0);
      this.cv = document.createElement('canvas');
      this.cv.width = this.w; this.cv.height = this.h;
      this.cx = this.cv.getContext('2d', { willReadFrequently: false });
      this.cx.fillStyle = '#000'; this.cx.fillRect(0, 0, this.w, this.h);
      this.cmap = cmapRGB || magmaRGB;
      this.col = this.cx.createImageData(1, this.h);
    }
    push(values) {
      this.cx.drawImage(this.cv, -1, 0);
      const d = this.col.data, n = values.length;
      for (let y = 0; y < this.h; y++) {
        const idx = Math.min(n - 1, Math.round((1 - y / (this.h - 1)) * (n - 1)));
        const c = this.cmap(clamp(values[idx], 0, 1));
        const o = y * 4;
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
      }
      this.cx.putImageData(this.col, this.w - 1, 0);
    }
    draw(ctx, x, y, w, h) {
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.cv, x, y, w, h);
      ctx.restore();
    }
  }

  const U = {
    TAU, clamp, lerp, smoothstep, ease, hash, noise1,
    fft, rfft, rfftMag, hann,
    magma, viridis, magmaRGB, viridisRGB,
    gen, fmt,
    text, roundRect, node, arrow, wave, bars, frame, chip, glowDot,
    Scroller,
    FONT, MONO,
  };

  /* ---------- Registre des modules ---------- */
  const modules = [];
  const byId = {};
  function register(def) {
    if (!def || !def.id || !def.init) { console.warn('AtlasRegister: définition invalide', def); return; }
    if (byId[def.id]) { console.warn('AtlasRegister: id en double', def.id); return; }
    modules.push(def);
    byId[def.id] = def;
  }

  /* ---------- Moteur : montage d'un module ---------- */
  function mount(def, stageWrap, controlsEl) {
    const canvas = document.createElement('canvas');
    canvas.className = 'stage-canvas';
    stageWrap.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    let frameCb = null, resizeCb = null;

    const stage = {
      canvas, ctx,
      W: 0, H: 0,
      t: 0, speed: 1, paused: false,
      onFrame(cb) { frameCb = cb; },
      onResize(cb) { resizeCb = cb; },
      clear(color) {
        ctx.fillStyle = color || palette.stage;
        ctx.fillRect(0, 0, stage.W, stage.H);
      },
      addSlider(o) {
        const row = document.createElement('label');
        row.className = 'ctl';
        const name = document.createElement('span');
        name.className = 'ctl-name'; name.textContent = o.label;
        const input = document.createElement('input');
        input.type = 'range';
        input.min = o.min; input.max = o.max; input.step = o.step ?? 1; input.value = o.value;
        const val = document.createElement('span');
        val.className = 'ctl-val';
        const f = o.format || ((v) => v);
        const upd = () => { val.textContent = f(+input.value); };
        input.addEventListener('input', () => { upd(); o.onChange && o.onChange(+input.value); });
        upd();
        row.append(name, input, val);
        controlsEl.appendChild(row);
        return { el: row, get value() { return +input.value; }, set value(v) { input.value = v; upd(); } };
      },
      addSelect(o) {
        const row = document.createElement('label');
        row.className = 'ctl';
        const name = document.createElement('span');
        name.className = 'ctl-name'; name.textContent = o.label;
        const sel = document.createElement('select');
        for (const opt of o.options) {
          const e = document.createElement('option');
          if (typeof opt === 'string') { e.value = opt; e.textContent = opt; }
          else { e.value = opt.value; e.textContent = opt.label; }
          sel.appendChild(e);
        }
        if (o.value != null) sel.value = o.value;
        sel.addEventListener('change', () => o.onChange && o.onChange(sel.value));
        row.append(name, sel);
        controlsEl.appendChild(row);
        return { el: row, get value() { return sel.value; }, set value(v) { sel.value = v; } };
      },
      addToggle(o) {
        const row = document.createElement('label');
        row.className = 'ctl ctl-toggle';
        const name = document.createElement('span');
        name.className = 'ctl-name'; name.textContent = o.label;
        const input = document.createElement('input');
        input.type = 'checkbox'; input.checked = !!o.value;
        const sw = document.createElement('span'); sw.className = 'switch';
        input.addEventListener('change', () => o.onChange && o.onChange(input.checked));
        row.append(name, input, sw);
        controlsEl.appendChild(row);
        return { el: row, get value() { return input.checked; }, set value(v) { input.checked = !!v; } };
      },
      addButton(o) {
        const btn = document.createElement('button');
        btn.className = 'ctl-btn'; btn.type = 'button'; btn.textContent = o.label;
        btn.addEventListener('click', () => o.onClick && o.onClick());
        controlsEl.appendChild(btn);
        return { el: btn };
      },
    };

    function resize() {
      const w = stageWrap.clientWidth, h = stageWrap.clientHeight;
      if (!w || !h) return;
      stage.W = w; stage.H = h;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      resizeCb && resizeCb(w, h);
    }
    const ro = new ResizeObserver(resize);
    ro.observe(stageWrap);
    resize();

    let rafId = 0, last = performance.now(), alive = true;
    function loop(now) {
      if (!alive) return;
      rafId = requestAnimationFrame(loop);
      let dt = (now - last) / 1000;
      last = now;
      if (dt > 0.05) dt = 0.05;
      if (stage.paused) dt = 0;
      dt *= stage.speed;
      stage.t += dt;
      if (frameCb && stage.W > 0) {
        try { frameCb(stage.t, dt); }
        catch (e) { console.error(`[${def.id}]`, e); alive = false; }
      }
    }

    let cleanup = null;
    try { cleanup = def.init(stage) || null; }
    catch (e) { console.error(`[${def.id}] init`, e); }
    rafId = requestAnimationFrame(loop);

    return {
      stage,
      destroy() {
        alive = false;
        cancelAnimationFrame(rafId);
        ro.disconnect();
        if (typeof cleanup === 'function') { try { cleanup(); } catch (e) { /* noop */ } }
        canvas.remove();
      },
    };
  }

  window.Atlas = { register, modules, byId, palette, U, mount };
  window.AtlasRegister = register;
})();
