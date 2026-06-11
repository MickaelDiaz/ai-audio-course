/* ============================================================
   Audio AI Atlas — « Masquage & cohérence additive »
   Masque spectral appliqué à un VRAI spectre de mélange :
   M_s = |S|² / (|S|² + |N|²), sorties complémentaires,
   erreur additive mesurée si les masques ne somment plus à 1.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  const SR = 16000;
  const N = 1024;
  const BINS = N / 2;
  const NYQ = SR / 2;
  const EPS = 1e-9;
  const LOG61 = Math.log(61);

  const db = (p) => 10 * Math.log10(Math.max(p, EPS));
  const fmtDb = (v) => (v > 0 ? '+' : '') + v.toFixed(1).replace('.', ',') + ' dB';
  const fr2 = (v) => v.toFixed(2).replace('.', ',');

  AtlasRegister({
    id: 'masking',
    title: 'Masquage spectral — séparer sans casser le mix',
    category: 'signal',
    icon: '◐',
    summary: 'Un réseau de débruitage prédit souvent un masque : il garde la voix, rejette le reste, et doit rester additif.',
    explain: `
      <p>Dans beaucoup de systèmes de séparation ou de débruitage, le réseau ne reconstruit pas directement
      l'audio : il prédit un <strong>masque spectral</strong> <code>M(f,t)</code> appliqué au mélange
      <code>X(f,t)</code>. La sortie voix devient <code>Ŝ = M·X</code>. Si le masque vaut 1, on garde ce bin ;
      s'il vaut 0, on l'éteint ; entre les deux, on atténue progressivement. Le masque idéal affiché ici est
      l'<strong>Ideal Ratio Mask</strong> : <code>|S|² / (|S|² + |N|²)</code>, calculé depuis les énergies
      réelles de la voix et de l'interférent.</p>
      <p>Le <strong>masquage binaire</strong> paraît séduisant parce qu'il décide vite : voix ou bruit. Mais il
      crée des trous et des marches brutales bin par bin, souvent perçus comme des artefacts musicaux. Un masque
      doux conserve mieux les harmoniques faibles et laisse une transition continue entre zones dominées par la
      voix et zones dominées par le reste. Les réseaux modernes apprennent souvent cette courbe douce, parfois
      avec un biais plus agressif pour supprimer davantage de bruit.</p>
      <p>La contrainte importante est la <strong>cohérence additive</strong>. Si l'on prédit deux sorties
      complémentaires, <code>Ŝ = M·X</code> et <code>Ñ = (1-M)·X</code>, alors <code>Ŝ + Ñ = X</code> exactement
      dans le domaine STFT : aucune énergie ne disparaît, aucune énergie n'est créée. Si deux masques sont prédits
      indépendamment, leur somme peut devenir inférieure ou supérieure à 1 : le remix contient alors des trous,
      des boosts, ou une coloration spectrale. Le panneau du bas mesure cette erreur en direct.</p>` ,

    init(stage) {
      const ctx = stage.ctx;

      const selNoise = stage.addSelect({
        label: 'Interférent',
        options: [
          { value: 'noise', label: 'Bruit large bande' },
          { value: 'music', label: 'Musique' },
          { value: 'both', label: 'Musique + bruit' },
        ],
        value: 'both',
      });
      const selMask = stage.addSelect({
        label: 'Type de masque',
        options: [
          { value: 'irm', label: 'IRM doux' },
          { value: 'ibm', label: 'Binaire dur' },
          { value: 'learned', label: 'Réseau agressif' },
        ],
        value: 'irm',
      });
      const slSnr = stage.addSlider({
        label: 'SNR entrée',
        min: -12, max: 12, step: 1, value: -3,
        format: (v) => fmtDb(v),
      });
      const tgComp = stage.addToggle({ label: 'Masques complémentaires', value: true });

      const voice = new Float32Array(N);
      const rest = new Float32Array(N);
      const mix = new Float32Array(N);

      const magV = new Float32Array(BINS);
      const magR = new Float32Array(BINS);
      const magX = new Float32Array(BINS);
      const dispV = new Float32Array(BINS);
      const dispR = new Float32Array(BINS);
      const dispX = new Float32Array(BINS);
      const dispEstV = new Float32Array(BINS);
      const dispEstR = new Float32Array(BINS);
      const dispSum = new Float32Array(BINS);
      const mask = new Float32Array(BINS);
      const maskRest = new Float32Array(BINS);
      const maskS = new Float32Array(BINS);
      const maskRestS = new Float32Array(BINS);
      const sumMaskS = new Float32Array(BINS);

      let specMax = 0.02;
      let vPowSm = 0.04;
      let rPowSm = 0.04;
      const met = { inSnr: -3, outSnr: 0, improve: 0, loss: 0, err: 0, keep: 1 };

      function restSample(t, kind) {
        if (kind === 'noise') {
          return U.gen.noise(t) * 0.82 + Math.sin(U.TAU * 92 * t + 0.3) * 0.08;
        }
        if (kind === 'music') {
          return U.gen.music(t) * 0.92;
        }
        return U.gen.music(t) * 0.56 + U.gen.noise(t) * 0.72;
      }

      function shapedMask(base, mode, i, t) {
        if (mode === 'ibm') return base > 0.52 ? 1 : 0;
        if (mode === 'learned') {
          const a = 1.85;
          const pa = Math.pow(Math.max(base, 0), a);
          const qa = Math.pow(Math.max(1 - base, 0), a);
          const warped = pa / (pa + qa + EPS);
          const ripple = (U.noise1(i * 0.045 + t * 0.45) - 0.5) * 0.08;
          return U.clamp(warped - 0.07 * Math.pow(1 - base, 0.7) + ripple, 0, 1);
        }
        return base;
      }

      function tag(s, x, y, o = {}) {
        ctx.save();
        const size = o.size || 10;
        ctx.font = `600 ${size}px ${U.FONT}`;
        const tw = ctx.measureText(s).width;
        const ax = o.align === 'right' ? x - tw : o.align === 'center' ? x - tw / 2 : x;
        ctx.globalAlpha = 0.72 * (o.alpha ?? 1);
        ctx.fillStyle = palette.stage;
        ctx.fillRect(ax - 4, y - size - 3, tw + 8, size + 7);
        ctx.globalAlpha = o.alpha ?? 1;
        U.text(ctx, s, ax, y, { size, color: o.color || palette.dim, bold: true, mono: o.mono });
        ctx.restore();
      }

      function freqTicks(x, y, w, alignBottom) {
        ctx.save();
        ctx.strokeStyle = palette.grid;
        for (const f of [0, 2000, 4000, 6000, 8000]) {
          const tx = x + (f / NYQ) * w;
          ctx.beginPath();
          ctx.moveTo(tx, y);
          ctx.lineTo(tx, y + (alignBottom ? 4 : -4));
          ctx.stroke();
          U.text(ctx, U.fmt.hz(f), U.clamp(tx, x + 12, x + w - 18), y + (alignBottom ? 14 : -8),
            { size: 8.5, color: palette.faint, align: 'center', mono: true });
        }
        ctx.restore();
      }

      function drawWave(r, narrow) {
        U.frame(ctx, r.x, r.y, r.w, r.h, narrow ? 'Mélange temporel' : 'Temps — x = voix + interférent');
        const cy = r.y + r.h / 2;
        ctx.save();
        ctx.strokeStyle = palette.grid;
        ctx.beginPath();
        ctx.moveTo(r.x + 2, cy);
        ctx.lineTo(r.x + r.w - 2, cy);
        ctx.stroke();
        ctx.restore();
        U.wave(ctx, rest, r.x + 2, r.y + 4, r.w - 4, r.h - 8,
          { color: palette.rest, lw: 1, alpha: 0.35, scale: 0.7 });
        U.wave(ctx, voice, r.x + 2, r.y + 4, r.w - 4, r.h - 8,
          { color: palette.voice, lw: 1, alpha: 0.5, scale: 0.7 });
        U.wave(ctx, mix, r.x + 2, r.y + 4, r.w - 4, r.h - 8,
          { color: palette.mix, lw: 1.7, alpha: 0.94, scale: 0.58 });
        if (r.w > 300) {
          let x = r.x + 8;
          x += U.chip(ctx, 'voix S', x, r.y + 14, { color: palette.voice, size: 9 }) + 6;
          x += U.chip(ctx, 'reste N', x, r.y + 14, { color: palette.rest, size: 9 }) + 6;
          U.chip(ctx, 'mix X', x, r.y + 14, { color: palette.mix, size: 9 });
        }
        tag(`SNR réel ${fmtDb(met.inSnr)}`, r.x + r.w - 8, r.y + 16,
          { color: palette.mix, align: 'right', size: 9.5 });
      }

      function drawSpectrum(r, narrow) {
        U.frame(ctx, r.x, r.y, r.w, r.h, narrow ? 'Spectre' : 'Spectre du mélange — énergies réelles');
        const base = r.y + r.h - 16;
        const h = Math.max(12, r.h - 24);
        const bw = r.w / BINS;
        ctx.save();
        for (let i = 0; i < BINS; i++) {
          const x = r.x + i * bw;
          const hv = dispV[i] * h;
          const hr = dispR[i] * h;
          ctx.globalAlpha = 0.28;
          ctx.fillStyle = palette.rest;
          ctx.fillRect(x, base - hr, Math.max(bw, 0.55), hr);
          ctx.globalAlpha = 0.38;
          ctx.fillStyle = palette.voice;
          ctx.fillRect(x, base - hv, Math.max(bw, 0.55), hv);
          ctx.globalAlpha = 0.62;
          ctx.fillStyle = palette.mix;
          const hx = dispX[i] * h;
          ctx.fillRect(x, base - hx, Math.max(bw - 0.2, 0.45), Math.max(1, hx));
        }
        ctx.restore();
        freqTicks(r.x, base + 1, r.w, true);
        if (!narrow) {
          tag('où la voix domine → masque proche de 1', r.x + 8, r.y + 17,
            { color: palette.voice, size: 9 });
          tag('où le reste domine → masque proche de 0', r.x + r.w - 8, r.y + 17,
            { color: palette.rest, align: 'right', size: 9 });
        }
      }

      function drawMask(r, active, narrow) {
        U.frame(ctx, r.x, r.y, r.w, r.h, narrow ? 'Masque M(f,t)' : 'Masque spectral — garder la voix, rejeter le reste');
        const left = r.x + 2;
        const right = r.x + r.w - 2;
        const top = r.y + 16;
        const bottom = r.y + r.h - 18;
        const hh = Math.max(8, bottom - top);
        const xOf = (i) => left + (i / (BINS - 1)) * (right - left);
        const yOf = (v) => bottom - U.clamp(v, 0, 1) * hh;

        ctx.save();
        ctx.strokeStyle = palette.grid;
        ctx.beginPath();
        ctx.moveTo(left, yOf(0.5));
        ctx.lineTo(right, yOf(0.5));
        ctx.stroke();
        ctx.globalAlpha = 0.16;
        for (let i = 0; i < BINS; i++) {
          const x = xOf(i);
          const w = Math.max((right - left) / BINS, 0.55);
          ctx.fillStyle = dispV[i] >= dispR[i] ? palette.voice : palette.rest;
          ctx.fillRect(x, bottom - Math.max(dispX[i] * hh * 0.55, 1), w, Math.max(dispX[i] * hh * 0.55, 1));
        }
        ctx.globalAlpha = 1;

        ctx.beginPath();
        ctx.moveTo(left, bottom);
        for (let i = 0; i < BINS; i++) ctx.lineTo(xOf(i), yOf(maskS[i]));
        ctx.lineTo(right, bottom);
        ctx.closePath();
        ctx.fillStyle = palette.voice + '33';
        ctx.fill();
        ctx.strokeStyle = palette.voice;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let i = 0; i < BINS; i++) {
          const x = xOf(i), y = yOf(maskS[i]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();

        ctx.strokeStyle = palette.rest;
        ctx.lineWidth = 1.25;
        ctx.globalAlpha = 0.82;
        ctx.beginPath();
        for (let i = 0; i < BINS; i++) {
          const x = xOf(i), y = yOf(maskRestS[i]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;

        if (!tgComp.value) {
          ctx.strokeStyle = palette.red;
          ctx.lineWidth = 1.2;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          for (let i = 0; i < BINS; i++) {
            const x = xOf(i);
            const y = yOf(U.clamp(sumMaskS[i] - 0.5, 0, 1));
            i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          }
          ctx.stroke();
          ctx.setLineDash([]);
        }

        const ax = xOf(active);
        const ay = yOf(maskS[active]);
        ctx.strokeStyle = palette.blue;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.moveTo(ax, top);
        ctx.lineTo(ax, bottom);
        ctx.stroke();
        ctx.globalAlpha = 1;
        U.glowDot(ctx, ax, ay, 2.5, palette.voice);

        const f = active * SR / N;
        const m = maskS[active];
        const mrActive = maskRestS[active];
        const sum = m + mrActive;
        const labelY = top + 12;
        tag(`M_voix(${U.fmt.hz(f)}) = ${fr2(m)}`, U.clamp(ax + 8, r.x + 90, r.x + r.w - 90), labelY,
          { color: palette.voice, size: narrow ? 8.5 : 9.5, align: 'center', mono: true });
        if (!narrow) {
          tag(`M_reste = ${fr2(mrActive)} · somme = ${fr2(sum)}`, r.x + r.w - 8, bottom - 3,
            { color: Math.abs(sum - 1) < 0.02 ? palette.green : palette.red, align: 'right', size: 9, mono: true });
          U.text(ctx, '0', left, bottom + 12, { size: 8.5, color: palette.faint, mono: true });
          U.text(ctx, '0,5', left, yOf(0.5) - 3, { size: 8.5, color: palette.faint, mono: true });
          U.text(ctx, '1', left, top + 2, { size: 8.5, color: palette.faint, mono: true });
        }
        freqTicks(left, bottom + 1, right - left, true);
      }

      function drawStrip(data, x, y, w, h, color, alpha) {
        const step = Math.max(1, Math.floor(BINS / Math.max(40, Math.floor(w))));
        const cols = Math.ceil(BINS / step);
        const bw = w / cols;
        ctx.save();
        ctx.fillStyle = palette.panel;
        ctx.globalAlpha = 0.65;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = alpha ?? 0.88;
        ctx.fillStyle = color;
        for (let c = 0; c < cols; c++) {
          let mx = 0;
          const end = Math.min(BINS, c * step + step);
          for (let i = c * step; i < end; i++) if (data[i] > mx) mx = data[i];
          const bh = Math.max(1, mx * (h - 2));
          ctx.fillRect(x + c * bw, y + h - bh, Math.max(bw - 0.4, 0.6), bh);
        }
        ctx.restore();
      }

      function drawReadouts(x, y, w, h, narrow) {
        ctx.save();
        U.roundRect(ctx, x, y, w, h, 6);
        ctx.fillStyle = palette.panel;
        ctx.globalAlpha = 0.72;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = palette.grid;
        ctx.stroke();
        const px = x + 10;
        const pr = x + w - 10;
        let yy = y + 18;
        const row = (lab, val, color) => {
          U.text(ctx, lab, px, yy, { size: narrow ? 8.5 : 9.5, color: palette.dim });
          U.text(ctx, val, pr, yy, { size: narrow ? 9.5 : 11, color, align: 'right', bold: true, mono: true });
          yy += narrow ? 14 : 16;
        };
        row('SNR entrée', fmtDb(met.inSnr), palette.mix);
        row('SNR sortie voix', fmtDb(met.outSnr), palette.voice);
        row('gain obtenu', fmtDb(met.improve), met.improve >= 0 ? palette.green : palette.red);
        if (!narrow || h > 86) row('voix conservée', U.fmt.pct(met.keep), palette.voice);
        const errColor = met.err < 0.015 ? palette.green : met.err < 0.08 ? palette.orange : palette.red;
        row('erreur additive', U.fmt.pct(met.err), errColor);
        if (!narrow && h > 118) {
          yy += 3;
          U.text(ctx, tgComp.value ? 'Ŝ + Ñ = X : remix contraint' : 'masques indépendants : trous / boosts',
            px, yy, { size: 9.5, color: errColor, bold: true });
          U.text(ctx, 'L’erreur est pondérée par |X(f,t)|.', px, yy + 14,
            { size: 8.5, color: palette.faint });
        }
        ctx.restore();
      }

      function drawCoherence(r, narrow) {
        U.frame(ctx, r.x, r.y, r.w, r.h, narrow ? 'Cohérence additive' : 'Séparation — cohérence additive du remix');
        const pad = narrow ? 7 : 10;

        if (narrow) {
          const labelW = 58;
          const laneH = Math.max(10, Math.min(16, (r.h - 46) / 2));
          const lanesX = r.x + pad;
          const lanesY = r.y + 15;
          const lanesW = r.w - 2 * pad;
          const rows = [
            ['X', dispX, palette.mix],
            ['Ŝ+Ñ', dispSum, met.err < 0.015 ? palette.green : palette.red],
          ];
          for (let k = 0; k < rows.length; k++) {
            const yy = lanesY + k * (laneH + 6);
            U.text(ctx, rows[k][0], lanesX, yy + laneH - 3,
              { size: 9, color: rows[k][2], bold: true, mono: true });
            drawStrip(rows[k][1], lanesX + labelW, yy, lanesW - labelW, laneH, rows[k][2], 0.84);
          }
          const errColor = met.err < 0.015 ? palette.green : met.err < 0.08 ? palette.orange : palette.red;
          const yb = r.y + r.h - 20;
          U.roundRect(ctx, r.x + pad, yb - 10, r.w - 2 * pad, 22, 6);
          ctx.fillStyle = palette.panel;
          ctx.globalAlpha = 0.72;
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = palette.grid;
          ctx.stroke();
          U.text(ctx, `${fmtDb(met.inSnr)} → ${fmtDb(met.outSnr)}`, r.x + pad + 8, yb + 4,
            { size: 9, color: palette.voice, mono: true, bold: true });
          U.text(ctx, `err ${U.fmt.pct(met.err)}`, r.x + r.w - pad - 8, yb + 4,
            { size: 9, color: errColor, align: 'right', mono: true, bold: true });
          return;
        }

        const readsW = narrow ? r.w : Math.max(170, Math.min(240, r.w * 0.30));
        const lanesW = r.w - readsW - 3 * pad;
        const lanesX = r.x + pad;
        const lanesY = r.y + 18;
        const laneH = Math.max(13, Math.min(24, (r.h - 38) / 4));
        const gap = Math.max(4, Math.min(8, laneH * 0.28));

        const rows = [
          ['X mélange', dispX, palette.mix],
          ['Ŝ = M·X', dispEstV, palette.voice],
          [tgComp.value ? 'Ñ = (1-M)·X' : 'Ñ = M_n·X', dispEstR, palette.rest],
          ['Ŝ + Ñ', dispSum, met.err < 0.015 ? palette.green : palette.red],
        ];
        for (let k = 0; k < rows.length; k++) {
          const yy = lanesY + k * (laneH + gap);
          U.text(ctx, rows[k][0], lanesX, yy + laneH - 4,
            { size: narrow ? 8.5 : 9.5, color: rows[k][2], bold: true, mono: true });
          drawStrip(rows[k][1], lanesX + (narrow ? 66 : 88), yy, lanesW - (narrow ? 68 : 90), laneH, rows[k][2], k === 3 ? 0.72 : 0.86);
        }

        const bx = lanesX;
        const by = r.y + r.h - 24;
        const bw = lanesW - 4;
        const mx = bx + bw * 0.52;
        U.node(ctx, bx, by - 1, 62, 22, { title: 'X', sub: 'mix', color: palette.mix, size: 11 });
        U.arrow(ctx, bx + 64, by + 10, mx - 8, by + 10, { color: palette.dim, lw: 1.1, head: 5 });
        U.node(ctx, mx - 6, by - 1, 84, 22, {
          title: tgComp.value ? 'M / 1-M' : 'M_s / M_n',
          sub: tgComp.value ? 'somme=1' : 'libres',
          color: tgComp.value ? palette.green : palette.red,
          active: !tgComp.value && met.err > 0.03,
          size: 10,
        });
        U.arrow(ctx, mx + 80, by + 10, bx + bw - 74, by + 10, { color: palette.dim, lw: 1.1, head: 5 });
        U.node(ctx, bx + bw - 70, by - 1, 70, 22, {
          title: 'Σ',
          sub: tgComp.value ? '= X' : '≠ X',
          color: tgComp.value ? palette.green : palette.red,
          size: 11,
        });

        drawReadouts(r.x + r.w - readsW - pad, r.y + 18, readsW, r.h - 28, false);
      }

      stage.onFrame((t, dt) => {
        const W = stage.W, H = stage.H;
        const narrow = W < 560;
        stage.clear();

        const snrDb = slSnr.value;
        const kind = selNoise.value;
        const mode = selMask.value;
        const coherent = tgComp.value;

        let vPow = 0;
        let rPow = 0;
        for (let i = 0; i < N; i++) {
          const tt = t + i / SR;
          const s = U.gen.speech(tt) * 1.25;
          const n = restSample(tt, kind);
          voice[i] = s;
          rest[i] = n;
          vPow += s * s;
          rPow += n * n;
        }
        vPow /= N;
        rPow /= N;
        const kg = dt > 0 ? 1 - Math.exp(-dt * 3) : 1;
        vPowSm += (vPow - vPowSm) * kg;
        rPowSm += (rPow - rPowSm) * kg;

        const targetRatio = Math.pow(10, snrDb / 10);
        const restGain = U.clamp(Math.sqrt(Math.max(vPowSm, 1e-5) / (Math.max(rPowSm, 1e-5) * targetRatio)), 0.05, 6);
        for (let i = 0; i < N; i++) {
          rest[i] *= restGain;
          mix[i] = voice[i] + rest[i];
        }

        magV.set(U.rfftMag(voice));
        magR.set(U.rfftMag(rest));
        magX.set(U.rfftMag(mix));

        let maxNow = 1e-4;
        let sumV = 0;
        let sumR = 0;
        let keep = 0;
        let leak = 0;
        let keptVoice = 0;
        let sumErr = 0;
        let sumWeight = 0;
        const kS = dt > 0 ? 1 - Math.exp(-dt * 12) : 1;

        for (let i = 0; i < BINS; i++) {
          const sv = magV[i];
          const nr = magR[i];
          const x = magX[i];
          const vE = sv * sv;
          const rE = nr * nr;
          const base = vE / (vE + rE + EPS);
          const m = shapedMask(base, mode, i, t);
          let mr;
          if (coherent) {
            mr = 1 - m;
          } else {
            const inv = 1 - base;
            mr = shapedMask(inv, mode, i + 97, t + 0.31);
            mr = U.clamp(mr + 0.08 * Math.sin(i * 0.035 + t * 1.7), 0, 1);
          }
          mask[i] = m;
          maskRest[i] = mr;
          maskS[i] += (m - maskS[i]) * kS;
          maskRestS[i] += (mr - maskRestS[i]) * kS;
          sumMaskS[i] += (m + mr - sumMaskS[i]) * kS;

          sumV += vE;
          sumR += rE;
          keep += m * m * vE;
          leak += m * m * rE;
          keptVoice += m * vE;
          sumErr += Math.abs(m + mr - 1) * x;
          sumWeight += x;
          if (x > maxNow) maxNow = x;
        }
        specMax = Math.max(maxNow, specMax * (1 - 0.45 * dt), 1e-4);

        let outVMax = 0;
        for (let i = 0; i < BINS; i++) {
          const x = magX[i];
          const d = (v) => Math.log1p((v / specMax) * 60) / LOG61;
          dispV[i] = d(magV[i]);
          dispR[i] = d(magR[i]);
          dispX[i] = d(x);
          dispEstV[i] = d(mask[i] * x);
          dispEstR[i] = d(maskRest[i] * x);
          dispSum[i] = d((mask[i] + maskRest[i]) * x);
          if (dispEstV[i] > outVMax) outVMax = dispEstV[i];
        }

        const inSnr = db(sumV / (sumR + EPS));
        const outSnr = db(keep / (leak + EPS));
        const keepPct = U.clamp(keptVoice / (sumV + EPS), 0, 1.2);
        const err = U.clamp(sumErr / (sumWeight + EPS), 0, 1);
        const km = dt > 0 ? 1 - Math.exp(-dt * 8) : 1;
        met.inSnr += (inSnr - met.inSnr) * km;
        met.outSnr += (outSnr - met.outSnr) * km;
        met.improve += ((outSnr - inSnr) - met.improve) * km;
        met.loss += (db(keep / (sumV + EPS)) - met.loss) * km;
        met.err += (err - met.err) * km;
        met.keep += (keepPct - met.keep) * km;

        const m = narrow ? 8 : 10;
        const topPad = narrow ? 18 : 24;
        const gap = narrow ? 16 : 20;
        const innerW = W - 2 * m;
        const innerH = H - topPad - m;

        if (!narrow) {
          let x = m;
          x += U.chip(ctx, mode === 'irm' ? 'IRM : ratio doux' : mode === 'ibm' ? 'IBM : décision dure' : 'réseau : masque biaisé',
            x, 13, { color: mode === 'ibm' ? palette.orange : palette.voice }) + 8;
          x += U.chip(ctx, coherent ? 'cohérence additive ON' : 'cohérence additive cassée',
            x, 13, { color: coherent ? palette.green : palette.red }) + 8;
          U.chip(ctx, `interférent : ${kind === 'noise' ? 'bruit' : kind === 'music' ? 'musique' : 'mix'}`,
            x, 13, { color: palette.rest });
          U.text(ctx, 'masquage spectral — séparation de sources', W - m, 17,
            { size: 11, color: palette.dim, align: 'right' });
        } else {
          U.text(ctx, 'Masquage spectral — M·X', m, 13, { size: 10, color: palette.dim, bold: true });
        }

        if (!narrow) {
          const totalH = Math.max(120, innerH - 2 * gap);
          const hTop = Math.max(58, Math.round(totalH * 0.31));
          const hMask = Math.max(70, Math.round(totalH * 0.35));
          const hBot = Math.max(62, totalH - hTop - hMask);
          const yTop = topPad;
          const yMask = yTop + hTop + gap;
          const yBot = yMask + hMask + gap;
          const waveW = Math.round(innerW * 0.47);
          drawWave({ x: m, y: yTop, w: waveW, h: hTop }, false);
          drawSpectrum({ x: m + waveW + 12, y: yTop, w: innerW - waveW - 12, h: hTop }, false);
          drawMask({ x: m, y: yMask, w: innerW, h: hMask }, Math.floor((t * 54) % BINS), false);
          drawCoherence({ x: m, y: yBot, w: innerW, h: hBot }, false);
        } else {
          const hTop = Math.max(48, Math.round(innerH * 0.25));
          const hMask = Math.max(74, Math.round(innerH * 0.36));
          const hBot = Math.max(72, innerH - hTop - hMask - 2 * gap);
          const yTop = topPad;
          const yMask = yTop + hTop + gap;
          const yBot = yMask + hMask + gap;
          drawWave({ x: m, y: yTop, w: innerW, h: hTop }, true);
          drawMask({ x: m, y: yMask, w: innerW, h: hMask }, Math.floor((t * 42) % BINS), true);
          drawCoherence({ x: m, y: yBot, w: innerW, h: hBot }, true);
        }
      });
    },
  });
})();
