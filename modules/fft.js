/* ============================================================
   Audio AI Atlas — module « Fourier : du temps à la fréquence »
   Gauche : forme d'onde vivante (+ fenêtre de Hann réelle).
   Droite : spectre VRAI (U.rfftMag, 512 pts) en barres annotées en Hz.
   Mode décomposition : chaque sinusoïde reliée à SA barre du spectre.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;
  const TAU = U.TAU;

  /* ---- constantes du module (toutes les valeurs affichées en découlent) ---- */
  const FS = 8000;             // Hz — fréquence d'échantillonnage
  const N = 512;               // échantillons par trame
  const DF = FS / N;           // 15,625 Hz — résolution fréquentielle VRAIE
  const DUR = N / FS;          // 0,064 s — durée d'observation
  const SHOW = 124;            // bins affichés (sur N/2 = 256), lisibilité
  const SINE_F = 220;          // Hz — préréglages sinus
  const SINE2_F = 700;         // Hz — second sinus (volontairement hors-bin)
  const SQ_F0 = 110;           // Hz — fondamentale du carré (k ≤ 17 reste visible)
  const SQ_A = 0.72;           // amplitude du carré idéal
  const CYC = 2;               // s — cycle de mise en avant d'une composante

  const COMP_COLORS = [palette.blue, palette.teal, palette.violet, palette.pink,
    palette.orange, palette.green, palette.yellow, palette.red, palette.voice];

  const frNum = (v) => String(v).replace('.', ',');
  const fr1 = (v) => v.toFixed(1).replace('.', ',');

  AtlasRegister({
    id: 'fft',
    title: 'Fourier — du temps à la fréquence',
    category: 'signal',
    icon: 'ƒ',
    summary: 'Tout signal est une somme de sinusoïdes : la FFT révèle lesquelles, et avec quelle force.',
    explain: `
      <p>Le théorème de Fourier dit que <strong>tout signal</strong> peut s'écrire comme une
      <strong>somme de sinusoïdes</strong>, chacune définie par sa fréquence, son <strong>amplitude</strong>
      (la hauteur de sa barre dans le spectre) et sa <strong>phase</strong> (son décalage temporel, que la
      magnitude seule ne montre pas). La <strong>FFT</strong> (<em>Fast Fourier Transform</em>) calcule cette
      décomposition en <code>O(N log N)</code> au lieu de <code>O(N²)</code> pour la DFT naïve — c'est ce qui
      rend l'analyse spectrale temps réel possible : ici une trame de 512 échantillons est analysée à chaque image.</p>
      <p>La <strong>résolution fréquentielle</strong> vaut <code>Δf = fs / N</code> : avec fs = 8 kHz et
      N = 512, chaque bin couvre 15,625 Hz. C'est un compromis fondamental : distinguer deux fréquences
      proches exige d'observer le signal <em>plus longtemps</em> (N grand), mais on perd alors en précision
      temporelle — on sait <em>quoi</em> sans savoir exactement <em>quand</em>. Ce dilemme temps-fréquence
      est au cœur du spectrogramme et de tous les front-ends audio.</p>
      <p>Découper une trame de 64 ms revient à multiplier le signal par une <strong>fenêtre rectangulaire</strong> :
      les discontinuités aux bords font « fuir » l'énergie d'une sinusoïde sur les bins voisins
      (<strong>fuites spectrales</strong>, sidelobes à −13 dB). La <strong>fenêtre de Hann</strong> adoucit
      les bords : le lobe principal s'élargit un peu, mais les fuites chutent drastiquement — activez le
      toggle pour comparer sur le sinus à 700 Hz, volontairement placé entre deux bins.</p>
      <p>Le préréglage « Carré » illustre la convergence : un carré = somme des harmoniques
      <strong>impaires</strong> d'amplitude 4/(πk). Avec un nombre fini d'harmoniques, des oscillations
      persistent près des fronts et le dépassement tend vers ≈ 9 % quel que soit leur nombre :
      c'est le <strong>phénomène de Gibbs</strong>, mesuré en direct sur la trame.</p>`,

    init(stage) {
      const ctx = stage.ctx;

      /* ---- buffers pré-alloués (rien de gros n'est créé dans onFrame) ---- */
      const buf = new Float32Array(N);       // trame brute
      const wbuf = new Float32Array(N);      // trame fenêtrée (affichage)
      const sqBuf = new Float32Array(N);     // carré idéal (référence)
      const compBuf = new Float32Array(N);   // composante courante (réutilisé)
      const magS = new Float32Array(N / 2);  // spectre lissé pour l'affichage
      const binOwner = new Int16Array(SHOW); // bin -> index de composante
      const hannW = U.hann(N);
      let comps = [];                        // composantes du préréglage courant
      let compKey = '';

      /* ---- contrôles ---- */
      const selSig = stage.addSelect({
        label: 'Signal',
        options: ['Sinus pur', 'Deux sinus', 'Carré', 'Pseudo-parole'],
        value: 'Sinus pur',
        onChange: (v) => { if (v !== 'Pseudo-parole') tgDec.value = true; },
      });
      const slHarm = stage.addSlider({
        label: 'Harmoniques (carré)', min: 1, max: 9, step: 1, value: 5,
        format: (v) => `${v} → k ≤ ${2 * v - 1}`,
      });
      const tgHann = stage.addToggle({ label: 'Fenêtre de Hann', value: false });
      const tgDec = stage.addToggle({ label: 'Décomposition', value: true });

      /* ---- composantes : reconstruites uniquement au changement ---- */
      function rebuildComps(preset, nH) {
        comps = [];
        if (preset === 'Sinus pur') comps.push({ f: SINE_F, a: 0.85 });
        else if (preset === 'Deux sinus') {
          comps.push({ f: SINE_F, a: 0.70 });
          comps.push({ f: SINE2_F, a: 0.45 });
        } else if (preset === 'Carré') {
          for (let i = 0; i < nH; i++) {
            const k = 2 * i + 1;
            comps.push({ f: k * SQ_F0, a: SQ_A * 4 / (Math.PI * k), k });
          }
        }
        binOwner.fill(-1);
        comps.forEach((c, idx) => {
          c.bin = Math.round(c.f / DF);
          for (let o = -1; o <= 1; o++) {
            const b = c.bin + o;
            if (b >= 0 && b < SHOW) binOwner[b] = idx;
          }
        });
      }

      /* ---- panneau temporel ---- */
      function drawWavePanel(r, t, preset, hannOn, color, ampNow, nH) {
        U.frame(ctx, r.x, r.y, r.w, r.h,
          r.w > 240 ? `Temps — trame de ${Math.round(DUR * 1000)} ms (fs = 8 kHz)` : 'Temps');
        const cy = r.y + r.h / 2, SC = 0.82;
        ctx.save();
        ctx.strokeStyle = palette.grid;
        ctx.beginPath(); ctx.moveTo(r.x + 1, cy); ctx.lineTo(r.x + r.w - 1, cy); ctx.stroke();
        ctx.restore();

        if (preset === 'Carré') {            // carré idéal : référence de convergence
          ctx.save(); ctx.setLineDash([5, 4]);
          U.wave(ctx, sqBuf, r.x, r.y, r.w, r.h, { color: palette.dim, lw: 1, alpha: 0.4, scale: SC });
          ctx.restore();
        }
        if (hannOn) {
          U.wave(ctx, buf, r.x, r.y, r.w, r.h, { color, lw: 1, alpha: 0.16, scale: SC });
          const es = U.clamp(ampNow, 0.25, 1) * SC * r.h / 2;  // enveloppe à l'échelle du signal
          ctx.save();
          ctx.strokeStyle = palette.rest; ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]); ctx.globalAlpha = 0.85;
          for (let sgn = -1; sgn <= 1; sgn += 2) {
            ctx.beginPath();
            for (let i = 0; i < N; i += 4) {
              const px = r.x + (i / (N - 1)) * r.w, py = cy - sgn * hannW[i] * es;
              if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
          }
          ctx.restore();
          U.wave(ctx, wbuf, r.x, r.y, r.w, r.h, { color, lw: 1.6, scale: SC });
          if (r.w > 230) U.text(ctx, 'fenêtre de Hann × signal', r.x + r.w / 2, r.y + 13,
            { size: 10, color: palette.rest, align: 'center' });
        } else {
          U.wave(ctx, buf, r.x, r.y, r.w, r.h, { color, lw: 1.6, scale: SC });
        }

        if (preset === 'Carré') {
          /* Gibbs : dépassement réellement mesuré sur la trame courante */
          let pk = 0, pi = 0;
          for (let i = 0; i < N; i++) { const a = Math.abs(buf[i]); if (a > pk) { pk = a; pi = i; } }
          const over = (pk - SQ_A) / SQ_A;
          if (nH >= 2 && r.w > 200 && r.h > 60) {
            const px = r.x + (pi / (N - 1)) * r.w;
            const py = cy - U.clamp(buf[pi] * SC * r.h / 2, -r.h / 2, r.h / 2);
            U.glowDot(ctx, px, py, 2.5, palette.orange);
            const lx = U.clamp(px, r.x + 80, r.x + r.w - 80);
            const ly = buf[pi] > 0 ? Math.max(r.y + 12, py - 12) : Math.min(r.y + r.h - 8, py + 18);
            U.text(ctx, `Gibbs : dépassement +${fr1(over * 100)} %`, lx, ly,
              { size: 10, color: palette.orange, align: 'center' });
          }
          if (r.w > 260) U.text(ctx,
            `${nH} harmonique${nH > 1 ? 's' : ''} impaire${nH > 1 ? 's' : ''} (k ≤ ${2 * nH - 1}) — convergence vers le carré`,
            r.x + r.w / 2, r.y + r.h - 6, { size: 10, color: palette.dim, align: 'center' });
        }
        if (preset === 'Pseudo-parole' && r.w > 260) U.text(ctx,
          'pseudo-parole : f₀ ≈ 120 Hz, harmoniques + bruit large bande',
          r.x + r.w / 2, r.y + r.h - 6, { size: 10, color: palette.dim, align: 'center' });
      }

      /* ---- panneau spectral ---- */
      function drawSpectrum(r, color, activeIdx, pulse, showComps, hannOn) {
        U.frame(ctx, r.x, r.y, r.w, r.h,
          r.w > 260 ? 'Fréquence — spectre |X(f)| (FFT réelle, 512 points)' : 'Fréquence');
        const bw = r.w / SHOW, baseY = r.y + r.h - 1;
        ctx.save();
        for (let i = 0; i < SHOW; i++) {
          const v = U.clamp(magS[i], 0, 1), bh = v * (r.h - 5);
          const owner = binOwner[i];
          const col = owner >= 0 ? COMP_COLORS[owner % COMP_COLORS.length] : color;
          const act = showComps && owner === activeIdx;
          ctx.globalAlpha = owner >= 0 ? (act ? 0.6 + 0.4 * pulse : 0.85) : 0.45;
          ctx.fillStyle = col;
          if (act && pulse > 0.05) { ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = 12 * pulse; }
          ctx.fillRect(r.x + i * bw, baseY - bh, Math.max(bw - 1, 0.6), bh);
          if (act && pulse > 0.05) ctx.restore();
        }
        ctx.restore();

        /* halo + fréquence vraie sur la barre active */
        if (showComps && activeIdx >= 0 && activeIdx < comps.length) {
          const c = comps[activeIdx];
          if (c.bin < SHOW) {
            const col = COMP_COLORS[activeIdx % COMP_COLORS.length];
            const bx = r.x + (c.bin + 0.5) * bw;
            const by = baseY - U.clamp(magS[c.bin], 0, 1) * (r.h - 5);
            U.glowDot(ctx, bx, by, 2 + 2.5 * pulse, col);
            U.text(ctx, `${Math.round(c.f)} Hz`, U.clamp(bx, r.x + 22, r.x + r.w - 24),
              Math.max(r.y + 11, by - 7), { size: 10, bold: true, color: col, align: 'center', mono: true });
          }
        }

        /* axe en Hz (étendue vraie : SHOW × Δf) */
        const maxF = SHOW * DF;
        ctx.save();
        ctx.strokeStyle = palette.grid;
        ctx.beginPath(); ctx.moveTo(r.x, baseY + 1); ctx.lineTo(r.x + r.w, baseY + 1); ctx.stroke();
        for (let f = 0; f <= maxF; f += 500) {
          const tx = r.x + (f / maxF) * r.w;
          ctx.beginPath(); ctx.moveTo(tx, baseY + 1); ctx.lineTo(tx, baseY + 5); ctx.stroke();
          U.text(ctx, U.fmt.hz(f), Math.max(tx, r.x + 10), baseY + 14,
            { size: 9, color: palette.faint, align: f === 0 ? 'left' : 'center', mono: true });
        }
        if (r.w > 300) U.text(ctx, U.fmt.hz(maxF), r.x + r.w, baseY + 14,
          { size: 9, color: palette.faint, align: 'right', mono: true });
        ctx.restore();

        if (r.w > 280) U.text(ctx,
          hannOn ? 'Hann : fuites réduites, lobe élargi' : 'rectangulaire : fuites spectrales',
          r.x + r.w - 6, r.y + 12, { size: 10, color: hannOn ? palette.rest : palette.dim, align: 'right' });
      }

      /* ---- composantes empilées + liens pointillés vers le spectre ---- */
      function drawComps(r, rowH, nShown, t, activeIdx, pulse, specR, intro) {
        const bw = specR.w / SHOW;
        for (let s = 0; s < nShown; s++) {
          const c = comps[s];
          const col = COMP_COLORS[s % COMP_COLORS.length];
          const ry = r.y + s * rowH, rcy = ry + rowH / 2;
          const act = s === activeIdx;
          for (let i = 0; i < N; i++) compBuf[i] = Math.sin(TAU * c.f * (t + i / FS));
          const rel = c.a / comps[0].a;       // amplitudes relatives VRAIES (décroissance 1/k)
          if (act && pulse > 0.05) { ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = 9 * pulse; }
          U.wave(ctx, compBuf, r.x, ry + 1, r.w, rowH - 2,
            { color: col, lw: act ? 1.5 : 1, alpha: act ? 0.55 + 0.45 * pulse : 0.42, scale: 0.85 * rel });
          if (act && pulse > 0.05) ctx.restore();
          U.text(ctx, (c.k ? `k=${c.k} · ` : '') + `${Math.round(c.f)} Hz · a=${frNum(c.a.toFixed(2))}`,
            r.x + 3, rcy + 3, { size: 9, color: col, mono: true });

          /* lien vers SA barre */
          if (c.bin < SHOW) {
            const bx = specR.x + (c.bin + 0.5) * bw;
            const by = specR.y + specR.h - 1 - U.clamp(magS[c.bin], 0, 1) * (specR.h - 5);
            ctx.save();
            ctx.strokeStyle = col; ctx.lineWidth = 1;
            ctx.globalAlpha = (act ? 0.25 + 0.5 * pulse : 0.12) * intro;
            ctx.setLineDash([3, 5]);
            ctx.beginPath(); ctx.moveTo(r.x + r.w - 2, rcy); ctx.lineTo(bx, by); ctx.stroke();
            ctx.restore();
            if (act && pulse > 0.05) U.glowDot(ctx, r.x + r.w - 2, rcy, 1.5 + 1.5 * pulse, col);
          }
        }
        if (nShown < comps.length) U.text(ctx, `… +${comps.length - nShown} harmoniques`,
          r.x + r.w - 4, r.y + r.h - 4, { size: 9, color: palette.faint, align: 'right' });
      }

      /* ================== boucle de rendu ================== */
      stage.onFrame((t, dt) => {
        const W = stage.W, H = stage.H;
        stage.clear();

        const preset = selSig.value;
        const nH = Math.round(slHarm.value);
        const hannOn = tgHann.value;
        const key = preset + '|' + nH;
        if (key !== compKey) { compKey = key; rebuildComps(preset, nH); }

        /* -- ré-échantillonnage continu : la trame démarre au temps t (le signal vit) -- */
        let ampNow = 0;
        if (preset === 'Pseudo-parole') {
          for (let i = 0; i < N; i++) {
            const v = U.gen.speech(t + i / FS) * 1.5;
            buf[i] = v;
            const a = Math.abs(v); if (a > ampNow) ampNow = a;
          }
        } else {
          for (let i = 0; i < N; i++) {
            const tt = t + i / FS;
            let v = 0;
            for (let c = 0; c < comps.length; c++) v += Math.sin(TAU * comps[c].f * tt) * comps[c].a;
            buf[i] = v;
            const a = Math.abs(v); if (a > ampNow) ampNow = a;
          }
        }
        if (preset === 'Carré') {
          for (let i = 0; i < N; i++) sqBuf[i] = Math.sin(TAU * SQ_F0 * (t + i / FS)) >= 0 ? SQ_A : -SQ_A;
        }
        if (hannOn) for (let i = 0; i < N; i++) wbuf[i] = buf[i] * hannW[i];

        /* -- spectre VRAI, lissé pour la fluidité (snap si en pause : dt = 0) -- */
        const mag = U.rfftMag(buf, hannOn);
        const gain = (preset === 'Pseudo-parole' ? 2.4 : 1) * (hannOn ? 1 : 0.5);
        const kS = dt > 0 ? 1 - Math.exp(-dt * 14) : 1;
        for (let i = 0; i < N / 2; i++) magS[i] += (mag[i] * gain - magS[i]) * kS;
        let peakBin = 1;
        for (let i = 2; i < SHOW; i++) if (magS[i] > magS[peakBin]) peakBin = i;

        const showComps = tgDec.value && comps.length > 0;
        const activeIdx = showComps ? Math.floor(t / CYC) % comps.length : -1;
        const pulse = U.smoothstep(Math.sin(Math.PI * ((t % CYC) / CYC)));
        const intro = U.ease(U.clamp(t / 0.8, 0, 1));
        const mainColor = preset === 'Pseudo-parole' ? palette.voice : palette.blue;

        /* -- mise en page, recalculée depuis W/H à chaque frame -- */
        const narrow = W < 560;
        const pad = narrow ? 8 : 12;
        const top = pad + (narrow ? 16 : 20);
        const bottom = H - pad;
        const LBL = 14;

        U.text(ctx, `fs = 8 kHz · N = ${N} · Δf = fs/N = ${frNum(DF)} Hz · trame = ${Math.round(DUR * 1000)} ms`,
          pad, pad + 9, { size: narrow ? 9 : 11, color: palette.dim, mono: true });
        if (W > 430) U.text(ctx, `pic ≈ ${U.fmt.hz(peakBin * DF)}`, W - pad, pad + 9,
          { size: narrow ? 9 : 11, color: mainColor, align: 'right', mono: true, bold: true });

        let waveR, specR, compR = null, rowH = 0, nShown = 0;
        if (!narrow) {
          const leftX = pad, leftW = Math.round((W - 3 * pad) * 0.55);
          const rightX = leftX + leftW + pad, rightW = W - rightX - pad;
          const cTop = top + LBL, leftH = bottom - cTop;
          specR = { x: rightX, y: cTop, w: rightW, h: Math.max(40, leftH - 16) };
          waveR = { x: leftX, y: cTop, w: leftW, h: leftH };
          if (showComps) {
            const mainH = Math.max(56, Math.min(leftH * 0.44, leftH - 64));
            const compTop = cTop + mainH + LBL + 2;
            const compH = bottom - compTop;
            nShown = Math.max(0, Math.min(comps.length, Math.floor(compH / 17)));
            if (nShown > 0) {
              compR = { x: leftX, y: compTop, w: leftW, h: compH };
              rowH = compH / nShown;
              waveR.h = mainH;
              U.text(ctx, 'composantes sinusoïdales', leftX + 2, compTop - 5, { size: 10, color: palette.dim });
            }
          }
        } else {
          const x = pad, w = W - 2 * pad;
          const avail = bottom - top;
          const wantComp = showComps && avail > 200;
          let y = top + LBL;
          waveR = { x, y, w, h: Math.max(40, Math.round(avail * (wantComp ? 0.30 : 0.42)) - LBL) };
          y += waveR.h + 6;
          if (wantComp) {
            const compH = Math.round(avail * 0.24);
            nShown = Math.max(0, Math.min(comps.length, Math.floor(compH / 15)));
            if (nShown > 0) { compR = { x, y, w, h: compH }; rowH = compH / nShown; y += compH + 6; }
          }
          y += LBL;
          specR = { x, y, w, h: Math.max(40, bottom - 16 - y) };
        }

        drawWavePanel(waveR, t, preset, hannOn, mainColor, ampNow, nH);
        drawSpectrum(specR, mainColor, activeIdx, pulse, showComps, hannOn);
        if (compR) drawComps(compR, rowH, nShown, t, activeIdx, pulse, specR, intro);
      });
    },
  });
})();
