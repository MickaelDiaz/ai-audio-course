/* ============================================================
   Audio AI Atlas — module « Échantillonnage & quantification »
   Panneau HAUT : signal continu + échantillons + aliasing réel.
   Panneau BAS  : quantification b bits, erreur, SNR théorique.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;
  const TAU = U.TAU;

  AtlasRegister({
    id: 'sampling',
    title: 'Échantillonnage & quantification',
    category: 'signal',
    icon: '⎍',
    summary: 'Du signal continu aux nombres : cadence fs, repliement (aliasing) et bruit de quantification.',
    explain: `
      <p>Pour entrer dans un système numérique, un signal continu est mesuré <strong>f<sub>s</sub></strong>
      fois par seconde (échantillonnage), puis chaque mesure est arrondie sur un nombre fini de niveaux
      (quantification). Le <strong>théorème de Shannon-Nyquist</strong> garantit qu'une sinusoïde de
      fréquence <code>f</code> est parfaitement reconstructible si <code>f &lt; f_s/2</code> (la
      <strong>fréquence de Nyquist</strong>). Au-delà, les échantillons deviennent ambigus : ils coïncident
      exactement avec ceux d'une sinusoïde plus grave de fréquence <code>|f − k·f_s|</code>. C'est le
      <strong>repliement spectral</strong> (aliasing) — dans la visualisation, la courbe bleue passe
      réellement par les mêmes points que la courbe corail, et rien ne permet de les distinguer après coup.</p>
      <p>C'est pourquoi l'audio « full-band » utilise <strong>48 kHz</strong> : l'oreille humaine perçoit
      jusqu'à ~20 kHz, donc Nyquist à 24 kHz laisse une bande de transition confortable pour le
      <strong>filtre anti-repliement</strong> placé avant le convertisseur (le CD, à 44,1 kHz, laisse une
      marge plus serrée). À l'inverse, la parole concentre l'essentiel de son information sous 8 kHz :
      beaucoup de modèles d'IA vocale (ASR, TTS, codecs neuronaux) travaillent à <strong>16 ou
      24 kHz</strong> pour réduire le nombre d'échantillons à traiter.</p>
      <p>La quantification sur <strong>b bits</strong> projette chaque échantillon sur
      <code>2^b</code> niveaux espacés d'un pas <code>Δ = 2/2^b</code> (pleine échelle ±1). L'erreur
      commise reste dans ±Δ/2 et se comporte comme un <strong>bruit uniforme</strong>, d'où le rapport
      signal/bruit théorique <code>SNR ≈ 6,02·b + 1,76 dB</code> pour une sinusoïde pleine échelle :
      <strong>chaque bit ajoute ~6 dB</strong>. Le 16 bits du CD offre ainsi ~98 dB, et les convertisseurs
      24 bits des interfaces audio repoussent le plancher de bruit sous celui de l'électronique analogique.
      En pratique on ajoute un <em>dither</em> pour décorréler cette erreur du signal aux faibles niveaux.</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const { clamp, lerp, smoothstep, fmt, text, frame, chip, glowDot } = U;

      /* ---------- Contrôles ---------- */
      const ctlF = stage.addSlider({
        label: 'Fréquence du signal', min: 100, max: 7800, step: 10, value: 440,
        format: fmt.hz,
      });
      const ctlFs = stage.addSlider({
        label: 'Cadence d’échantillonnage fs', min: 4000, max: 48000, step: 100, value: 16000,
        format: fmt.hz,
      });
      const ctlBits = stage.addSlider({
        label: 'Résolution (bits)', min: 2, max: 16, step: 1, value: 4,
        format: (v) => v + ' bits',
      });

      /* ---------- Constantes & état (buffers pré-alloués) ---------- */
      const WIN = 0.002;          // fenêtre visualisée : 2 ms
      const AMP = 0.85;           // amplitude du signal (pleine échelle quantif. = ±1)
      const MAXS = 128;           // max échantillons : 0.002 × 48000 = 96 (+1)
      const sVal = new Float64Array(MAXS);   // échantillons bruts
      const qVal = new Float64Array(MAXS);   // échantillons quantifiés

      let fSm = ctlF.value, fsSm = ctlFs.value, bSm = ctlBits.value;
      let aliasAmt = 0, axisMax = 9000;

      const fmtDelta = (d) => d >= 0.001 ? d.toFixed(d >= 0.01 ? 3 : 4) : d.toExponential(1);

      stage.onFrame((t, dt) => {
        stage.clear();
        const W = stage.W, H = stage.H;
        const small = W < 560;

        /* ---------- Lissage des paramètres (transitions easées, robuste si dt=0) ---------- */
        const k = 1 - Math.exp(-9 * Math.max(dt, 1 / 120));
        fSm += (ctlF.value - fSm) * k;
        fsSm += (ctlFs.value - fsSm) * k;
        bSm += (ctlBits.value - bSm) * k;
        const aliasing = ctlF.value > ctlFs.value / 2;
        aliasAmt += ((aliasing ? 1 : 0) - aliasAmt) * k;
        const aliasA = smoothstep(aliasAmt);
        axisMax += (Math.max(6000, Math.max(fSm, fsSm / 2) * 1.3) - axisMax) * k;

        /* ---------- Modèle : signal, échantillons, quantification ---------- */
        const ph = 0.9 * t;                       // phase qui avance doucement
        const sig = (tau) => AMP * Math.sin(TAU * fSm * tau + ph);
        const fold = Math.round(fSm / fsSm);
        const fAliasSigned = fSm - fold * fsSm;   // passe par les mêmes échantillons
        const aliasSig = (tau) => AMP * Math.sin(TAU * fAliasSigned * tau + ph);
        const faTrue = Math.abs(ctlF.value - Math.round(ctlF.value / ctlFs.value) * ctlFs.value);

        const bInt = ctlBits.value;
        const L = Math.pow(2, bSm);               // niveaux (lissé pour le morphing visuel)
        const delta = 2 / L;
        const quant = (v) => clamp(Math.round(v / delta) * delta, -1, 1);
        const nS = Math.min(MAXS, Math.floor(WIN * fsSm) + 1);
        for (let i = 0; i < nS; i++) {
          sVal[i] = sig(i / fsSm);
          qVal[i] = quant(sVal[i]);
        }
        const Ldisp = 1 << bInt;                  // valeurs affichées : exactes
        const deltaDisp = 2 / Ldisp;
        const snr = 6.02 * bInt + 1.76;

        /* ---------- Mise en page (recalculée chaque frame) ---------- */
        const pad = clamp(W * 0.025, 8, 14);
        const labelGap = 18, gap = 12;
        const availH = Math.max(60, H - pad * 2 - labelGap * 2 - gap);
        const topH = availH * 0.46, botH = availH - topH;
        const px = pad, pw = W - pad * 2;
        const topY = pad + labelGap;
        const botY = topY + topH + gap + labelGap;
        const fs1 = small ? 10 : 11;              // taille annotations

        /* ================= PANNEAU HAUT : signal continu & échantillons ================= */
        frame(ctx, px, topY, pw, topH, small ? 'Signal continu — 2 ms' : 'Signal continu & échantillons — fenêtre de 2 ms');
        const axisH = clamp(topH * 0.22, 18, 30);
        const wx = px + 6, ww = pw - 12;
        const wy = topY + 6, wh = topH - axisH - 14;
        const midY = wy + wh / 2, sc = (wh / 2) * 0.92;
        const xOfTau = (tau) => wx + (tau / WIN) * ww;

        // ligne zéro
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(wx, midY); ctx.lineTo(wx + ww, midY); ctx.stroke();

        // tiges des échantillons (ambre, discret)
        ctx.strokeStyle = palette.rest; ctx.globalAlpha = 0.3; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < nS; i++) {
          const x = xOfTau(i / fsSm);
          ctx.moveTo(x, midY); ctx.lineTo(x, midY - sVal[i] * sc);
        }
        ctx.stroke(); ctx.globalAlpha = 1;

        // sinusoïde continue (corail), lisse
        ctx.strokeStyle = palette.voice; ctx.lineWidth = 2;
        ctx.beginPath();
        for (let xi = 0; xi <= ww; xi++) {
          const y = midY - sig((xi / ww) * WIN) * sc;
          xi === 0 ? ctx.moveTo(wx, y) : ctx.lineTo(wx + xi, y);
        }
        ctx.stroke();

        // sinusoïde alias reconstruite (bleue) : passe par les mêmes points
        if (aliasA > 0.02) {
          ctx.strokeStyle = palette.blue; ctx.lineWidth = 2; ctx.globalAlpha = aliasA;
          ctx.beginPath();
          for (let xi = 0; xi <= ww; xi++) {
            const y = midY - aliasSig((xi / ww) * WIN) * sc;
            xi === 0 ? ctx.moveTo(wx, y) : ctx.lineTo(wx + xi, y);
          }
          ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        }

        // points d'échantillonnage lumineux, pulsation douce
        const useGlow = nS <= 44;
        for (let i = 0; i < nS; i++) {
          const x = xOfTau(i / fsSm), y = midY - sVal[i] * sc;
          const r = (small ? 2.2 : 2.8) * (1 + 0.18 * Math.sin(TAU * 1.2 * t + i * 0.6));
          if (useGlow) glowDot(ctx, x, y, r, palette.rest);
          else { ctx.fillStyle = palette.rest; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill(); }
        }

        // lectures (valeurs vraies des contrôles)
        text(ctx, `f = ${fmt.hz(ctlF.value)}`, wx + 4, wy + 12, { size: fs1, color: palette.voice, mono: true, bold: true });
        text(ctx, `fs = ${fmt.hz(ctlFs.value)}`, wx + 4, wy + 26, { size: fs1, color: palette.rest, mono: true });

        // chip ALIASING + fréquence alias réelle
        if (aliasA > 0.02) {
          ctx.font = `600 10px ${U.FONT}`;
          const cw = ctx.measureText('ALIASING').width + 14;
          ctx.save(); ctx.globalAlpha = aliasA;
          chip(ctx, 'ALIASING', wx + ww - cw - 4, wy + 12, { color: palette.blue });
          text(ctx, `alias perçu ≈ ${fmt.hz(faTrue)}`, wx + ww - 4, wy + 30,
            { size: fs1, color: palette.blue, align: 'right', mono: true });
          ctx.restore();
        }

        /* ----- Axe des fréquences avec marqueur Nyquist fs/2 ----- */
        const ax = wx, aw = ww, ay = topY + topH - axisH + 6;
        const xOfF = (f) => ax + clamp(f / axisMax, 0, 1) * aw;
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + aw, ay); ctx.stroke();

        const xNy = xOfF(fsSm / 2);
        ctx.fillStyle = 'rgba(96,165,250,0.08)';     // zone de repliement
        ctx.fillRect(xNy, ay - 7, ax + aw - xNy, 14);
        if (!small && ax + aw - xNy > 110) {
          text(ctx, 'zone de repliement', ax + aw - 4, ay - 10, { size: 9, color: palette.blue, align: 'right' });
        }
        // graduations
        const tickStep = axisMax > 20000 ? 10000 : axisMax > 10000 ? 5000 : 2000;
        for (let f = tickStep; f < axisMax; f += tickStep) {
          const x = xOfF(f);
          ctx.strokeStyle = palette.grid;
          ctx.beginPath(); ctx.moveTo(x, ay - 3); ctx.lineTo(x, ay + 3); ctx.stroke();
          if (!small) text(ctx, fmt.hz(f), x, ay + 14, { size: 9, color: palette.faint, align: 'center' });
        }
        // marqueur Nyquist
        ctx.strokeStyle = palette.rest; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(xNy, ay - 8); ctx.lineTo(xNy, ay + 6); ctx.stroke();
        text(ctx, small ? 'fs/2' : `Nyquist fs/2 = ${fmt.hz(ctlFs.value / 2)}`,
          Math.min(xNy + 5, ax + aw - (small ? 30 : 130)), ay - 10, { size: fs1 - 1, color: palette.rest });
        // position du signal sur l'axe
        const dotCol = aliasA > 0.5 ? palette.blue : palette.voice;
        glowDot(ctx, xOfF(fSm), ay, 3.4, dotCol);
        text(ctx, 'f', xOfF(fSm), ay + 14, { size: 9, color: dotCol, align: 'center', bold: true });

        /* ================= PANNEAU BAS : quantification ================= */
        frame(ctx, px, botY, pw, botH,
          small ? `Quantification — ${bInt} bits` : `Quantification sur ${bInt} bits — sample & hold`);
        const errH = clamp(botH * 0.27, 22, 56);
        const qx = px + 6, qw = pw - 12;
        const qy = botY + 6, qh = botH - errH - 20;
        const qMid = qy + qh / 2, qsc = (qh / 2) * 0.9;

        // grille fine des niveaux de quantification (si lisible)
        const lvlPx = delta * qsc;
        if (lvlPx >= 3) {
          ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
          ctx.globalAlpha = clamp((lvlPx - 3) / 6, 0.25, 0.8);
          ctx.beginPath();
          for (let v = 0; v <= 1.0001; v += delta) {
            const y1 = qMid - v * qsc, y2 = qMid + v * qsc;
            ctx.moveTo(qx, y1); ctx.lineTo(qx + qw, y1);
            if (v > 0) { ctx.moveTo(qx, y2); ctx.lineTo(qx + qw, y2); }
          }
          ctx.stroke(); ctx.globalAlpha = 1;
        }

        // signal original en filigrane (référence)
        ctx.strokeStyle = palette.voice; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.3;
        ctx.beginPath();
        for (let xi = 0; xi <= qw; xi++) {
          const y = qMid - sig((xi / qw) * WIN) * qsc;
          xi === 0 ? ctx.moveTo(qx, y) : ctx.lineTo(qx + xi, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;

        // marches d'escalier (sample & hold) en violet
        const xOfQ = (tau) => qx + clamp(tau / WIN, 0, 1) * qw;
        ctx.strokeStyle = palette.mix; ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < nS; i++) {
          const x1 = xOfQ(i / fsSm), x2 = xOfQ((i + 1) / fsSm);
          const y = qMid - qVal[i] * qsc;
          i === 0 ? ctx.moveTo(x1, y) : ctx.lineTo(x1, y);
          ctx.lineTo(x2, y);
        }
        ctx.stroke();

        // bandeau d'erreur de quantification (rouge, petit, zoomé à ±Δ/2)
        const ey = botY + botH - errH - 6, eMid = ey + errH / 2;
        const esc = (errH / 2) * 0.78 / (delta / 2);   // ±Δ/2 remplit le bandeau
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(qx, eMid - (delta / 2) * esc); ctx.lineTo(qx + qw, eMid - (delta / 2) * esc);
        ctx.moveTo(qx, eMid + (delta / 2) * esc); ctx.lineTo(qx + qw, eMid + (delta / 2) * esc);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeStyle = palette.red; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.9;
        ctx.beginPath();
        for (let i = 0; i < nS; i++) {
          const x1 = xOfQ(i / fsSm), x2 = xOfQ((i + 1) / fsSm);
          const y = eMid - (qVal[i] - sVal[i]) * esc;
          i === 0 ? ctx.moveTo(x1, y) : ctx.lineTo(x1, y);
          ctx.lineTo(x2, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
        text(ctx, small ? 'erreur e[n] (±Δ/2)' : 'erreur de quantification e[n] = Q(x[n]) − x[n], bornée à ±Δ/2',
          qx + 4, ey - 3, { size: fs1 - 1, color: palette.red });

        // lectures du bas (valeurs exactes)
        text(ctx, `${bInt} bits → ${fmt.k(Ldisp)} niveaux`, qx + 4, qy + 12,
          { size: fs1, color: palette.mix, mono: true, bold: true });
        if (!small) {
          text(ctx, `Δ = 2/2^${bInt} = ${fmtDelta(deltaDisp)}`, qx + 4, qy + 26,
            { size: fs1, color: palette.dim, mono: true });
        }
        text(ctx, `SNR ≈ 6,02·b + 1,76 = ${snr.toFixed(1)} dB`, qx + qw - 4, qy + 12,
          { size: fs1, color: palette.text, align: 'right', mono: true, bold: true });
        if (!small) {
          text(ctx, 'sinusoïde pleine échelle', qx + qw - 4, qy + 26,
            { size: fs1 - 1, color: palette.faint, align: 'right' });
        }
      });
    },
  });
})();
