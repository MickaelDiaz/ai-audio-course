/* ============================================================
   Audio AI Atlas — module « Échantillonnage & quantification »
   Panneau HAUT : signal continu + échantillons + aliasing réel + axe Nyquist.
   Panneau BAS  : quantification b bits, erreur, SNR théorique.
   Mise en page responsive : empilée et agrandie sur mobile (stage.compact),
   identique au desktop au-delà de 560 px. Le dessin est factorisé en helpers
   communs ; seules les positions et les tailles de police varient.
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
      fois par seconde (<dfn class="term" data-term="echantillonnage">échantillonnage</dfn>), puis chaque mesure est arrondie sur un nombre fini de niveaux
      (<dfn class="term" data-term="quantization">quantification</dfn>). Le <strong><dfn class="term" data-term="nyquist">théorème de Shannon-Nyquist</dfn></strong> garantit qu'une sinusoïde de
      <dfn class="term" data-term="frequence">fréquence</dfn> <code>f</code> est parfaitement reconstructible si <code>f &lt; f_s/2</code> (la
      <strong>fréquence de Nyquist</strong>). Au-delà, les <dfn class="term" data-term="sample">échantillons</dfn> deviennent ambigus : ils coïncident
      exactement avec ceux d'une sinusoïde plus grave de fréquence <code>|f − k·f_s|</code>. C'est le
      <strong><dfn class="term" data-term="aliasing">repliement spectral</dfn></strong> (aliasing) — dans la visualisation, la courbe bleue passe
      réellement par les mêmes points que la courbe corail, et rien ne permet de les distinguer après coup.</p>
      <p>C'est pourquoi l'audio « full-band » utilise <strong>48 kHz</strong> : l'oreille humaine perçoit
      jusqu'à ~20 kHz, donc Nyquist à 24 kHz laisse une bande de transition confortable pour le
      <strong><dfn class="term" data-term="filtre-anti-repliement">filtre anti-repliement</dfn></strong> placé avant le convertisseur (le CD, à 44,1 kHz, laisse une
      marge plus serrée). À l'inverse, la parole concentre l'essentiel de son information sous 8 kHz :
      beaucoup de modèles d'IA vocale (ASR, TTS, codecs neuronaux) travaillent à <strong>16 ou
      24 kHz</strong> (<dfn class="term" data-term="sample-rate">fréquence d'échantillonnage</dfn> plus basse) pour réduire le nombre d'échantillons à traiter.</p>
      <p>La quantification sur <strong>b bits</strong> projette chaque échantillon (son <dfn class="term" data-term="amplitude">amplitude</dfn>) sur
      <code>2^b</code> niveaux espacés d'un pas <code>Δ = 2/2^b</code> (pleine échelle ±1). L'erreur
      commise reste dans ±Δ/2 et se comporte comme un <strong><dfn class="term" data-term="bruit-quantification">bruit uniforme</dfn></strong>, d'où le rapport
      signal/bruit théorique <code>SNR ≈ 6,02·b + 1,76 <dfn class="term" data-term="db">dB</dfn></code> pour une sinusoïde pleine échelle :
      <strong>chaque bit ajoute ~6 dB</strong>. Le 16 bits du CD offre ainsi ~98 dB, et les convertisseurs
      24 bits des interfaces audio repoussent le plancher de bruit sous celui de l'électronique analogique.
      En pratique on ajoute un <em><dfn class="term" data-term="dither">dither</dfn></em> pour décorréler cette erreur du signal aux faibles niveaux.</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const { clamp, lerp, smoothstep, fmt, text, frame, chip, glowDot } = U;
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

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

      /* ============================================================
         HELPERS DE DESSIN — partagés par les mises en page mobile & desktop.
         `m` = facteur d'échelle de police de base (fs(11) sur mobile, 11 sur desktop) :
         on le passe à chaque helper pour que tout le texte reste lisible.
         Tous les rectangles reçus sont déjà bornés (largeur/hauteur > 0).
         m porte aussi un drapeau `compact` pour les libellés courts/longs.
         ============================================================ */

      /* Modèle de frame : tout ce que les deux panneaux consomment. */
      let M = null;

      /* ---------- Panneau HAUT : signal continu, échantillons, aliasing ---------- */
      function drawSignalPanel(px, py, pw, ph, opt) {
        const compact = opt.compact;
        const fs1 = opt.fs1;                 // taille annotations principale
        frame(ctx, px, py, pw, ph, compact ? 'Signal continu — fenêtre 2 ms'
          : 'Signal continu & échantillons — fenêtre de 2 ms');

        // L'axe des fréquences (Nyquist) occupe une bande en bas du panneau.
        const axisH = clamp(ph * (compact ? 0.26 : 0.22), compact ? 26 : 18, compact ? 46 : 30);
        const wx = px + 6, ww = Math.max(10, pw - 12);
        const wy = py + 6, wh = Math.max(20, ph - axisH - 14);
        const midY = wy + wh / 2, sc = (wh / 2) * 0.92;
        const xOfTau = (tau) => wx + (tau / WIN) * ww;

        // ligne zéro
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(wx, midY); ctx.lineTo(wx + ww, midY); ctx.stroke();

        // tiges des échantillons (ambre, discret)
        ctx.strokeStyle = palette.rest; ctx.globalAlpha = 0.3; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < M.nS; i++) {
          const x = xOfTau(i / M.fsSm);
          ctx.moveTo(x, midY); ctx.lineTo(x, midY - sVal[i] * sc);
        }
        ctx.stroke(); ctx.globalAlpha = 1;

        // sinusoïde continue (corail), lisse
        ctx.strokeStyle = palette.voice; ctx.lineWidth = 2;
        ctx.beginPath();
        for (let xi = 0; xi <= ww; xi++) {
          const y = midY - M.sig((xi / ww) * WIN) * sc;
          xi === 0 ? ctx.moveTo(wx, y) : ctx.lineTo(wx + xi, y);
        }
        ctx.stroke();

        // sinusoïde alias reconstruite (bleue) : passe par les mêmes points
        if (M.aliasA > 0.02) {
          ctx.strokeStyle = palette.blue; ctx.lineWidth = 2; ctx.globalAlpha = M.aliasA;
          ctx.beginPath();
          for (let xi = 0; xi <= ww; xi++) {
            const y = midY - M.aliasSig((xi / ww) * WIN) * sc;
            xi === 0 ? ctx.moveTo(wx, y) : ctx.lineTo(wx + xi, y);
          }
          ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        }

        // points d'échantillonnage lumineux, pulsation douce
        const useGlow = M.nS <= 44;
        const rBase = compact ? 3.0 : 2.8;
        for (let i = 0; i < M.nS; i++) {
          const x = xOfTau(i / M.fsSm), y = midY - sVal[i] * sc;
          const r = rBase * (1 + 0.18 * Math.sin(TAU * 1.2 * M.t + i * 0.6));
          if (useGlow) glowDot(ctx, x, y, r, palette.rest);
          else { ctx.fillStyle = palette.rest; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill(); }
        }

        // lectures (valeurs vraies des contrôles)
        const lh = fs1 + (compact ? 4 : 3);
        text(ctx, `f = ${fmt.hz(M.fTrue)}`, wx + 4, wy + fs1 + 1, { size: fs1, color: palette.voice, mono: true, bold: true });
        text(ctx, `fs = ${fmt.hz(M.fsTrue)}`, wx + 4, wy + fs1 + 1 + lh, { size: fs1, color: palette.rest, mono: true });

        // chip ALIASING + fréquence alias réelle (à droite, opposé aux lectures f/fs)
        if (M.aliasA > 0.02) {
          ctx.save(); ctx.globalAlpha = M.aliasA;
          const chSize = compact ? fs(10) : 10;
          ctx.font = `600 ${chSize}px ${U.FONT}`;
          const cw = ctx.measureText('ALIASING').width + 14;
          chip(ctx, 'ALIASING', wx + ww - cw - 4, wy + fs1 + 1, { color: palette.blue, size: chSize });
          text(ctx, `alias perçu ≈ ${fmt.hz(M.faTrue)}`, wx + ww - 4, wy + fs1 + 1 + lh,
            { size: fs1, color: palette.blue, align: 'right', mono: true });
          ctx.restore();
        }

        /* ----- Axe des fréquences avec marqueur Nyquist fs/2 ----- */
        const ax = wx, aw = ww, ay = py + ph - axisH + (compact ? 10 : 6);
        const xOfF = (f) => ax + clamp(f / M.axisMax, 0, 1) * aw;
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + aw, ay); ctx.stroke();

        const xNy = xOfF(M.fsSm / 2);
        ctx.fillStyle = 'rgba(96,165,250,0.08)';     // zone de repliement
        ctx.fillRect(xNy, ay - 7, Math.max(0, ax + aw - xNy), 14);
        if (ax + aw - xNy > (compact ? 80 : 110)) {
          text(ctx, 'zone de repliement', ax + aw - 4, ay - 10, { size: compact ? fs(9.5) : 9, color: palette.blue, align: 'right' });
        }
        // graduations
        const tickStep = M.axisMax > 20000 ? 10000 : M.axisMax > 10000 ? 5000 : 2000;
        for (let f = tickStep; f < M.axisMax; f += tickStep) {
          const x = xOfF(f);
          ctx.strokeStyle = palette.grid;
          ctx.beginPath(); ctx.moveTo(x, ay - 3); ctx.lineTo(x, ay + 3); ctx.stroke();
          text(ctx, fmt.hz(f), x, ay + (compact ? fs(13) : 14), { size: compact ? fs(9) : 9, color: palette.faint, align: 'center' });
        }
        // marqueur Nyquist
        ctx.strokeStyle = palette.rest; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(xNy, ay - 8); ctx.lineTo(xNy, ay + 6); ctx.stroke();
        const nyLabel = `Nyquist fs/2 = ${fmt.hz(M.fsTrue / 2)}`;
        // place le label sans déborder à droite
        ctx.font = `${fs1 - 1}px ${U.FONT}`;
        const nyW = ctx.measureText(nyLabel).width;
        const nyX = Math.min(xNy + 5, ax + aw - nyW - 2);
        text(ctx, nyLabel, Math.max(ax + 2, nyX), ay - 10, { size: fs1 - 1, color: palette.rest });
        // position du signal sur l'axe
        const dotCol = M.aliasA > 0.5 ? palette.blue : palette.voice;
        glowDot(ctx, xOfF(M.fSm), ay, 3.4, dotCol);
        text(ctx, 'f', xOfF(M.fSm), ay + (compact ? fs(13) : 14), { size: compact ? fs(9.5) : 9, color: dotCol, align: 'center', bold: true });
      }

      /* ---------- Panneau BAS : quantification (sample & hold + erreur) ---------- */
      function drawQuantPanel(px, py, pw, ph, opt) {
        const compact = opt.compact;
        const fs1 = opt.fs1;
        frame(ctx, px, py, pw, ph,
          compact ? `Quantification — ${M.bInt} bits`
            : `Quantification sur ${M.bInt} bits — sample & hold`);

        // Bandeau d'erreur en bas du panneau ; le reste = courbe quantifiée.
        const errH = clamp(ph * (compact ? 0.30 : 0.27), compact ? 30 : 22, compact ? 70 : 56);
        // Sur mobile on RÉSERVE de la place en haut pour les lectures (texte agrandi) ;
        // sur desktop on garde le comportement d'origine (lectures superposées en haut de courbe).
        const headH = compact ? (fs1 + 6) * 2 + 4 : 0;
        const qx = px + 6, qw = Math.max(10, pw - 12);
        const qh = Math.max(20, ph - errH - headH - (compact ? 14 : 20));
        const qy = py + 6;
        const qMid = qy + headH + qh / 2, qsc = (qh / 2) * 0.9;

        // grille fine des niveaux de quantification (si lisible)
        const lvlPx = M.delta * qsc;
        if (lvlPx >= 3) {
          ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
          ctx.globalAlpha = clamp((lvlPx - 3) / 6, 0.25, 0.8);
          ctx.beginPath();
          for (let v = 0; v <= 1.0001; v += M.delta) {
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
          const y = qMid - M.sig((xi / qw) * WIN) * qsc;
          xi === 0 ? ctx.moveTo(qx, y) : ctx.lineTo(qx + xi, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;

        // marches d'escalier (sample & hold) en violet
        const xOfQ = (tau) => qx + clamp(tau / WIN, 0, 1) * qw;
        ctx.strokeStyle = palette.mix; ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < M.nS; i++) {
          const x1 = xOfQ(i / M.fsSm), x2 = xOfQ((i + 1) / M.fsSm);
          const y = qMid - qVal[i] * qsc;
          i === 0 ? ctx.moveTo(x1, y) : ctx.lineTo(x1, y);
          ctx.lineTo(x2, y);
        }
        ctx.stroke();

        // bandeau d'erreur de quantification (rouge, zoomé à ±Δ/2)
        const ey = py + ph - errH - 6, eMid = ey + errH / 2;
        const esc = (errH / 2) * 0.78 / (M.delta / 2);   // ±Δ/2 remplit le bandeau
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(qx, eMid - (M.delta / 2) * esc); ctx.lineTo(qx + qw, eMid - (M.delta / 2) * esc);
        ctx.moveTo(qx, eMid + (M.delta / 2) * esc); ctx.lineTo(qx + qw, eMid + (M.delta / 2) * esc);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeStyle = palette.red; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.9;
        ctx.beginPath();
        for (let i = 0; i < M.nS; i++) {
          const x1 = xOfQ(i / M.fsSm), x2 = xOfQ((i + 1) / M.fsSm);
          const y = eMid - (qVal[i] - sVal[i]) * esc;
          i === 0 ? ctx.moveTo(x1, y) : ctx.lineTo(x1, y);
          ctx.lineTo(x2, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
        text(ctx, compact ? 'erreur e[n], bornée à ±Δ/2'
          : 'erreur de quantification e[n] = Q(x[n]) − x[n], bornée à ±Δ/2',
          qx + 4, ey - 3, { size: fs1 - 1, color: palette.red });

        // lectures du haut (valeurs exactes)
        text(ctx, `${M.bInt} bits → ${fmt.k(M.Ldisp)} niveaux`, qx + 4, qy + fs1 + 1,
          { size: fs1, color: palette.mix, mono: true, bold: true });
        const lh = fs1 + (compact ? 5 : 3);
        text(ctx, `Δ = 2/2^${M.bInt} = ${fmtDelta(M.deltaDisp)}`, qx + 4, qy + fs1 + 1 + lh,
          { size: fs1, color: palette.dim, mono: true });
        text(ctx, `SNR ≈ 6,02·b + 1,76 = ${M.snr.toFixed(1)} dB`, qx + qw - 4, qy + fs1 + 1,
          { size: fs1, color: palette.text, align: 'right', mono: true, bold: true });
        text(ctx, 'sinusoïde pleine échelle', qx + qw - 4, qy + fs1 + 1 + lh,
          { size: fs1 - 1, color: palette.faint, align: 'right' });
      }

      stage.onFrame((t, dt) => {
        stage.clear();
        const W = stage.W, H = stage.H;
        const compact = stage.compact;

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
        const fold = Math.round(fSm / Math.max(1, fsSm));
        const fAliasSigned = fSm - fold * fsSm;   // passe par les mêmes échantillons
        const aliasSig = (tau) => AMP * Math.sin(TAU * fAliasSigned * tau + ph);
        const faTrue = Math.abs(ctlF.value - Math.round(ctlF.value / ctlFs.value) * ctlFs.value);

        const bInt = ctlBits.value;
        const L = Math.pow(2, bSm);               // niveaux (lissé pour le morphing visuel)
        const delta = 2 / Math.max(2, L);
        const quant = (v) => clamp(Math.round(v / delta) * delta, -1, 1);
        const nS = Math.max(0, Math.min(MAXS, Math.floor(WIN * fsSm) + 1));
        for (let i = 0; i < nS; i++) {
          sVal[i] = sig(i / fsSm);
          qVal[i] = quant(sVal[i]);
        }
        const Ldisp = 1 << bInt;                  // valeurs affichées : exactes
        const deltaDisp = 2 / Ldisp;
        const snr = 6.02 * bInt + 1.76;

        // partage du modèle avec les helpers
        M = {
          t, sig, aliasSig, aliasA, fSm, fsSm, axisMax, nS, delta,
          fTrue: ctlF.value, fsTrue: ctlFs.value, faTrue,
          bInt, Ldisp, deltaDisp, snr,
        };

        /* ---------- Mise en page ---------- */
        const pad = clamp(W * 0.025, 8, 14);
        const labelGap = compact ? 22 : 18;       // hauteur réservée au libellé du cadre
        const gap = compact ? 26 : 12;            // entre les deux panneaux
        const px = pad, pw = Math.max(40, W - pad * 2);

        if (compact) {
          /* ===== Mobile : panneaux empilés, texte agrandi, rien de masqué ===== */
          const titleH = fs(15) + 6;
          text(ctx, 'Échantillonnage & quantification', W / 2, fs(15), { size: fs(13.5), bold: true, align: 'center' });

          const availH = Math.max(120, H - pad - titleH - labelGap * 2 - gap);
          // on partage la hauteur ~équitablement (le bas a besoin d'un peu plus pour l'erreur)
          const topH = Math.max(90, availH * 0.50);
          const botH = Math.max(90, availH - topH);
          const topY = pad + titleH + labelGap;
          const botY = topY + topH + gap + labelGap;
          const fs1 = fs(11.5);

          drawSignalPanel(px, topY, pw, topH, { compact: true, fs1 });
          drawQuantPanel(px, botY, pw, botH, { compact: true, fs1 });
        } else {
          /* ===== Desktop / tablette : disposition d'origine, inchangée ===== */
          const availH = Math.max(60, H - pad * 2 - labelGap * 2 - gap);
          const topH = availH * 0.46, botH = availH - topH;
          const topY = pad + labelGap;
          const botY = topY + topH + gap + labelGap;
          const fs1 = 11;

          drawSignalPanel(px, topY, pw, topH, { compact: false, fs1 });
          drawQuantPanel(px, botY, pw, botH, { compact: false, fs1 });
        }
      });
    },
  });
})();
