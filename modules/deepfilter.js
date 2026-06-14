/* ============================================================
   Audio AI Atlas — « Deep Filtering — au-delà du gain par bin »
   Grille STFT zoomée de phasors : gain scalaire par bin (impasse
   sur les transitoires) vs filtre FIR complexe multi-frames
   (l'idée de DeepFilterNet). Vanilla JS, IIFE, zéro dépendance.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;
  const TAU = U.TAU;

  /* ---------- Modèle du spectrogramme (déterministe en (frame k, bin b)) ---------- */
  const FRAMES = 12, BINS = 10;
  const HOP_VIS = 0.5;                 // secondes écran par hop (réel : 10 ms)
  const BIN_HZ = 960;                  // 10 lignes => 0..9,6 kHz (vue zoomée)
  const DF_BIN_HZ = 50;                // résolution STFT réelle (48 kHz, hop 480)
  const FULL_BINS = 24000 / DF_BIN_HZ; // 480 bins jusqu'à Nyquist
  const FPS_STFT = 100;                // frames réelles / seconde (hop 10 ms)
  const HARM = [1, 3, 5];              // bins porteurs (harmoniques de la voix)
  const HARM_MAG = [0.85, 0.66, 0.5];
  const PLO_T = 8;                     // période de la plosive : 8 hops ≈ 4 s
  const rotB = (b) => 0.55 + 0.47 * b; // rad/hop : rotation de phase propre au bin

  /* Contenu complexe d'une cellule — voix cohérente + plosive à phases hash. */
  function cellComp(k, b, o) {
    let vR = 0, vI = 0;
    const hi = HARM.indexOf(b);
    o.harm = hi >= 0;
    if (o.harm) {
      const m = HARM_MAG[hi] * (0.82 + 0.18 * U.noise1(k * 0.21 + b * 5.7));
      const ph = rotB(b) * k;          // phase qui tourne à vitesse constante
      vR = m * Math.cos(ph); vI = m * Math.sin(ph);
    }
    const nm = 0.05 + 0.05 * U.hash(k * 12.9 + b * 31.7);   // plancher de bruit
    const na = U.hash(k * 7.3 + b * 13.1) * TAU;
    vR += nm * Math.cos(na); vI += nm * Math.sin(na);
    let pR = 0, pI = 0;
    const pos = ((k % PLO_T) + PLO_T) % PLO_T;
    o.plo = pos <= 1;                  // burst sur 2 frames (attaque + décroissance)
    if (o.plo) {
      const pm = (pos === 0 ? 0.95 : 0.4) * (0.72 + 0.28 * U.hash(k * 3.1 + b * 7.7));
      const pa = U.hash(k * 17.3 + b * 5.9) * TAU;          // phases hash, incohérentes
      pR = pm * Math.cos(pa); pI = pm * Math.sin(pa);
    }
    o.vRe = vR; o.vIm = vI; o.pRe = pR; o.pIm = pI;
    o.re = vR + pR; o.im = vI + pR * 0 + pI;
    o.mag = Math.hypot(o.re, o.im);
  }

  /* Sortie deep filtering : y(k,b) = Σₙ wₙ·X(k−n,b), wₙ = e^{jρn}/N (aligne la voix). */
  const _cs = { re: 0, im: 0, vRe: 0, vIm: 0, pRe: 0, pIm: 0, mag: 0, plo: false, harm: false };
  function dfOut(k, b, N, o) {
    let yR = 0, yI = 0, pR = 0, pI = 0, maxP = 0;
    const rho = rotB(b);
    for (let n = 0; n < N; n++) {
      cellComp(k - n, b, _cs);
      const wR = Math.cos(rho * n) / N, wI = Math.sin(rho * n) / N;
      yR += _cs.re * wR - _cs.im * wI;
      yI += _cs.re * wI + _cs.im * wR;
      pR += _cs.pRe * wR - _cs.pIm * wI;     // part plosive de la sortie (mesure vraie)
      pI += _cs.pRe * wI + _cs.pIm * wR;
      const pm = Math.hypot(_cs.pRe, _cs.pIm);
      if (pm > maxP) maxP = pm;
    }
    o.re = yR; o.im = yI; o.mag = Math.hypot(yR, yI);
    o.pMag = Math.hypot(pR, pI); o.maxP = maxP;
  }

  AtlasRegister({
    id: 'deepfilter',
    title: 'Deep Filtering — au-delà du gain par bin',
    category: 'archi',
    icon: '⧉',
    summary: 'Pourquoi un gain par frame ne peut pas suivre une plosive — et comment un FIR complexe sur N frames y parvient.',
    explain: `
      <p>Un débruiteur fréquentiel classique prédit un <strong>masque de gains</strong> : un scalaire 0–1 par
      case temps-fréquence, multiplié au spectrogramme. Or une frame STFT typique couvre une
      <strong>fenêtre de 20 ms</strong> avancée par <strong>hops de 10 ms</strong> : le gain est constant sur
      toute la fenêtre. Une <strong>plosive</strong> (/p/, /t/, /k/) est un burst large bande de 10–25 ms —
      l'échelle d'une frame entière. Le gain ne peut donc physiquement pas agir <em>à l'intérieur</em> de la
      frame : gain haut, le burst fuit ; gain bas, on ampute aussi la voix présente dans la même case.
      Aucun scalaire ne fait les deux à la fois.</p>
      <p>Le <strong>deep filtering</strong> remplace le scalaire par un petit filtre <strong>FIR complexe</strong>
      par bin : <code>y(k,b) = Σₙ wₙ(k,b) · X(k−n,b)</code>, combinaison de N frames passées avec des poids
      complexes (rotation de phase + échelle) prédits par le réseau. Une telle combinaison linéaire
      multi-frames peut produire des variations <em>plus rapides</em> que la cadence des frames : des poids
      alignés sur la rotation de phase de la voix (cohérente d'une frame à l'autre) la reconstruisent
      presque exactement, tandis que les phases aléatoires du burst se moyennent — résidu ≈ 1/N. Le filtre
      « suit » l'enveloppe sub-frame du transitoire, chose impossible pour un gain par frame.</p>
      <p>C'est l'idée de <strong>DeepFilterNet</strong> : des gains grossiers par <strong>bandes ERB</strong> sur
      tout le spectre (peu coûteux), plus un FIR complexe d'<strong>ordre 5</strong> sur les basses fréquences
      (≤ 4,8 kHz par défaut), là où vivent pitch et harmoniques. Le coût est linéaire :
      <code>MACs ≈ ordre × bins filtrés × 4</code> par frame (multiplication complexe = 4 réelles). La largeur
      de la bande filtrée est donc un <strong>compromis calcul/qualité</strong> — et au-dessus de la coupure,
      seuls les gains agissent : les transitoires haute fréquence (l'attaque des /t/ et /k/) sont exactement
      ce qui en souffre.</p>`,
    init(stage) {
      const ctx = stage.ctx;

      const ctlMode = stage.addSelect({
        label: 'Mode',
        options: [{ value: 'gain', label: 'Gain par bin' }, { value: 'df', label: 'Deep filtering (ordre N)' }],
        value: 'gain',
      });
      const ctlGain = stage.addSlider({ label: 'Gain appliqué sur la plosive', min: 0, max: 1, step: 0.05, value: 0.3, format: (v) => U.fmt.pct(v) });
      const ctlOrder = stage.addSlider({ label: 'Ordre du filtre N', min: 1, max: 5, step: 1, value: 5, format: (v) => 'N = ' + v });
      const ctlBand = stage.addSlider({ label: 'Bande filtrée', min: 1.92, max: 7.2, step: 0.48, value: 4.8, format: (v) => U.fmt.hz(v * 1000) });

      /* Scratch pré-alloué — rien n'est alloué dans onFrame */
      const cT = { re: 0, im: 0, vRe: 0, vIm: 0, pRe: 0, pIm: 0, mag: 0, plo: false, harm: false };
      const yT = { re: 0, im: 0, mag: 0, pMag: 0, maxP: 0 };
      const tR = new Float32Array(5), tI = new Float32Array(5), tP = new Uint8Array(5);
      const NP = 22;
      const hIn = new Float32Array(NP), hOut = new Float32Array(NP);

      function phasor(cx, cy, re, im, r, color, alpha) {
        const m = Math.min(Math.hypot(re, im), 1.15) * r;
        if (m < 2.4) {  // quasi-silence : simple point
          ctx.save(); ctx.globalAlpha = alpha * 0.8; ctx.fillStyle = color;
          ctx.beginPath(); ctx.arc(cx, cy, 1.4, 0, TAU); ctx.fill(); ctx.restore();
          return;
        }
        const a = Math.atan2(im, re);
        U.arrow(ctx, cx, cy, cx + Math.cos(a) * m, cy - Math.sin(a) * m,
          { color, lw: 1.4, head: Math.max(3, r * 0.24), alpha });
      }

      stage.onFrame((t) => {
        stage.clear();
        const W = stage.W, H = stage.H, narrow = W < 560;
        const mode = ctlMode.value, g = ctlGain.value, N = Math.round(ctlOrder.value);
        const cutHz = ctlBand.value * 1000;

        const k = Math.floor(t / HOP_VIS);       // frame courante (absolue)
        const u = t / HOP_VIS - k;               // avancement 0..1 dans le hop
        const shift = U.smoothstep(u);           // défilement easé vers la gauche

        /* ----- layout (recalculé à chaque frame) ----- */
        const axL = narrow ? 30 : 46;
        const gridX = axL + 6, gridY = narrow ? 34 : 44;
        const pw = narrow ? 0 : U.clamp(W * 0.27, 190, 260);
        const gridW = W - gridX - (narrow ? 12 : pw + 18);
        const gridH = H - gridY - (narrow ? 104 : 54);
        const cw = gridW / FRAMES, ch = gridH / BINS;
        const r = Math.min(cw, ch) * 0.42;
        const colX = (f) => gridX + (f - k + FRAMES - 1 - shift) * cw;
        const rowY = (b) => gridY + gridH - (b + 1) * ch;
        const showSpark = !narrow && H >= 400;
        const sx = gridX + gridW + 16, sy = H - 78, sw = pw - 6, sh = 50;

        /* bin observé par le DF : harmonique la plus haute sous la coupure */
        let bObs = HARM[0];
        for (let i = 0; i < HARM.length; i++) if ((HARM[i] + 0.5) * BIN_HZ < cutHz) bObs = HARM[i];
        const rho = rotB(bObs);

        /* ----- en-tête + compte à rebours plosive (calculé) ----- */
        U.text(ctx, narrow ? 'Phasors STFT : longueur = |X|, angle = phase'
          : 'Spectrogramme zoomé — un phasor par case (longueur = |X|, angle = phase)',
          12, narrow ? 16 : 20, { size: narrow ? 10 : 12, bold: true });
        const pos = ((k % PLO_T) + PLO_T) % PLO_T;
        if (!narrow) {
          if (pos <= 1) U.chip(ctx, 'PLOSIVE', W - 84, 16, { color: palette.rest });
          else U.text(ctx, `plosive dans ${((PLO_T - pos - u) * HOP_VIS).toFixed(1)} s`,
            W - 12, 20, { size: 10, color: palette.dim, align: 'right' });
        }

        /* ----- grille + contenu défilant (clippé) ----- */
        const cutY = gridY + gridH - (cutHz / BIN_HZ) * ch;
        ctx.save();
        ctx.beginPath(); ctx.rect(gridX, gridY, gridW, gridH); ctx.clip();
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        for (let b = 0; b <= BINS; b++) {
          const y = gridY + gridH - b * ch;
          ctx.beginPath(); ctx.moveTo(gridX, y); ctx.lineTo(gridX + gridW, y); ctx.stroke();
        }
        for (let f = k - FRAMES + 1; f <= k + 1; f++) {   // verticales + teinte plosive
          const x = colX(f);
          ctx.strokeStyle = palette.grid;
          ctx.beginPath(); ctx.moveTo(x, gridY); ctx.lineTo(x, gridY + gridH); ctx.stroke();
          const fp = ((f % PLO_T) + PLO_T) % PLO_T;
          if (fp <= 1) {
            ctx.fillStyle = palette.rest; ctx.globalAlpha = fp === 0 ? 0.07 : 0.04;
            ctx.fillRect(x, gridY, cw, gridH); ctx.globalAlpha = 1;
            if (fp === 0 && !narrow) U.text(ctx, 'plosive', x + cw / 2, gridY + 11, { size: 9, color: palette.rest, align: 'center' });
          }
        }
        if (mode === 'df') {                              // zones DF / gains seuls
          ctx.fillStyle = palette.voice; ctx.globalAlpha = 0.05;
          ctx.fillRect(gridX, cutY, gridW, gridY + gridH - cutY);
          ctx.fillStyle = palette.rest; ctx.globalAlpha = 0.035;
          ctx.fillRect(gridX, gridY, gridW, cutY - gridY);
          ctx.globalAlpha = 1;
        }
        const enterS = U.ease(u);                         // la colonne entrante grandit (easé)
        for (let f = k - FRAMES + 1; f <= k + 1; f++) {
          const x = colX(f);
          const sc = f === k + 1 ? enterS : 1;
          for (let b = 0; b < BINS; b++) {
            cellComp(f, b, cT);
            const cx = x + cw / 2, cy = rowY(b) + ch / 2;
            const pm = Math.hypot(cT.pRe, cT.pIm);
            let col = palette.faint;
            if (pm > 0.12 && cT.harm) col = palette.mix;       // voix + burst superposés
            else if (pm > 0.12) col = palette.rest;            // burst seul
            else if (cT.harm) col = palette.voice;             // harmonique de voix
            let re = cT.re, im = cT.im, alpha = U.clamp(0.3 + 0.7 * cT.mag, 0, 1);
            if (mode === 'gain' && cT.plo) {                   // gain scalaire × frame ENTIÈRE
              phasor(cx, cy, re * sc, im * sc, r, col, alpha * 0.22);  // fantôme original
              re *= g; im *= g;
              alpha = U.clamp(0.35 + 0.7 * Math.hypot(re, im), 0, 1);
            }
            phasor(cx, cy, re * sc, im * sc, r, col, alpha);
          }
        }
        ctx.restore();

        /* ----- curseur « frame courante » ----- */
        const cxK = colX(k);
        ctx.strokeStyle = palette.blue; ctx.globalAlpha = 0.8; ctx.lineWidth = 1.4;
        U.roundRect(ctx, cxK, gridY - 1, cw, gridH + 2, 4); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.fillStyle = palette.blue;
        ctx.beginPath();
        ctx.moveTo(cxK + cw / 2, gridY + gridH + 3);
        ctx.lineTo(cxK + cw / 2 - 4, gridY + gridH + 9);
        ctx.lineTo(cxK + cw / 2 + 4, gridY + gridH + 9);
        ctx.closePath(); ctx.fill();
        if (!narrow) U.text(ctx, 'frame courante', cxK + cw / 2, gridY + gridH + 21, { size: 10, color: palette.blue, align: 'center' });

        /* ----- axes ----- */
        U.text(ctx, 'fréq.', axL, gridY - 6, { size: 9, color: palette.faint, align: 'right' });
        U.text(ctx, '9,6 k', axL, gridY + 8, { size: 9.5, color: palette.dim, align: 'right', mono: true });
        U.text(ctx, '4,8 k', axL, gridY + gridH / 2 + 3, { size: 9.5, color: palette.dim, align: 'right', mono: true });
        U.text(ctx, '0', axL, gridY + gridH, { size: 9.5, color: palette.dim, align: 'right', mono: true });
        U.text(ctx, narrow ? 'temps → 1 col = 1 hop (10 ms)' : 'temps → · 1 colonne = 1 hop (10 ms)',
          gridX, gridY + gridH + 21, { size: 10, color: palette.dim });

        /* ----- ligne de coupure « bande filtrée » (mode DF) ----- */
        if (mode === 'df') {
          ctx.save(); ctx.strokeStyle = palette.blue; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.3;
          ctx.beginPath(); ctx.moveTo(gridX, cutY); ctx.lineTo(gridX + gridW, cutY); ctx.stroke(); ctx.restore();
          U.text(ctx, `coupure ${U.fmt.hz(cutHz)}`, gridX + gridW - 4, cutY - 5, { size: 10, color: palette.blue, align: 'right', bold: true });
          U.text(ctx, narrow ? 'gains seuls (HF)' : 'gains seuls — ici, déficit sur les transitoires HF',
            gridX + 6, gridY + 24, { size: 10, color: palette.rest });
          U.text(ctx, 'deep filtering actif', gridX + 6, cutY + 14, { size: 10, color: palette.voice });
        }

        /* ----- mode GAIN : le dilemme, chiffres vrais ----- */
        if (mode === 'gain') {
          const px0 = narrow ? gridX : gridX + gridW + 16;
          let yy = narrow ? gridY + gridH + 44 : gridY + 18;
          const gdb = 20 * Math.log10(Math.max(g, 0.001));
          const leak = g >= 0.5;
          U.chip(ctx, (leak ? 'la plosive fuit : résidu ' : 'plosive atténuée : résidu ') + U.fmt.pct(g),
            px0, yy, { color: leak ? palette.yellow : palette.green, size: narrow ? 9 : 10 });
          yy += 27;
          U.chip(ctx, (g < 0.5 ? 'voix de la même frame coupée : ' : 'voix préservée : ') + U.fmt.db(gdb),
            px0, yy, { color: g < 0.5 ? palette.yellow : palette.green, size: narrow ? 9 : 10 });
          if (!narrow) {
            yy += 30;
            U.text(ctx, 'Un seul scalaire pour toute la', px0, yy, { size: 10, color: palette.dim });
            U.text(ctx, 'frame de 20 ms : impossible de', px0, yy + 14, { size: 10, color: palette.dim });
            U.text(ctx, 'couper le burst ET garder la voix.', px0, yy + 28, { size: 10, color: palette.dim });
          }
        }

        /* ----- mode DF : sources, poids, somme tip-to-tail, métriques ----- */
        if (mode === 'df') {
          const ry = rowY(bObs);
          const hx = Math.max(colX(k - N + 1), gridX);
          ctx.save();
          ctx.strokeStyle = palette.blue; ctx.lineWidth = 1.6;
          ctx.shadowColor = palette.blue; ctx.shadowBlur = 8;
          U.roundRect(ctx, hx, ry + 1, colX(k) + cw - hx, ch - 2, 5); ctx.stroke();
          ctx.restore();
          if (!narrow) U.text(ctx, `${N} dernières frames du bin observé`, hx, ry - 4, { size: 9.5, color: palette.blue });
          if (!narrow && ch > 24) {                       // poids complexes wₙ (rotation/échelle)
            for (let n = 0; n < N; n++) {
              const wx = colX(k - n) + 7, wy = ry + 8;
              const wl = (r * 1.5) / N + 3;               // longueur ∝ |wₙ| = 1/N
              U.arrow(ctx, wx, wy, wx + Math.cos(rho * n) * wl, wy - Math.sin(rho * n) * wl,
                { color: palette.pink, lw: 1.1, head: 3, alpha: 0.9 });
            }
          }

          dfOut(k, bObs, N, yT);                          // sortie vraie au temps k
          const oc = narrow ? 56 : Math.min(pw - 12, 104, gridH * 0.55);
          const limY = showSpark ? sy - 10 : H - 60;
          const ox = narrow ? gridX : gridX + gridW + 18;
          const oy = narrow ? gridY + gridH + 30
            : U.clamp(ry + ch / 2 - oc / 2, gridY + 12, Math.max(gridY + 12, limY - oc - 62));
          if (!narrow) U.arrow(ctx, colX(k) + cw + 2, ry + ch / 2, ox - 4, oy + oc / 2,
            { color: palette.dim, dash: [3, 3], alpha: 0.7 });
          U.roundRect(ctx, ox, oy, oc, oc, 8);
          ctx.fillStyle = palette.panel; ctx.fill();
          ctx.strokeStyle = palette.voice; ctx.lineWidth = 1.3; ctx.globalAlpha = 0.8; ctx.stroke(); ctx.globalAlpha = 1;
          U.text(ctx, 'sortie y(k)', ox + oc / 2, oy - 5, { size: 10, color: palette.voice, align: 'center', bold: true });

          let sR = 0, sI = 0;                             // termes wₙ·X(k−n)
          for (let n = 0; n < N; n++) {
            cellComp(k - n, bObs, cT);
            const wR = Math.cos(rho * n) / N, wI = Math.sin(rho * n) / N;
            tR[n] = cT.re * wR - cT.im * wI;
            tI[n] = cT.re * wI + cT.im * wR;
            tP[n] = cT.plo ? 1 : 0;
            sR += tR[n]; sI += tI[n];
          }
          const s = oc * 0.4;                             // échelle px par unité de magnitude
          const ccx = ox + oc / 2, ccy = oy + oc / 2;
          let px = ccx - sR * s / 2, py = ccy + sI * s / 2;       // chaîne centrée
          const prog = u * (N + 1.5);                     // addition vectorielle easée, 1 cycle/hop
          for (let j = 0; j < N; j++) {
            const n = N - 1 - j;                          // du plus ancien au plus récent
            const a = U.ease(U.clamp(prog - j, 0, 1));
            if (a <= 0.01) break;
            const nx = px + tR[n] * s * a, ny = py - tI[n] * s * a;
            U.arrow(ctx, px, py, nx, ny, { color: tP[n] ? palette.rest : palette.voice, lw: 1.6, head: 4, alpha: 0.9 });
            if (a < 1) break;
            px = nx; py = ny;
          }
          const ra = U.ease(U.clamp(prog - N, 0, 1));     // résultante (la somme)
          if (ra > 0.02) U.arrow(ctx, ccx - sR * s / 2, ccy + sI * s / 2, ccx + sR * s / 2, ccy - sI * s / 2,
            { color: palette.mix, lw: 2.2, head: 5, alpha: ra });

          cellComp(k, bObs, cT);                          // métriques vraies
          const vMag = Math.hypot(cT.vRe, cT.vIm);
          const err = Math.hypot(yT.re - cT.vRe, yT.im - cT.vIm) / Math.max(vMag, 1e-6);
          const mx = narrow ? gridX + oc + 12 : ox;
          let my = narrow ? oy + 10 : oy + oc + 14;
          if (!narrow) { U.text(ctx, 'somme tip-to-tail des wₙ·X(k−n)', ox + oc / 2, my, { size: 9.5, color: palette.dim, align: 'center' }); my += 16; }
          U.text(ctx, `|X(k)| = ${cT.mag.toFixed(2)}  →  |y(k)| = ${yT.mag.toFixed(2)}`, mx, my, { size: 10, color: palette.text, mono: true });
          my += 15;
          if (yT.maxP > 0.05) {
            U.text(ctx, `résidu du burst : ${U.fmt.pct(yT.pMag / yT.maxP)} (≈ 1/N)`, mx, my, { size: 10, color: palette.rest, mono: true });
            my += 15;
          }
          U.text(ctx, `écart à la voix cible : ${U.fmt.pct(err)}`, mx, my, { size: 10, color: palette.voice, mono: true });
        }

        /* ----- sparkline |X| vs |y| : la sortie suit la voix à travers le burst ----- */
        if (showSpark) {
          for (let i = 0; i < NP; i++) {
            const kk = k - NP + 1 + i;
            cellComp(kk, bObs, cT);
            hIn[i] = cT.mag;
            if (mode === 'df') { dfOut(kk, bObs, N, yT); hOut[i] = yT.mag; }
            else hOut[i] = (((kk % PLO_T) + PLO_T) % PLO_T) <= 1 ? g * cT.mag : cT.mag;
          }
          U.frame(ctx, sx, sy, sw, sh, `bin observé ${U.fmt.hz((bObs + 0.5) * BIN_HZ)}`);
          ctx.save();
          ctx.beginPath(); ctx.rect(sx, sy, sw, sh); ctx.clip();
          const step = sw / (NP - 1);
          for (let pass = 0; pass < 2; pass++) {
            const data = pass === 0 ? hIn : hOut;
            ctx.strokeStyle = pass === 0 ? palette.mix : palette.voice;
            ctx.lineWidth = pass === 0 ? 1.3 : 2;
            ctx.globalAlpha = pass === 0 ? 0.6 : 1;
            ctx.beginPath();
            for (let i = 0; i < NP; i++) {
              const xx = sx + (i - u) * step + step;
              const yv = sy + sh - 3 - U.clamp(data[i] / 1.6, 0, 1) * (sh - 8);
              i === 0 ? ctx.moveTo(xx, yv) : ctx.lineTo(xx, yv);
            }
            ctx.stroke();
          }
          ctx.restore();
          U.text(ctx, 'entrée |X|', sx + 5, sy + 11, { size: 9, color: palette.mix });
          U.text(ctx, 'sortie |y|', sx + sw - 5, sy + 11, { size: 9, color: palette.voice, align: 'right' });
        }

        /* ----- readout du coût : MACs ≈ ordre × bins filtrés × 4 (complexe) ----- */
        const dfBins = Math.round(cutHz / DF_BIN_HZ);     // bins réels de 50 Hz sous la coupure
        const macs = N * dfBins * 4;
        const macsF = N * FULL_BINS * 4;
        let cost;
        if (mode === 'gain') {
          cost = narrow
            ? `gains : ${FULL_BINS} MACs/frame · DF : ${N}×${dfBins}×4 = ${U.fmt.k(macs)}`
            : `Coût gains seuls : ${FULL_BINS} bins × 1 MAC = ${FULL_BINS}/frame · le DF (ordre ${N}, ≤ ${U.fmt.hz(cutHz)}) ajouterait ${N}×${dfBins}×4 = ${U.fmt.k(macs)}MACs/frame`;
        } else {
          cost = narrow
            ? `DF : ${N}×${dfBins}×4 = ${U.fmt.k(macs)}MACs/f · full-band ×${(macsF / macs).toFixed(1)}`
            : `Coût DF ≈ ordre × bins × 4 : ${N} × ${dfBins} bins (≤ ${U.fmt.hz(cutHz)}) × 4 = ${U.fmt.k(macs)}MACs/frame = ${U.fmt.k(macs * FPS_STFT)}MAC/s · full-band (${FULL_BINS} bins) : ${U.fmt.k(macsF)} (×${(macsF / macs).toFixed(1)})`;
        }
        U.text(ctx, cost, 12, H - 8, { size: narrow ? 9 : 10, color: palette.dim, mono: true });
      });
    },
  });
})();
