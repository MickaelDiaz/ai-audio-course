/* ============================================================
   Audio AI Atlas — « Deep Filtering — au-delà du gain par bin »
   Grille STFT zoomée de phasors : gain scalaire par bin (impasse
   sur les transitoires) vs filtre FIR complexe multi-frames
   (l'idée de DeepFilterNet). Vanilla JS, IIFE, zéro dépendance.
   Mise en page responsive : empilée verticalement sur mobile
   (grille phasors en grand, puis le panneau du mode en pleine
   largeur, puis sparkline et coût), texte agrandi via stage.fs().
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
      <p>Un débruiteur fréquentiel classique prédit un <dfn class="term" data-term="masking"><strong>masque de gains</strong></dfn> : un scalaire 0–1 par
      case <dfn class="term" data-term="stft">temps-fréquence</dfn>, multiplié au <dfn class="term" data-term="spectrogram">spectrogramme</dfn>. Or une <dfn class="term" data-term="frame">frame</dfn> STFT typique couvre une
      <dfn class="term" data-term="window"><strong>fenêtre de 20 ms</strong></dfn> avancée par <dfn class="term" data-term="hop"><strong>hops de 10 ms</strong></dfn> : le gain est constant sur
      toute la fenêtre. Une <dfn class="term" data-term="plosive"><strong>plosive</strong></dfn> (/p/, /t/, /k/) est un <dfn class="term" data-term="transient">burst large bande</dfn> de 10–25 ms —
      l'échelle d'une frame entière. Le gain ne peut donc physiquement pas agir <em>à l'intérieur</em> de la
      frame : gain haut, le burst fuit ; gain bas, on ampute aussi la voix présente dans la même case.
      Aucun scalaire ne fait les deux à la fois.</p>
      <p>Le <dfn class="term" data-term="deep-filtering"><strong>deep filtering</strong></dfn> remplace le scalaire par un petit <dfn class="term" data-term="fir"><strong>filtre FIR complexe</strong></dfn>
      par <dfn class="term" data-term="fft-bin">bin</dfn> : <code>y(k,b) = Σₙ wₙ(k,b) · X(k−n,b)</code>, <dfn class="term" data-term="convolution">combinaison</dfn> de N frames passées avec des <dfn class="term" data-term="complex-weight">poids
      complexes</dfn> (rotation de <dfn class="term" data-term="phase">phase</dfn> + échelle) prédits par le réseau. Une telle combinaison linéaire
      multi-frames peut produire des variations <em>plus rapides</em> que la cadence des frames : des poids
      alignés sur la rotation de phase de la voix (cohérente d'une frame à l'autre) la reconstruisent
      presque exactement, tandis que les phases aléatoires du burst se moyennent — résidu ≈ 1/N. Le filtre
      « suit » l'enveloppe sub-frame du transitoire, chose impossible pour un gain par frame.</p>
      <p>C'est l'idée de <dfn class="term" data-term="deepfilternet"><strong>DeepFilterNet</strong></dfn> : des gains grossiers par <dfn class="term" data-term="erb"><strong>bandes ERB</strong></dfn> sur
      tout le spectre (peu coûteux), plus un FIR complexe d'<strong>ordre 5</strong> sur les basses fréquences
      (≤ 4,8 kHz par défaut), là où vivent <dfn class="term" data-term="f0">pitch</dfn> et <dfn class="term" data-term="harmonique">harmoniques</dfn>. Le coût est linéaire :
      <code>MACs ≈ ordre × bins filtrés × 4</code> par frame (<dfn class="term" data-term="mac">multiplication complexe = 4 réelles</dfn>). La largeur
      de la bande filtrée est donc un <strong>compromis calcul/qualité</strong> — et au-dessus de la coupure,
      seuls les gains agissent : les transitoires haute fréquence (l'attaque des /t/ et /k/) sont exactement
      ce qui en souffre.</p>`,
    init(stage) {
      const ctx = stage.ctx;
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

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

      /* ====================================================================
         Helpers de dessin partagés par les deux mises en page.
         Chaque helper reçoit sa géométrie (lay) + l'état de frame (st) ; il
         ne suppose aucun ratio d'écran. Toutes les tailles passent par fs().
         ==================================================================== */

      /* ---------- la grille STFT de phasors (élément central) ---------- */
      function drawGrid(lay, st) {
        const { gridX, gridY, gridW, gridH, cw, ch, r, colX, rowY, cutY } = lay;
        const { k, u, shift, mode, g } = st;
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
            if (fp === 0) U.text(ctx, 'plosive', x + cw / 2, gridY + fs(11), { size: fs(9), color: palette.rest, align: 'center' });
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

        /* curseur « frame courante » */
        const cxK = colX(k);
        ctx.strokeStyle = palette.blue; ctx.globalAlpha = 0.8; ctx.lineWidth = 1.4;
        U.roundRect(ctx, cxK, gridY - 1, cw, gridH + 2, 4); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.fillStyle = palette.blue;
        ctx.beginPath();
        ctx.moveTo(cxK + cw / 2, gridY + gridH + 3);
        ctx.lineTo(cxK + cw / 2 - 4, gridY + gridH + 9);
        ctx.lineTo(cxK + cw / 2 + 4, gridY + gridH + 9);
        ctx.closePath(); ctx.fill();
        U.text(ctx, 'frame courante', cxK + cw / 2, gridY + gridH + fs(20), { size: fs(9.5), color: palette.blue, align: 'center' });

        /* axes */
        const axR = gridX - lay.axGap;
        U.text(ctx, 'fréq.', axR, gridY - fs(6), { size: fs(9), color: palette.faint, align: 'right' });
        U.text(ctx, '9,6 k', axR, gridY + fs(8), { size: fs(9.5), color: palette.dim, align: 'right', mono: true });
        U.text(ctx, '4,8 k', axR, gridY + gridH / 2 + fs(3), { size: fs(9.5), color: palette.dim, align: 'right', mono: true });
        U.text(ctx, '0', axR, gridY + gridH, { size: fs(9.5), color: palette.dim, align: 'right', mono: true });
        U.text(ctx, '1 colonne = 1 hop (10 ms)',
          gridX, gridY + gridH + fs(20), { size: fs(9.5), color: palette.dim });

        /* ligne de coupure « bande filtrée » (mode DF) */
        if (mode === 'df') {
          ctx.save(); ctx.strokeStyle = palette.blue; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.3;
          ctx.beginPath(); ctx.moveTo(gridX, cutY); ctx.lineTo(gridX + gridW, cutY); ctx.stroke(); ctx.restore();
          U.text(ctx, `coupure ${U.fmt.hz(st.cutHz)}`, gridX + gridW - 4, cutY - fs(5), { size: fs(10), color: palette.blue, align: 'right', bold: true });
          U.text(ctx, 'gains seuls (HF)', gridX + 6, gridY + fs(24), { size: fs(10), color: palette.rest });
          U.text(ctx, 'deep filtering actif', gridX + 6, cutY + fs(14), { size: fs(10), color: palette.voice });
        }
      }

      /* ---------- mode GAIN : le dilemme, chiffres vrais ---------- */
      function drawGainPanel(px0, yy, st, big) {
        const g = st.g;
        const gdb = 20 * Math.log10(Math.max(g, 0.001));
        const leak = g >= 0.5;
        const sz = fs(big ? 11 : 10);
        U.chip(ctx, (leak ? 'la plosive fuit : résidu ' : 'plosive atténuée : résidu ') + U.fmt.pct(g),
          px0, yy, { color: leak ? palette.yellow : palette.green, size: sz });
        yy += fs(big ? 30 : 27);
        U.chip(ctx, (g < 0.5 ? 'voix de la même frame coupée : ' : 'voix préservée : ') + U.fmt.db(gdb),
          px0, yy, { color: g < 0.5 ? palette.yellow : palette.green, size: sz });
        yy += fs(big ? 34 : 30);
        const tsz = fs(big ? 11 : 10), lh = fs(big ? 16 : 14);
        U.text(ctx, 'Un seul scalaire pour toute la', px0, yy, { size: tsz, color: palette.dim });
        U.text(ctx, 'frame de 20 ms : impossible de', px0, yy + lh, { size: tsz, color: palette.dim });
        U.text(ctx, 'couper le burst ET garder la voix.', px0, yy + 2 * lh, { size: tsz, color: palette.dim });
      }

      /* ---------- mode DF : boîte sortie y(k) + somme tip-to-tail + métriques ----------
         (ox, oy) = coin de la boîte ; oc = côté ; metrics dessinés sous la boîte si
         metricsBelow, sinon à droite (mx, my fournis par l'appelant). big = mobile. */
      function drawDFBox(lay, st, ox, oy, oc, mx, my, big) {
        const { rho, bObs, N } = st;
        U.roundRect(ctx, ox, oy, oc, oc, 8);
        ctx.fillStyle = palette.panel; ctx.fill();
        ctx.strokeStyle = palette.voice; ctx.lineWidth = 1.3; ctx.globalAlpha = 0.8; ctx.stroke(); ctx.globalAlpha = 1;
        U.text(ctx, 'sortie y(k)', ox + oc / 2, oy - fs(5), { size: fs(10), color: palette.voice, align: 'center', bold: true });

        let sR = 0, sI = 0;                             // termes wₙ·X(k−n)
        for (let n = 0; n < N; n++) {
          cellComp(st.k - n, bObs, cT);
          const wR = Math.cos(rho * n) / N, wI = Math.sin(rho * n) / N;
          tR[n] = cT.re * wR - cT.im * wI;
          tI[n] = cT.re * wI + cT.im * wR;
          tP[n] = cT.plo ? 1 : 0;
          sR += tR[n]; sI += tI[n];
        }
        const s = oc * 0.4;                             // échelle px par unité de magnitude
        const ccx = ox + oc / 2, ccy = oy + oc / 2;
        let px = ccx - sR * s / 2, py = ccy + sI * s / 2;       // chaîne centrée
        const prog = st.u * (N + 1.5);                  // addition vectorielle easée, 1 cycle/hop
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

        dfOut(st.k, bObs, N, yT);                       // sortie vraie au temps k
        cellComp(st.k, bObs, cT);                       // métriques vraies
        const vMag = Math.hypot(cT.vRe, cT.vIm);
        const err = Math.hypot(yT.re - cT.vRe, yT.im - cT.vIm) / Math.max(vMag, 1e-6);
        const msz = fs(big ? 11 : 10), lh = fs(big ? 16 : 15);
        let yyM = my;
        /* desktop : caption centrée sur la boîte, valeurs alignées à gauche en ox.
           mobile : tout aligné à gauche en mx (à droite de la boîte). */
        const capX = big ? mx : ox + oc / 2;
        U.text(ctx, 'somme tip-to-tail des wₙ·X(k−n)', capX, yyM, { size: fs(big ? 10 : 9.5), color: palette.dim, align: big ? 'left' : 'center' });
        yyM += lh;
        U.text(ctx, `|X(k)| = ${cT.mag.toFixed(2)}  →  |y(k)| = ${yT.mag.toFixed(2)}`, mx, yyM, { size: msz, color: palette.text, mono: true });
        yyM += lh;
        if (yT.maxP > 0.05) {
          U.text(ctx, `résidu du burst : ${U.fmt.pct(yT.pMag / yT.maxP)} (≈ 1/N)`, mx, yyM, { size: msz, color: palette.rest, mono: true });
          yyM += lh;
        }
        U.text(ctx, `écart à la voix cible : ${U.fmt.pct(err)}`, mx, yyM, { size: msz, color: palette.voice, mono: true });
      }

      /* halo sur les N frames sources + poids complexes wₙ (dessiné sur la grille) */
      function drawDFSources(lay, st) {
        const { colX, rowY, cw, ch, r, gridX } = lay;
        const { bObs, N, rho } = st;
        const ry = rowY(bObs);
        const hx = Math.max(colX(st.k - N + 1), gridX);
        ctx.save();
        ctx.strokeStyle = palette.blue; ctx.lineWidth = 1.6;
        ctx.shadowColor = palette.blue; ctx.shadowBlur = 8;
        U.roundRect(ctx, hx, ry + 1, colX(st.k) + cw - hx, ch - 2, 5); ctx.stroke();
        ctx.restore();
        U.text(ctx, `${N} dernières frames du bin observé`, hx, ry - fs(4), { size: fs(9.5), color: palette.blue });
        if (ch > fs(24)) {                              // poids complexes wₙ (rotation/échelle)
          for (let n = 0; n < N; n++) {
            const wx = colX(st.k - n) + 7, wy = ry + 8;
            const wl = (r * 1.5) / N + 3;               // longueur ∝ |wₙ| = 1/N
            U.arrow(ctx, wx, wy, wx + Math.cos(rho * n) * wl, wy - Math.sin(rho * n) * wl,
              { color: palette.pink, lw: 1.1, head: 3, alpha: 0.9 });
          }
        }
      }

      /* ---------- sparkline |X| vs |y| : la sortie suit la voix à travers le burst ---------- */
      function drawSparkline(sx, sy, sw, sh, st) {
        const { bObs, N, mode, g, u, k } = st;
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
        U.text(ctx, 'entrée |X|', sx + 5, sy + fs(11), { size: fs(9), color: palette.mix });
        U.text(ctx, 'sortie |y|', sx + sw - 5, sy + fs(11), { size: fs(9), color: palette.voice, align: 'right' });
      }

      /* ---------- readout du coût : MACs ≈ ordre × bins filtrés × 4 (complexe) ---------- */
      function drawCost(st, compact) {
        const W = stage.W, H = stage.H;
        const { mode, N, cutHz } = st;
        const dfBins = Math.round(cutHz / DF_BIN_HZ);     // bins réels de 50 Hz sous la coupure
        const macs = N * dfBins * 4;
        const macsF = N * FULL_BINS * 4;
        if (compact) {
          /* mobile : deux lignes centrées, ajustées à la largeur */
          const maxW = W - 20;
          const fit = (s, size) => { ctx.font = size + 'px ' + U.MONO; const w = ctx.measureText(s).width; return w > maxW ? size * maxW / w : size; };
          let l1, l2;
          if (mode === 'gain') {
            l1 = `gains seuls : ${FULL_BINS} bins × 1 MAC/frame`;
            l2 = `DF ajouterait ${N}×${dfBins}×4 = ${U.fmt.k(macs)} MACs/frame`;
          } else {
            l1 = `DF : ${N}×${dfBins}×4 = ${U.fmt.k(macs)} MACs/frame`;
            l2 = `full-band (${FULL_BINS} bins) : ${U.fmt.k(macsF)} (×${(macsF / macs).toFixed(1)})`;
          }
          const fz1 = fit(l1, fs(10.5)), fz2 = fit(l2, fs(10.5));
          const y = H - fz2 - 7;
          U.text(ctx, l1, W / 2, y - fz2, { size: fz1, mono: true, align: 'center', color: palette.dim });
          U.text(ctx, l2, W / 2, y, { size: fz2, mono: true, align: 'center', color: palette.dim });
          return;
        }
        let cost;
        if (mode === 'gain') {
          cost = `Coût gains seuls : ${FULL_BINS} bins × 1 MAC = ${FULL_BINS}/frame · le DF (ordre ${N}, ≤ ${U.fmt.hz(cutHz)}) ajouterait ${N}×${dfBins}×4 = ${U.fmt.k(macs)}MACs/frame`;
        } else {
          cost = `Coût DF ≈ ordre × bins × 4 : ${N} × ${dfBins} bins (≤ ${U.fmt.hz(cutHz)}) × 4 = ${U.fmt.k(macs)}MACs/frame = ${U.fmt.k(macs * FPS_STFT)}MAC/s · full-band (${FULL_BINS} bins) : ${U.fmt.k(macsF)} (×${(macsF / macs).toFixed(1)})`;
        }
        U.text(ctx, cost, 12, H - 8, { size: 10, color: palette.dim, mono: true });
      }

      /* ====================================================================
         Boucle de rendu : géométrie recalculée chaque frame depuis W/H.
         Branche compact (mobile, empilée) vs desktop (≥ 560, inchangée).
         ==================================================================== */
      stage.onFrame((t) => {
        stage.clear();
        const W = stage.W, H = stage.H;
        const compact = stage.compact;
        const mode = ctlMode.value, g = ctlGain.value, N = Math.round(ctlOrder.value);
        const cutHz = ctlBand.value * 1000;

        const k = Math.floor(t / HOP_VIS);       // frame courante (absolue)
        const u = t / HOP_VIS - k;               // avancement 0..1 dans le hop
        const shift = U.smoothstep(u);           // défilement easé vers la gauche

        /* bin observé par le DF : harmonique la plus haute sous la coupure */
        let bObs = HARM[0];
        for (let i = 0; i < HARM.length; i++) if ((HARM[i] + 0.5) * BIN_HZ < cutHz) bObs = HARM[i];
        const rho = rotB(bObs);

        const pos = ((k % PLO_T) + PLO_T) % PLO_T;
        const st = { k, u, shift, mode, g, N, cutHz, bObs, rho };

        if (compact) {
          /* ===== Mobile : empilement vertical (profite de la hauteur) ===== */
          const mx = 12;
          /* en-tête */
          U.text(ctx, 'Phasors STFT — longueur = |X|, angle = phase',
            W / 2, fs(16), { size: fs(12.5), bold: true, align: 'center' });
          if (pos <= 1) U.chip(ctx, 'PLOSIVE', W / 2 - fs(34), fs(33), { color: palette.rest, size: fs(10) });
          else U.text(ctx, `plosive dans ${((PLO_T - pos - u) * HOP_VIS).toFixed(1)} s`,
            W / 2, fs(36), { size: fs(10), color: palette.dim, align: 'center' });

          /* ----- géométrie de la grille (haute, pleine largeur) ----- */
          const axL = fs(34);
          const gridX = mx + axL, gridY = fs(48);
          const gridW = Math.max(40, W - gridX - mx);
          /* hauteur de grille : laisse la place sous elle au panneau du mode,
             à la sparkline et au coût (tout exprimé en fs pour rester lisible). */
          const belowGrid = fs(28);                       // axes + curseur sous la grille
          const panelH = mode === 'df' ? fs(132) : fs(118);
          const sparkH = fs(46);
          const costH = fs(34);
          const gridH = Math.max(120, H - gridY - belowGrid - panelH - sparkH - costH - fs(22));
          const cw = gridW / FRAMES, ch = gridH / BINS;
          const r = Math.max(2, Math.min(cw, ch) * 0.42);
          const colX = (f) => gridX + (f - k + FRAMES - 1 - shift) * cw;
          const rowY = (b) => gridY + gridH - (b + 1) * ch;
          const cutY = gridY + gridH - U.clamp(cutHz / BIN_HZ, 0, BINS) * ch;
          const lay = { gridX, gridY, gridW, gridH, cw, ch, r, colX, rowY, cutY, axGap: 6 };

          drawGrid(lay, st);
          if (mode === 'df') drawDFSources(lay, st);

          /* ----- panneau du mode, pleine largeur, EN GRAND ----- */
          const panelY = gridY + gridH + belowGrid + fs(14);
          if (mode === 'gain') {
            drawGainPanel(mx, panelY + fs(8), st, true);
          } else {
            /* boîte sortie y(k) à gauche, métriques à droite */
            const oc = U.clamp(Math.min(W - 2 * mx - fs(150), panelH - fs(8)), fs(72), fs(120));
            const ox = mx, oy = panelY + fs(6);
            const mxT = ox + oc + fs(14), myT = oy + fs(8);
            drawDFBox(lay, st, ox, oy, oc, mxT, myT, true);
          }

          /* ----- sparkline pleine largeur ----- */
          const sy = panelY + panelH + fs(8);
          drawSparkline(mx, sy, W - 2 * mx, sparkH, st);

          /* ----- coût ----- */
          drawCost(st, true);
        } else {
          /* ===== Desktop / tablette : mise en page horizontale (inchangée) ===== */
          const axL = 46;
          const gridX = axL + 6, gridY = 44;
          const pw = U.clamp(W * 0.27, 190, 260);
          const gridW = W - gridX - (pw + 18);
          const gridH = H - gridY - 54;
          const cw = gridW / FRAMES, ch = gridH / BINS;
          const r = Math.min(cw, ch) * 0.42;
          const colX = (f) => gridX + (f - k + FRAMES - 1 - shift) * cw;
          const rowY = (b) => gridY + gridH - (b + 1) * ch;
          const cutY = gridY + gridH - (cutHz / BIN_HZ) * ch;
          const showSpark = H >= 400;
          const sx = gridX + gridW + 16, sy = H - 78, sw = pw - 6, sh = 50;
          const lay = { gridX, gridY, gridW, gridH, cw, ch, r, colX, rowY, cutY, axGap: 0 };

          /* en-tête + compte à rebours plosive */
          U.text(ctx, 'Spectrogramme zoomé — un phasor par case (longueur = |X|, angle = phase)',
            12, 20, { size: 12, bold: true });
          if (pos <= 1) U.chip(ctx, 'PLOSIVE', W - 84, 16, { color: palette.rest });
          else U.text(ctx, `plosive dans ${((PLO_T - pos - u) * HOP_VIS).toFixed(1)} s`,
            W - 12, 20, { size: 10, color: palette.dim, align: 'right' });

          drawGrid(lay, st);

          if (mode === 'gain') {
            drawGainPanel(gridX + gridW + 16, gridY + 18, st, false);
          }

          if (mode === 'df') {
            drawDFSources(lay, st);
            const ry = rowY(bObs);
            const oc = Math.min(pw - 12, 104, gridH * 0.55);
            const limY = showSpark ? sy - 10 : H - 60;
            const ox = gridX + gridW + 18;
            const oy = U.clamp(ry + ch / 2 - oc / 2, gridY + 12, Math.max(gridY + 12, limY - oc - 62));
            U.arrow(ctx, colX(k) + cw + 2, ry + ch / 2, ox - 4, oy + oc / 2,
              { color: palette.dim, dash: [3, 3], alpha: 0.7 });
            drawDFBox(lay, st, ox, oy, oc, ox, oy + oc + 14, false);
          }

          if (showSpark) drawSparkline(sx, sy, sw, sh, st);
          drawCost(st, false);
        }
      });
    },
  });
})();
