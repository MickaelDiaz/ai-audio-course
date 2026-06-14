/* ============================================================
   Audio AI Atlas — module « Quantification — FP32 → INT8 »
   Gauche : histogramme de 2000 poids + grille de niveaux réelle,
   snap animé par vagues, histogramme d'erreur et SQNR calculés.
   Droite : taille du modèle, streaming DMA DRAM→cache, et le vrai
   goulot embarqué : la bande passante mémoire, pas les MACs.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette: P } = Atlas;

  /* ---------- Constantes du modèle ---------- */
  const NW = 2000;                      // poids de base (gaussienne)
  const MAXO = 100;                     // outliers max (5 % de 2000)
  const NBINS = 52, EBINS = 56;         // bins histogrammes (valeurs / erreur)
  const NDOTS = 220;                    // points animés (sous-échantillon)
  const DR = 3.9;                       // demi-plage affichée : là où vivent 99 % des poids
  const SIG = [1.0, 0.6, 0.9, 0.35];    // écart-type par canal (4 groupes)
  const NPARAMS = 2.6e6;                // 2.6 M paramètres
  const INF_FPS = 100;                  // inférences/s (hop de 10 ms, streaming temps réel)
  const CACHE_MB = 2;                   // cache on-chip du NPU
  const CYC = 5.2;                      // période du cycle d'animation (s)
  const BYTES = { FP32: 4, FP16: 2, INT8: 1, INT4: 0.5 };

  /* ---------- Quantification (vraie, pas décorative) ---------- */
  function ulp16(v) {                   // pas de la grille FP16 autour de v (mantisse 10 bits)
    const a = Math.abs(v);
    if (a < 6.104e-5) return Math.pow(2, -24);
    return Math.pow(2, Math.min(15, Math.floor(Math.log2(a))) - 10);
  }
  function qFp16(v) { const u = ulp16(v); return Math.round(v / u) * u; }
  function qUni(v, mn, mx, L) {         // grille uniforme à L niveaux sur [mn, mx] réels
    const s = (mx - mn) / (L - 1);
    if (s <= 0) return mn;
    return mn + Math.round((v - mn) / s) * s;
  }

  AtlasRegister({
    id: 'quantization',
    title: 'Quantification — FP32 → INT8',
    category: 'system',
    icon: '▱',
    summary: 'Pourquoi INT8 souffre des outliers, pourquoi per-channel sauve la mise — et pourquoi le vrai goulot embarqué est la bande passante mémoire.',
    explain: `
      <p><strong>Quantifier</strong>, c'est représenter chaque poids sur moins de bits : on remplace la valeur
      réelle par le niveau le plus proche d'une <strong>grille</strong>. En INT8 uniforme, 256 niveaux couvrent
      l'intervalle [min, max] des poids <em>réels</em> : le pas vaut (max − min) / 255 et l'erreur au pire
      ± pas/2. Le <strong>SQNR</strong> affiché (rapport signal / bruit de quantification) est calculé sur les
      vrais poids : 10·log₁₀(Σw² / Σe²). <strong>FP16</strong>, lui, garde une mantisse de 10 bits : sa grille
      est si dense que la perte est négligeable — c'est le <em>premier réflexe embarqué</em> : moitié du débit
      mémoire pour ~zéro perte, sans calibration.</p>
      <p>Le talon d'Achille d'INT8, ce sont les <strong>outliers</strong> : quelques poids à ±4–6σ suffisent à
      étirer [min, max], donc à grossir le pas — et les 99 % de poids concentrés au centre perdent en précision
      (regardez la grille s'espacer et l'histogramme d'erreur s'élargir). La parade : la quantification
      <strong>per-channel</strong> (ou per-group) — chaque canal reçoit son propre min/max et sa propre grille.
      Les canaux sans outliers retrouvent un pas fin et le SQNR remonte de plusieurs dB. C'est pourquoi la
      quantification post-entraînement (<strong>PTQ</strong>) exige une <strong>calibration</strong> : mesurer
      les vraies plages, canal par canal.</p>
      <p>Côté mémoire, le calcul est trivial : taille = params × octets/param. Pour 2.6 M de paramètres :
      FP32 → 10.4 Mo, FP16 → 5.2 Mo, INT8 → 2.6 Mo, INT4 → 1.3 Mo. Diviser les bits, c'est diviser les
      octets à stocker… et surtout à <em>déplacer</em>.</p>
      <p>Car voici le point que tout le monde rate : sur un NPU embarqué au petit cache (2 Mo ici), un modèle
      <em>streaming</em> qui infère 100 fois par seconde doit relire ses poids depuis la DRAM à
      <strong>chaque frame</strong> dès qu'ils ne tiennent pas en cache. Débit nécessaire = taille × 100/s :
      1.04 Go/s en FP32, 0.26 Go/s en INT8. Le facteur limitant n'est pas le calcul (MACs) mais les
      <strong>octets de poids à déplacer par frame</strong> — la bande passante DDR. Quantifier, c'est d'abord
      réduire ce débit.</p>`,

    init(stage) {
      const ctx = stage.ctx;

      /* ----- Poids synthétiques (déterministes, pré-alloués une fois) ----- */
      const w = new Float32Array(NW + MAXO);
      const chan = new Uint8Array(NW + MAXO);
      const qv = new Float32Array(NW + MAXO);
      for (let i = 0; i < NW; i++) {            // Box-Muller sur U.hash → gaussienne stable
        const c = i & 3;
        const u1 = Math.max(1e-6, U.hash(i * 2.13 + 1.7));
        const u2 = U.hash(i * 3.71 + 9.13);
        w[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(U.TAU * u2) * SIG[c];
        chan[i] = c;
      }
      for (let k = 0; k < MAXO; k++) {          // outliers à ±(4–6)σ, 80 % sur le canal 2
        const i = NW + k;
        w[i] = (U.hash(k * 7.7 + 0.31) < 0.5 ? -1 : 1) * (4 + 2 * U.hash(k * 5.1 + 2.2));
        chan[i] = U.hash(k * 9.3 + 4.4) < 0.8 ? 2 : (U.hash(k * 11.1 + 6.6) * 4) | 0;
      }

      /* ----- Buffers de stats / rendu (hors onFrame) ----- */
      const bins = [new Float32Array(NBINS), new Float32Array(NBINS), new Float32Array(NBINS), new Float32Array(NBINS)];
      const binMax = [1, 1, 1, 1];
      const ebinsN = new Float32Array(EBINS);
      const cMin = new Float32Array(4), cMax = new Float32Array(4);
      const dMin = new Float32Array(4), dMax = new Float32Array(4), stepV = new Float32Array(4);
      const outUp = new Int32Array(4), outDn = new Int32Array(4);
      const dotV = new Float32Array(NDOTS), dotQ = new Float32Array(NDOTS), dotX = new Float32Array(NDOTS);
      const dotG = new Uint8Array(NDOTS);
      const st = { count: NW, cols: 1, L: 256, prec: 'INT8', perCh: false, nOut: 0, dotN: 0,
                   sqnr: 0, sqnrT: 0, sqnrC: 0, eRange: 0, mn: 0, mx: 0 };

      /* ----- Contrôles ----- */
      const cOut = stage.addSlider({ label: 'Outliers', min: 0, max: 5, step: 0.5, value: 0, format: (v) => v + ' %' });
      const cPrec = stage.addSelect({ label: 'Précision', options: ['FP16', 'INT8', 'INT4'], value: 'INT8' });
      const cPer = stage.addToggle({ label: 'Per-channel (4 groupes)', value: false });
      let animStart = 0, lastT = 0, sig = '';
      stage.addButton({ label: 'Rejouer l’animation', onClick: () => { animStart = lastT; } });

      /* ----- Recalcul complet (uniquement quand un contrôle change) ----- */
      function recompute() {
        const nOut = Math.round(cOut.value / 100 * NW);
        const count = NW + nOut;
        const prec = cPrec.value, perCh = cPer.value;
        const L = prec === 'INT8' ? 256 : prec === 'INT4' ? 16 : 0;
        // min / max réels — globaux et par canal
        let mnA = Infinity, mxA = -Infinity;
        cMin.fill(Infinity); cMax.fill(-Infinity);
        for (let i = 0; i < count; i++) {
          const v = w[i], c = chan[i];
          if (v < cMin[c]) cMin[c] = v;
          if (v > cMax[c]) cMax[c] = v;
          if (v < mnA) mnA = v;
          if (v > mxA) mxA = v;
        }
        // quantification réelle des deux modes → SQNR comparés
        let sv = 0, seT = 0, seC = 0, eR = 1e-9;
        for (let i = 0; i < count; i++) {
          const v = w[i], c = chan[i];
          const qT = L ? qUni(v, mnA, mxA, L) : qFp16(v);
          const qC = L ? qUni(v, cMin[c], cMax[c], L) : qT;
          sv += v * v;
          seT += (qT - v) * (qT - v);
          seC += (qC - v) * (qC - v);
          const q = perCh ? qC : qT;
          qv[i] = q;
          const ae = Math.abs(q - v);
          if (ae > eR) eR = ae;
        }
        st.sqnrT = 10 * Math.log10(sv / Math.max(seT, 1e-12));
        st.sqnrC = 10 * Math.log10(sv / Math.max(seC, 1e-12));
        st.sqnr = perCh ? st.sqnrC : st.sqnrT;
        st.eRange = eR;
        // histogramme d'erreur normalisé (vraies erreurs q(w) − w)
        ebinsN.fill(0);
        for (let i = 0; i < count; i++) {
          const b = Math.min(EBINS - 1, Math.max(0, Math.floor(((qv[i] - w[i]) / eR + 1) / 2 * EBINS)));
          ebinsN[b]++;
        }
        let em = 1;
        for (let b = 0; b < EBINS; b++) if (ebinsN[b] > em) em = ebinsN[b];
        for (let b = 0; b < EBINS; b++) ebinsN[b] /= em;
        // colonnes (1 ou 4) : bornes de grille, pas, densités, outliers hors-vue
        const cols = perCh ? 4 : 1;
        for (let g = 0; g < cols; g++) {
          dMin[g] = perCh ? cMin[g] : mnA;
          dMax[g] = perCh ? cMax[g] : mxA;
          stepV[g] = L ? (dMax[g] - dMin[g]) / (L - 1) : 0;
          bins[g].fill(0); binMax[g] = 1; outUp[g] = 0; outDn[g] = 0;
        }
        for (let i = 0; i < count; i++) {
          const v = w[i], g = perCh ? chan[i] : 0;
          if (v > DR) { outUp[g]++; continue; }
          if (v < -DR) { outDn[g]++; continue; }
          bins[g][Math.min(NBINS - 1, Math.floor((v + DR) / (2 * DR) * NBINS))]++;
        }
        for (let g = 0; g < cols; g++)
          for (let b = 0; b < NBINS; b++) if (bins[g][b] > binMax[g]) binMax[g] = bins[g][b];
        // sous-échantillon de points animés
        st.dotN = Math.min(NDOTS, count);
        for (let d = 0; d < st.dotN; d++) {
          const i = Math.floor(d * count / st.dotN);
          dotV[d] = w[i]; dotQ[d] = qv[i];
          dotG[d] = perCh ? chan[i] : 0;
          dotX[d] = U.hash(i * 1.618 + 0.421);
        }
        st.count = count; st.cols = cols; st.L = L; st.prec = prec;
        st.perCh = perCh; st.nOut = nOut; st.mn = mnA; st.mx = mxA;
      }

      /* ----- Grille de niveaux d'une colonne ----- */
      function drawGrid(cx, cw, g, py, ph, vy) {
        ctx.save();
        ctx.strokeStyle = P.mix;
        if (!st.L) {                       // FP16 : fines lignes denses — quasi continu
          const top = Math.max(py + 1, vy(Math.min(dMax[g], DR)));
          const bot = Math.min(py + ph - 1, vy(Math.max(dMin[g], -DR)));
          ctx.globalAlpha = 0.16; ctx.lineWidth = 0.6;
          ctx.beginPath();
          for (let yy = top; yy <= bot; yy += 2.5) { ctx.moveTo(cx + 1, yy); ctx.lineTo(cx + cw - 1, yy); }
          ctx.stroke();
        } else {                           // INT8 / INT4 : niveaux uniformes sur [min, max] réels
          ctx.globalAlpha = st.L === 16 ? 0.55 : 0.3;
          ctx.lineWidth = st.L === 16 ? 1.2 : 0.7;
          ctx.beginPath();
          for (let k = 0; k < st.L; k++) {
            const v = dMin[g] + k * stepV[g];
            if (v < -DR || v > DR) continue;
            const yy = vy(v);
            ctx.moveTo(cx + 1, yy); ctx.lineTo(cx + cw - 1, yy);
          }
          ctx.stroke();
        }
        ctx.restore();
      }

      /* ----- Marqueur d'outliers hors de la vue (ils étirent le range) ----- */
      function edgeMark(mx, tipY, baseY, n, narrow) {
        ctx.fillStyle = P.rest; ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.moveTo(mx, tipY); ctx.lineTo(mx - 5, baseY); ctx.lineTo(mx + 5, baseY);
        ctx.closePath(); ctx.fill();
        ctx.globalAlpha = 1;
        if (!narrow) U.text(ctx, n + '× hors vue', mx + 8, (tipY + baseY) / 2 + 3, { size: 9, color: P.rest, mono: true });
      }

      /* ----- Panneau gauche : poids, grille, snap animé, erreur, SQNR ----- */
      function drawLeft(x, y, wd, ht, t, narrow) {
        if (ht < 90 || wd < 80) return;
        const cyc = ((t - animStart) % CYC + CYC) % CYC;
        const rel = cyc > CYC - 0.6 ? U.smoothstep((cyc - (CYC - 0.6)) / 0.6) : 0;
        const gp = U.ease((cyc - 0.35) / 2) * (1 - rel);   // progression globale de la vague

        U.text(ctx, 'Poids du modèle — ' + st.count + ' valeurs' + (st.nOut ? ' (dont ' + st.nOut + ' outliers)' : ''),
          x, y + 12, { size: 12, bold: true });
        if (!narrow) {
          if (!st.L) U.chip(ctx, '~0 perte', x + wd - 66, y + 9, { color: P.green });
          else U.text(ctx, st.perCh ? st.L + ' niveaux × 4 grilles' : st.L + ' niveaux sur [' + st.mn.toFixed(1) + ', ' + st.mx.toFixed(1) + '] réels',
            x + wd, y + 12, { size: 10, color: P.mix, align: 'right', mono: true });
        }

        const errH = U.clamp(ht * 0.2, 30, 80);
        const py = y + 22, ph = Math.max(36, ht - errH - 66);
        const gapC = 8, cw = (wd - (st.cols - 1) * gapC) / st.cols;
        const vy = (v) => py + (1 - (v + DR) / (2 * DR)) * ph;

        for (let g = 0; g < st.cols; g++) {
          const cx = x + g * (cw + gapC);
          ctx.strokeStyle = P.grid; ctx.lineWidth = 1;
          ctx.strokeRect(cx + 0.5, py + 0.5, cw - 1, ph - 1);
          // silhouette de densité (corail)
          ctx.fillStyle = P.voice; ctx.globalAlpha = 0.26;
          const bh = ph / NBINS;
          for (let b = 0; b < NBINS; b++) {
            const c = bins[g][b];
            if (c) ctx.fillRect(cx + 1, py + ph - (b + 1) * bh, (c / binMax[g]) * (cw - 6) * 0.92, Math.max(1, bh - 0.6));
          }
          ctx.globalAlpha = 1;
          drawGrid(cx, cw, g, py, ph, vy);
          if (st.perCh && !narrow)
            U.text(ctx, 'canal ' + g + (st.L ? '  Δ=' + stepV[g].toFixed(3) : ''), cx + 2, py + ph + 11, { size: 9, color: P.dim, mono: true });
          if (outUp[g]) edgeMark(cx + cw * 0.5, py + 2, py + 10, outUp[g], narrow || st.perCh);
          if (outDn[g]) edgeMark(cx + cw * 0.5, py + ph - 2, py + ph - 10, outDn[g], narrow || st.perCh);
        }
        if (!narrow) {
          U.text(ctx, '+' + DR, x + 3, vy(DR) + 11, { size: 9, color: P.faint, mono: true });
          U.text(ctx, '−' + DR, x + 3, vy(-DR) - 4, { size: 9, color: P.faint, mono: true });
        }
        // points : vague easée (~2 s) qui balaie chaque colonne, snap vers le niveau le plus proche
        for (let d = 0; d < st.dotN; d++) {
          const p = U.ease((cyc - 0.35 - dotX[d] * 1.15) / 0.9) * (1 - rel);
          const v = U.lerp(dotV[d], dotQ[d], p);
          if (v > DR || v < -DR) continue;
          const cx = x + dotG[d] * (cw + gapC);
          ctx.fillStyle = p > 0.5 ? P.mix : P.voice;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.arc(cx + 4 + dotX[d] * (cw - 8), vy(v), 1.7, 0, U.TAU);
          ctx.fill();
        }
        ctx.globalAlpha = 1;

        // histogramme d'erreur (bleu) — pousse au rythme de la vague
        const ey = py + ph + 16;
        const eRtxt = st.eRange < 0.01 ? st.eRange.toFixed(4) : st.eRange.toFixed(3);
        U.text(ctx, 'Erreur de quantification q(w) − w', x, ey + 9, { size: 10, color: P.dim });
        U.text(ctx, 'max ±' + eRtxt, x + wd, ey + 9, { size: 9, color: P.blue, align: 'right', mono: true });
        U.bars(ctx, ebinsN, x, ey + 13, wd, errH - 13, { color: P.blue, scale: gp, gap: 1, alpha: 0.75 });
        ctx.strokeStyle = P.grid; ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, ey + 13.5, wd - 1, errH - 14);

        // readout SQNR réel + comparaison per-tensor / per-channel
        const sy = ey + errH + 13;
        const col = st.sqnr > 40 ? P.green : st.sqnr > 22 ? P.yellow : P.red;
        U.text(ctx, 'SQNR = ' + st.sqnr.toFixed(1) + ' dB', x, sy, { size: 12, bold: true, mono: true, color: col });
        if (st.L) {
          const d = st.sqnrC - st.sqnrT, ds = (d >= 0 ? '+' : '') + d.toFixed(1);
          const cmp = st.perCh
            ? (narrow ? 'per-channel : ' + ds + ' dB'
                      : 'per-tensor ' + st.sqnrT.toFixed(1) + ' dB → per-channel ' + st.sqnrC.toFixed(1) + ' dB (' + ds + ' dB)')
            : (narrow ? '' : 'per-channel donnerait ' + st.sqnrC.toFixed(1) + ' dB');
          if (cmp) U.text(ctx, cmp, x + wd, sy, { size: 10, color: st.perCh ? P.green : P.dim, align: 'right', mono: true });
        }
      }

      /* ----- Animation DMA : DRAM → double cache (ping-pong) ----- */
      function drawDMA(x, y, wd, ht, t, selMB) {
        const TF = 1.1;                     // 1 frame d'inférence (10 ms réels) ralentie ~110×
        const fIdx = Math.floor(t / TF);
        const dw = Math.min(wd * 0.32, 110), cwd = Math.min(wd * 0.36, 128);
        const cx2 = x + wd - cwd;
        U.node(ctx, x, y + 6, dw, ht - 26, { title: 'DRAM', sub: selMB.toFixed(1) + ' Mo de poids', color: P.rest });
        const cH = (ht - 36) / 2;
        for (let k = 0; k < 2; k++)
          U.node(ctx, cx2, y + 2 + k * (cH + 8), cwd, cH, { title: 'cache ' + (k ? 'B' : 'A'), sub: CACHE_MB + ' Mo', color: P.voice, active: fIdx % 2 === k });
        // tuiles de poids streamées en continu, à chaque frame
        const x1 = x + dw + 2;
        for (let j = 0; j < 7; j++) {
          const u = t / TF + j / 7;
          const phF = u % 1, idx = Math.floor(u), k = idx % 2;
          const sy = y + 14 + (ht - 44) * U.hash(idx * 3.17 + j * 7.31);
          const ty = y + 2 + k * (cH + 8) + cH * (0.3 + 0.45 * U.hash(idx * 5.39 + j * 2.91));
          ctx.fillStyle = P.voice;
          ctx.globalAlpha = 0.9 * Math.min(1, phF * 6, (1 - phF) * 6 + 0.15);
          ctx.fillRect(U.lerp(x1, cx2 - 4, phF) - 3, U.lerp(sy, ty, phF) - 3, 6, 6);
        }
        ctx.globalAlpha = 1;
        U.text(ctx, 'streaming DMA — frame n°' + (fIdx % 1000) + ' (ralenti ≈100×)', x, y + ht - 2, { size: 9, color: P.faint, mono: true });
      }

      /* ----- Panneau droit : tailles, DMA, débit, verdict cache ----- */
      function drawRight(x, y, wd, ht, t, narrow) {
        if (ht < 70 || wd < 120) return;
        U.text(ctx, 'Le goulot embarqué — 2.6 M params', x, y + 12, { size: 12, bold: true });
        const rowH = ht < 200 ? 14 : 18;
        const bx = x + 38, bw = Math.max(40, wd - 116);
        let yy = y + 24;
        for (const name of ['FP32', 'FP16', 'INT8', 'INT4']) {
          const mb = NPARAMS * BYTES[name] / 1e6;
          const selr = name === st.prec;
          U.text(ctx, name, x, yy + rowH - 6, { size: 10, bold: selr, color: selr ? P.voice : P.dim, mono: true });
          const bwd = bw * mb / 10.4;
          ctx.fillStyle = selr ? P.voice : P.faint;
          ctx.globalAlpha = selr ? 0.9 : 0.4;
          U.roundRect(ctx, bx, yy + 2, Math.max(2, bwd), rowH - 6, 3);
          ctx.fill(); ctx.globalAlpha = 1;
          U.text(ctx, mb.toFixed(1) + ' Mo', bx + bwd + 6, yy + rowH - 6, { size: 10, color: selr ? P.text : P.dim, mono: true });
          yy += rowH;
        }
        const selMB = NPARAMS * BYTES[st.prec] / 1e6;
        if (ht >= 180) {
          U.text(ctx, 'taille = params × octets : 2.6 M × ' + BYTES[st.prec] + ' o = ' + selMB.toFixed(1) + ' Mo',
            x, yy + 12, { size: 10, color: P.dim, mono: true });
          yy += 20;
        }
        const remain = y + ht - yy - 60;
        if (remain >= 84) {
          const dh = Math.min(remain, 200);
          drawDMA(x, yy + 6, wd, dh, t, selMB);
          yy += dh + 10;
        }
        // débit réel : taille × cadence d'inférence
        const gbs = selMB * INF_FPS / 1000;
        const fits = selMB <= CACHE_MB;
        yy += 14;
        U.text(ctx, 'Débit poids : ' + selMB.toFixed(1) + ' Mo × ' + INF_FPS + ' inf./s = ' + gbs.toFixed(2) + ' Go/s',
          x, yy, { size: 11, bold: true, mono: true, color: fits ? P.green : P.rest });
        yy += 9;
        U.chip(ctx, selMB.toFixed(1) + ' Mo ' + (fits ? '≤' : '>') + ' cache ' + CACHE_MB + ' Mo',
          x, yy + 8, { color: fits ? P.green : P.rest });
        if (yy + 44 <= y + ht) {
          if (fits) {
            U.text(ctx, 'les poids tiennent en cache : chargés une seule fois,', x, yy + 30, { size: 10, color: P.green });
            U.text(ctx, 'le facteur limitant redevient le calcul (MACs)', x, yy + 43, { size: 10, color: P.green });
          } else {
            U.text(ctx, 'poids re-streamés depuis la DRAM à CHAQUE frame :', x, yy + 30, { size: 10, color: P.rest });
            U.text(ctx, 'le goulot = bande passante DDR, pas les MACs', x, yy + 43, { size: 10, color: P.rest });
          }
        }
      }

      /* ----- Boucle de rendu ----- */
      stage.onFrame((t) => {
        lastT = t;
        const s = cOut.value + '|' + cPrec.value + '|' + cPer.value;
        if (s !== sig) { sig = s; recompute(); animStart = t; }   // relance la vague à chaque changement
        stage.clear();
        const W = stage.W, H = stage.H, pad = 10;
        if (W < 560) {                                            // empilé (mobile)
          const lh = U.clamp(H * 0.56, 140, Math.max(140, H - 170));
          drawLeft(pad, pad, W - 2 * pad, lh, t, true);
          drawRight(pad, lh + 2 * pad, W - 2 * pad, H - lh - 3 * pad, t, true);
        } else {                                                  // gauche ~55 % / droite ~45 %
          const lw = (W - 3 * pad) * 0.55;
          drawLeft(pad, pad, lw, H - 2 * pad, t, false);
          drawRight(pad * 2 + lw, pad, W - 3 * pad - lw, H - 2 * pad, t, false);
        }
      });
    },
  });
})();
