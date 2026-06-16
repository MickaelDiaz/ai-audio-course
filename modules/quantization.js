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
      <p><dfn class="term" data-term="quantization">Quantifier</dfn>, c'est représenter chaque <dfn class="term" data-term="parameters">poids</dfn>
      sur moins de bits : on remplace la valeur réelle par le niveau le plus proche d'une
      <dfn class="term" data-term="quant-grid">grille</dfn>. En <dfn class="term" data-term="int8">INT8</dfn> uniforme, 256 niveaux
      couvrent l'intervalle [min, max] des poids <em>réels</em> : le <dfn class="term" data-term="quant-step">pas</dfn> vaut
      (max − min) / 255 et l'erreur au pire ± pas/2. Le <dfn class="term" data-term="sqnr">SQNR</dfn> affiché (rapport signal /
      bruit de quantification) est calculé sur les vrais poids : 10·log₁₀(Σw² / Σe²).
      <dfn class="term" data-term="fp16">FP16</dfn>, lui, garde une mantisse de 10 bits : sa grille est si dense que la perte
      est négligeable — c'est le <em>premier réflexe embarqué</em> : moitié du débit mémoire pour ~zéro perte, sans
      calibration.</p>
      <p>Le talon d'Achille d'INT8, ce sont les <dfn class="term" data-term="outlier">outliers</dfn> : quelques poids à ±4–6σ
      suffisent à étirer [min, max], donc à grossir le pas — et les 99 % de poids concentrés au centre perdent en
      précision (regardez la grille s'espacer et l'histogramme d'erreur s'élargir). La parade : la quantification
      <dfn class="term" data-term="per-channel">per-channel</dfn> (ou per-group) — chaque <dfn class="term" data-term="channel">canal</dfn>
      reçoit son propre min/max et sa propre grille. Les canaux sans outliers retrouvent un pas fin et le SQNR remonte
      de plusieurs dB. C'est pourquoi la quantification post-entraînement (<dfn class="term" data-term="ptq">PTQ</dfn>) exige une
      <dfn class="term" data-term="calibration">calibration</dfn> : mesurer les vraies plages, canal par canal.</p>
      <p>Côté mémoire, le calcul est trivial : taille = params × octets/param. Pour 2.6 M de paramètres :
      <dfn class="term" data-term="fp32">FP32</dfn> → 10.4 Mo, FP16 → 5.2 Mo, INT8 → 2.6 Mo,
      <dfn class="term" data-term="int4">INT4</dfn> → 1.3 Mo. Diviser les bits, c'est diviser les octets à stocker… et
      surtout à <em>déplacer</em>.</p>
      <p>Car voici le point que tout le monde rate : sur un <dfn class="term" data-term="npu">NPU</dfn> embarqué au petit
      <dfn class="term" data-term="cache">cache</dfn> (2 Mo ici), un modèle <dfn class="term" data-term="streaming">streaming</dfn>
      qui infère 100 fois par seconde doit relire ses poids depuis la <dfn class="term" data-term="dram">DRAM</dfn> à
      <strong>chaque <dfn class="term" data-term="frame">frame</dfn></strong> dès qu'ils ne tiennent pas en cache. Débit
      nécessaire = taille × 100/s : 1.04 Go/s en FP32, 0.26 Go/s en INT8. Le facteur limitant n'est pas le calcul
      (<dfn class="term" data-term="mac">MACs</dfn>) mais les <strong>octets de poids à déplacer par frame</strong> — la
      <dfn class="term" data-term="memory-bandwidth">bande passante</dfn> DDR. Quantifier, c'est d'abord réduire ce
      débit.</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

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
      function edgeMark(mx, tipY, baseY, n, showLabel, fz) {
        ctx.fillStyle = P.rest; ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.moveTo(mx, tipY); ctx.lineTo(mx - 5, baseY); ctx.lineTo(mx + 5, baseY);
        ctx.closePath(); ctx.fill();
        ctx.globalAlpha = 1;
        if (showLabel) U.text(ctx, n + '× hors vue', mx + 8, (tipY + baseY) / 2 + 3, { size: fz, color: P.rest, mono: true });
      }

      /* ----- Phase d'animation de la vague (commune aux deux layouts) ----- */
      function wavePhase(t) {
        const cyc = ((t - animStart) % CYC + CYC) % CYC;
        const rel = cyc > CYC - 0.6 ? U.smoothstep((cyc - (CYC - 0.6)) / 0.6) : 0;
        const gp = U.ease((cyc - 0.35) / 2) * (1 - rel);   // progression globale de la vague
        return { cyc, rel, gp };
      }

      /* ----- Bloc « histogramme de poids + grille + snap animé » -----
         Dessine les st.cols colonnes dans [x, x+wd] × [py, py+ph].
         Toutes les tailles de texte passent par fz(). */
      function drawHist(x, py, wd, ph, wave, opt) {
        const fz = opt.fz, showAxis = opt.showAxis, showChan = opt.showChan;
        const gapC = opt.gapC != null ? opt.gapC : 8;
        const cols = Math.max(1, st.cols);
        const cw = Math.max(8, (wd - (cols - 1) * gapC) / cols);
        const vy = (v) => py + (1 - (v + DR) / (2 * DR)) * ph;
        const dotR = opt.dotR || 1.7;

        for (let g = 0; g < cols; g++) {
          const cx = x + g * (cw + gapC);
          ctx.strokeStyle = P.grid; ctx.lineWidth = 1;
          ctx.strokeRect(cx + 0.5, py + 0.5, Math.max(1, cw - 1), Math.max(1, ph - 1));
          // silhouette de densité (corail)
          ctx.fillStyle = P.voice; ctx.globalAlpha = 0.26;
          const bh = ph / NBINS;
          for (let b = 0; b < NBINS; b++) {
            const c = bins[g][b];
            if (c) ctx.fillRect(cx + 1, py + ph - (b + 1) * bh, (c / binMax[g]) * Math.max(1, cw - 6) * 0.92, Math.max(1, bh - 0.6));
          }
          ctx.globalAlpha = 1;
          drawGrid(cx, cw, g, py, ph, vy);
          if (st.perCh && showChan)
            U.text(ctx, 'canal ' + g + (st.L ? '  Δ=' + stepV[g].toFixed(3) : ''), cx + 2, py + ph + fz + 2, { size: fz, color: P.dim, mono: true });
          const labelEdge = showChan ? !st.perCh : showChan;  // hors-vue lisible seulement si 1 colonne large
          if (outUp[g]) edgeMark(cx + cw * 0.5, py + 2, py + 10, outUp[g], labelEdge, fz);
          if (outDn[g]) edgeMark(cx + cw * 0.5, py + ph - 2, py + ph - 10, outDn[g], labelEdge, fz);
        }
        if (showAxis) {
          U.text(ctx, '+' + DR, x + 3, vy(DR) + fz + 2, { size: fz, color: P.faint, mono: true });
          U.text(ctx, '−' + DR, x + 3, vy(-DR) - 4, { size: fz, color: P.faint, mono: true });
        }
        // points : vague easée qui balaie chaque colonne, snap vers le niveau le plus proche
        for (let d = 0; d < st.dotN; d++) {
          const p = U.ease((wave.cyc - 0.35 - dotX[d] * 1.15) / 0.9) * (1 - wave.rel);
          const v = U.lerp(dotV[d], dotQ[d], p);
          if (v > DR || v < -DR) continue;
          const g = Math.min(cols - 1, dotG[d]);
          const cx = x + g * (cw + gapC);
          ctx.fillStyle = p > 0.5 ? P.mix : P.voice;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.arc(cx + 4 + dotX[d] * Math.max(1, cw - 8), vy(v), dotR, 0, U.TAU);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      /* ----- Bloc « histogramme d'erreur q(w) − w » ----- */
      function drawErr(x, ey, wd, eh, gp, fz) {
        const eRtxt = st.eRange < 0.01 ? st.eRange.toFixed(4) : st.eRange.toFixed(3);
        U.text(ctx, 'Erreur de quantification q(w) − w', x, ey, { size: fz, color: P.dim });
        U.text(ctx, 'max ±' + eRtxt, x + wd, ey, { size: fz, color: P.blue, align: 'right', mono: true });
        const by = ey + 4, bh = Math.max(8, eh - (fz + 4));
        U.bars(ctx, ebinsN, x, by, wd, bh, { color: P.blue, scale: gp, gap: 1, alpha: 0.75 });
        ctx.strokeStyle = P.grid; ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, by + 0.5, Math.max(1, wd - 1), Math.max(1, bh));
      }

      /* ----- Readout SQNR réel + comparaison per-tensor / per-channel ----- */
      function drawSqnr(x, sy, wd, fzBig, fzSmall, stacked) {
        const col = st.sqnr > 40 ? P.green : st.sqnr > 22 ? P.yellow : P.red;
        U.text(ctx, 'SQNR = ' + st.sqnr.toFixed(1) + ' dB', x, sy, { size: fzBig, bold: true, mono: true, color: col });
        if (!st.L) return;
        const d = st.sqnrC - st.sqnrT, ds = (d >= 0 ? '+' : '') + d.toFixed(1);
        const cmp = st.perCh
          ? 'per-tensor ' + st.sqnrT.toFixed(1) + ' → per-channel ' + st.sqnrC.toFixed(1) + ' dB (' + ds + ')'
          : 'per-channel donnerait ' + st.sqnrC.toFixed(1) + ' dB (' + ds + ')';
        const cmpCol = st.perCh ? P.green : P.dim;
        if (stacked) U.text(ctx, cmp, x, sy + fzSmall + 5, { size: fzSmall, color: cmpCol, mono: true });
        else U.text(ctx, cmp, x + wd, sy, { size: fzSmall, color: cmpCol, align: 'right', mono: true });
      }

      /* ----- Animation DMA : DRAM → double cache (ping-pong) ----- */
      function drawDMA(x, y, wd, ht, t, selMB, fz) {
        const TF = 1.1;                     // 1 frame d'inférence (10 ms réels) ralentie ~110×
        const fIdx = Math.floor(t / TF);
        const dw = U.clamp(wd * 0.32, 70, 130), cwd = U.clamp(wd * 0.36, 80, 150);
        const cx2 = x + wd - cwd;
        const nodeH = Math.max(20, ht - 26);
        U.node(ctx, x, y + 6, dw, nodeH, { title: 'DRAM', sub: selMB.toFixed(1) + ' Mo de poids', color: P.rest, size: fz });
        const cH = Math.max(16, (ht - 36) / 2);
        for (let k = 0; k < 2; k++)
          U.node(ctx, cx2, y + 2 + k * (cH + 8), cwd, cH, { title: 'cache ' + (k ? 'B' : 'A'), sub: CACHE_MB + ' Mo', color: P.voice, active: fIdx % 2 === k, size: fz });
        // tuiles de poids streamées en continu, à chaque frame
        const x1 = x + dw + 2;
        for (let j = 0; j < 7; j++) {
          const u = t / TF + j / 7;
          const phF = u % 1, idx = Math.floor(u), k = idx % 2;
          const sy = y + 14 + Math.max(0, ht - 44) * U.hash(idx * 3.17 + j * 7.31);
          const ty = y + 2 + k * (cH + 8) + cH * (0.3 + 0.45 * U.hash(idx * 5.39 + j * 2.91));
          ctx.fillStyle = P.voice;
          ctx.globalAlpha = 0.9 * Math.min(1, phF * 6, (1 - phF) * 6 + 0.15);
          ctx.fillRect(U.lerp(x1, cx2 - 4, phF) - 3, U.lerp(sy, ty, phF) - 3, 6, 6);
        }
        ctx.globalAlpha = 1;
        U.text(ctx, 'streaming DMA — frame n°' + (fIdx % 1000) + ' (ralenti ≈100×)', x, y + ht - 2, { size: fz, color: P.faint, mono: true });
      }

      /* ----- Barres de tailles FP32/FP16/INT8/INT4 ----- */
      function drawSizeBars(x, y, wd, rowH, fz) {
        const bx = x + Math.max(34, fz * 3.4), bw = Math.max(40, wd - (bx - x) - fz * 5.5);
        let yy = y;
        for (const name of ['FP32', 'FP16', 'INT8', 'INT4']) {
          const mb = NPARAMS * BYTES[name] / 1e6;
          const selr = name === st.prec;
          U.text(ctx, name, x, yy + rowH - 6, { size: fz, bold: selr, color: selr ? P.voice : P.dim, mono: true });
          const bwd = bw * mb / 10.4;
          ctx.fillStyle = selr ? P.voice : P.faint;
          ctx.globalAlpha = selr ? 0.9 : 0.4;
          U.roundRect(ctx, bx, yy + 2, Math.max(2, bwd), Math.max(2, rowH - 6), 3);
          ctx.fill(); ctx.globalAlpha = 1;
          U.text(ctx, mb.toFixed(1) + ' Mo', bx + bwd + 6, yy + rowH - 6, { size: fz, color: selr ? P.text : P.dim, mono: true });
          yy += rowH;
        }
        return yy;
      }

      /* ----- Verdict débit + cache (commun) ----- */
      function drawVerdict(x, yy, selMB, fzBig, fz, maxBottom) {
        const gbs = selMB * INF_FPS / 1000;
        const fits = selMB <= CACHE_MB;
        U.text(ctx, 'Débit poids : ' + selMB.toFixed(1) + ' Mo × ' + INF_FPS + '/s = ' + gbs.toFixed(2) + ' Go/s',
          x, yy, { size: fzBig, bold: true, mono: true, color: fits ? P.green : P.rest });
        yy += 9;
        U.chip(ctx, selMB.toFixed(1) + ' Mo ' + (fits ? '≤' : '>') + ' cache ' + CACHE_MB + ' Mo',
          x, yy + 8, { color: fits ? P.green : P.rest, size: fz });
        const l1 = yy + 8 + fz + 12, l2 = l1 + fz + 3;
        if (l2 <= maxBottom) {
          if (fits) {
            U.text(ctx, 'les poids tiennent en cache : chargés une seule fois,', x, l1, { size: fz, color: P.green });
            U.text(ctx, 'le facteur limitant redevient le calcul (MACs)', x, l2, { size: fz, color: P.green });
          } else {
            U.text(ctx, 'poids re-streamés depuis la DRAM à CHAQUE frame :', x, l1, { size: fz, color: P.rest });
            U.text(ctx, 'le goulot = bande passante DDR, pas les MACs', x, l2, { size: fz, color: P.rest });
          }
        }
      }

      /* ===================================================================
         LAYOUT DESKTOP (largeur ≥ 560) — gauche ~55 % / droite ~45 %
         Quasi identique à l'historique, factorisé via les helpers ci-dessus.
         =================================================================== */
      function drawLeftDesktop(x, y, wd, ht, t) {
        if (ht < 90 || wd < 80) return;
        const wave = wavePhase(t);
        U.text(ctx, 'Poids du modèle — ' + st.count + ' valeurs' + (st.nOut ? ' (dont ' + st.nOut + ' outliers)' : ''),
          x, y + 12, { size: 12, bold: true });
        if (!st.L) U.chip(ctx, '~0 perte', x + wd - 66, y + 9, { color: P.green });
        else U.text(ctx, st.perCh ? st.L + ' niveaux × 4 grilles' : st.L + ' niveaux sur [' + st.mn.toFixed(1) + ', ' + st.mx.toFixed(1) + '] réels',
          x + wd, y + 12, { size: 10, color: P.mix, align: 'right', mono: true });

        const errH = U.clamp(ht * 0.2, 30, 80);
        const py = y + 22, ph = Math.max(36, ht - errH - 66);
        drawHist(x, py, wd, ph, wave, { fz: 9, showAxis: true, showChan: true });

        const ey = py + ph + 16;
        drawErr(x, ey + 9, wd, errH - 1, wave.gp, 10);
        drawSqnr(x, ey + errH + 13, wd, 12, 10, false);
      }

      function drawRightDesktop(x, y, wd, ht, t) {
        if (ht < 70 || wd < 120) return;
        U.text(ctx, 'Le goulot embarqué — 2.6 M params', x, y + 12, { size: 12, bold: true });
        const rowH = ht < 200 ? 14 : 18;
        let yy = drawSizeBars(x, y + 24, wd, rowH, 10);
        const selMB = NPARAMS * BYTES[st.prec] / 1e6;
        if (ht >= 180) {
          U.text(ctx, 'taille = params × octets : 2.6 M × ' + BYTES[st.prec] + ' o = ' + selMB.toFixed(1) + ' Mo',
            x, yy + 12, { size: 10, color: P.dim, mono: true });
          yy += 20;
        }
        const remain = y + ht - yy - 60;
        if (remain >= 84) {
          const dh = Math.min(remain, 200);
          drawDMA(x, yy + 6, wd, dh, t, selMB, 12);
          yy += dh + 10;
        }
        yy += 14;
        drawVerdict(x, yy, selMB, 11, 10, y + ht);
      }

      /* ===================================================================
         LAYOUT MOBILE (stage.compact) — panneaux EMPILÉS, texte agrandi.
         On profite de la hauteur : histogramme en grand, erreur + SQNR,
         barres de tailles, DMA, verdict — rien n'est masqué.
         =================================================================== */
      function drawMobile(t) {
        const W = stage.W, H = stage.H, pad = 12;
        const x = pad, wd = Math.max(40, W - 2 * pad);
        const wave = wavePhase(t);
        let y = fs(8);

        /* --- En-tête --- */
        U.text(ctx, 'Quantification — FP32 → INT8', x, y + fs(15), { size: fs(14), bold: true });
        y += fs(15) + fs(7);
        U.text(ctx, 'Poids du modèle — ' + st.count + ' valeurs' + (st.nOut ? ' (' + st.nOut + ' outliers)' : ''),
          x, y + fs(12), { size: fs(12.5), bold: true });
        if (!st.L) U.chip(ctx, '~0 perte', x + wd - fs(58), y + fs(8), { color: P.green, size: fs(10) });
        y += fs(12) + fs(4);
        U.text(ctx, st.L ? (st.perCh ? st.L + ' niveaux × 4 grilles' : st.L + ' niveaux sur [' + st.mn.toFixed(1) + ', ' + st.mx.toFixed(1) + ']')
                         : 'grille FP16 quasi continue',
          x, y + fs(11), { size: fs(11), color: P.mix, mono: true });
        y += fs(11) + fs(8);

        /* --- Hauteurs des blocs, calculées à partir de la place dispo --- */
        const bottom = H - pad;
        const gap = fs(7);
        const avail = Math.max(120, bottom - y);
        // réserves fixes (texte + marges) : erreur, SQNR, tailles, verdict
        const sqnrH = fs(13) + (st.L ? fs(11) + 5 : 0) + fs(8);
        const errH = U.clamp(avail * 0.11, fs(20), 56) + fs(11) + 6;   // label + barres
        const rowH = Math.max(fs(13), 16);
        const sizeBlockH = (fs(12) + 6) + rowH * 4 + fs(11) + fs(4);   // titre + 4 barres + formule
        const verdictH = fs(11.5) + 9 + fs(11) + 12 + 2 * (fs(11) + 3);
        const fixed = errH + sqnrH + sizeBlockH + verdictH + gap * 5;
        // l'espace restant se partage entre l'histogramme (prioritaire) et la DMA.
        // Si la place le permet, on réserve d'abord une DMA lisible (~84 px) puis
        // l'histogramme prend tout le reste ; sinon la DMA s'efface au profit de l'histo.
        const rest = Math.max(0, avail - fixed);
        let dmaH = 0, histH;
        if (rest >= 88 + 70) { dmaH = U.clamp(rest - 150, 70, 150); histH = rest - dmaH - gap; }
        else { histH = U.clamp(rest, 70, 240); }
        histH = Math.max(70, histH);
        const errBarH = errH - fs(11) - 6;

        /* --- 1. Histogramme de poids + grille + snap (en grand) --- */
        drawHist(x, y, wd, histH, wave, { fz: fs(10), showAxis: true, showChan: true, dotR: 2 });
        y += histH + gap;

        /* --- 2. Histogramme d'erreur --- */
        U.text(ctx, 'Erreur de quantification q(w) − w', x, y + fs(11), { size: fs(11), color: P.dim });
        const eRtxt = st.eRange < 0.01 ? st.eRange.toFixed(4) : st.eRange.toFixed(3);
        U.text(ctx, 'max ±' + eRtxt, x + wd, y + fs(11), { size: fs(10.5), color: P.blue, align: 'right', mono: true });
        const eby = y + fs(11) + 5;
        U.bars(ctx, ebinsN, x, eby, wd, errBarH, { color: P.blue, scale: wave.gp, gap: 1, alpha: 0.75 });
        ctx.strokeStyle = P.grid; ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, eby + 0.5, Math.max(1, wd - 1), Math.max(1, errBarH));
        y = eby + errBarH + gap;

        /* --- 3. SQNR + comparaison (empilée) --- */
        drawSqnr(x, y + fs(11), wd, fs(13), fs(11), true);
        y += sqnrH + gap;

        /* --- 4. Le goulot mémoire : tailles, DMA, verdict --- */
        U.text(ctx, 'Le goulot embarqué — 2.6 M params', x, y + fs(12), { size: fs(12.5), bold: true });
        y += fs(12) + 6;
        const selMB = NPARAMS * BYTES[st.prec] / 1e6;
        y = drawSizeBars(x, y, wd, rowH, fs(11));
        U.text(ctx, '2.6 M × ' + BYTES[st.prec] + ' o = ' + selMB.toFixed(1) + ' Mo',
          x, y + fs(11), { size: fs(10.5), color: P.dim, mono: true });
        y += fs(11) + gap;

        if (dmaH >= 64 && y + dmaH + verdictH <= bottom) {
          drawDMA(x, y, wd, dmaH, t, selMB, fs(11));
          y += dmaH + gap;
        }
        drawVerdict(x, y + fs(11), selMB, fs(11.5), fs(11), bottom);
      }

      /* ----- Boucle de rendu ----- */
      stage.onFrame((t) => {
        lastT = t;
        const s = cOut.value + '|' + cPrec.value + '|' + cPer.value;
        if (s !== sig) { sig = s; recompute(); animStart = t; }   // relance la vague à chaque changement
        stage.clear();
        const W = stage.W, H = stage.H, pad = 10;
        if (stage.compact) {                                      // empilé (mobile, portrait haut)
          drawMobile(t);
        } else {                                                  // gauche ~55 % / droite ~45 %
          const lw = (W - 3 * pad) * 0.55;
          drawLeftDesktop(pad, pad, lw, H - 2 * pad, t);
          drawRightDesktop(pad * 2 + lw, pad, W - 3 * pad - lw, H - 2 * pad, t);
        }
      });
    },
  });
})();
