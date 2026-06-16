/* ============================================================
   Audio AI Atlas — module DPRNN (dual-path pour longues séquences)
   Pliage 1D → 2D, passes RNN intra-chunk / inter-chunk alternées,
   champ réceptif et chemins de gradient en O(√T).
   Mise en page responsive : empilée verticalement sur mobile (grille de
   pliage en grand, puis paramètres et comparatif gradient pleine largeur) ;
   disposition desktop (grille | panneau, comparatif en bas) inchangée.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  const T = 48;                       // longueur de la séquence (frames)
  const KS = [4, 6, 8, 12];           // tailles de chunk proposées
  const SQT = Math.sqrt(T);           // √48 ≈ 6.93
  const DECAY = 0.93;                 // atténuation du gradient par pas (illustration)

  /* Contenu pseudo-spectral stable du bandeau (1 valeur par frame) */
  const cellC = new Array(T);
  for (let i = 0; i < T; i++) {
    const v = U.clamp(0.55 * U.noise1(i * 0.31 + 4.2) + 0.55 * U.noise1(i * 0.93 + 17.5), 0.06, 0.97);
    cellC[i] = U.magma(v);
  }
  /* readout de minimisation : K + T/K pour chaque K proposé (vrais calculs) */
  const PATHS = KS.map((k) => 'K=' + k + '→' + (k + T / k)).join('  ·  ');

  /* Durées (secondes) */
  const HOLD = 0.7, FOLD = 1.5, SWEEP = 1.6, PSE1 = 0.55, PSE2 = 1.1;
  const BLOCK = SWEEP + PSE1 + SWEEP + PSE2;

  const PHASE_LABEL = {
    hold: 'séquence 1D — T = ' + T,
    fold: 'pliage en chunks…',
    intra: 'passe INTRA (RNN local →)',
    p1: 'intra terminée : la cible voit sa ligne',
    inter: 'passe INTER (RNN global ↓)',
    p2: 'contexte GLOBAL atteint',
  };

  AtlasRegister({
    id: 'dprnn',
    title: 'DPRNN — dual-path pour longues séquences',
    category: 'archi',
    icon: '⫴',
    summary: 'Plier la séquence en chunks puis alterner RNN intra / inter : contexte global en O(√T).',
    explain: `
      <p><strong><dfn class="term" data-term="dprnn">DPRNN</dfn></strong> (<em>Dual-Path RNN</em>, Luo et al. 2020) répond à un problème très concret :
      en <dfn class="term" data-term="source-separation">séparation de sources</dfn> « bout en bout » (<dfn class="term" data-term="tasnet">TasNet</dfn>), l'<dfn class="term" data-term="encoder-decoder">encodeur</dfn> produit des milliers de <dfn class="term" data-term="frame">frames</dfn> par seconde.
      Un <dfn class="term" data-term="rnn">RNN</dfn> unique doit alors propager l'information — et le <dfn class="term" data-term="gradient-descent">gradient</dfn> — sur <code>T</code> pas consécutifs :
      le chemin séquentiel est en <strong><dfn class="term" data-term="big-o">O(T)</dfn></strong>, et le gradient <dfn class="term" data-term="vanishing-gradient">s'évanouit</dfn> bien avant la fin
      (chaîne rouge en bas de l'animation).</p>
      <p>L'idée <dfn class="term" data-term="dual-path">dual-path</dfn> : <strong>plier</strong> la séquence en <code>S = T/K</code> <dfn class="term" data-term="chunk">chunks</dfn> de <code>K</code> frames,
      posés en matrice 2D. On alterne ensuite deux petits RNN : la passe <strong>intra-chunk</strong> (corail)
      parcourt chaque ligne et modélise la structure locale fine ; la passe <strong>inter-chunk</strong> (ambre)
      parcourt chaque colonne et relie les chunks entre eux. Après <strong>un seul bloc intra+inter</strong>,
      chaque frame a accès à toute la séquence : sa ligne d'abord, puis toutes les lignes via les colonnes —
      c'est le <dfn class="term" data-term="receptive-field">champ réceptif</dfn> violet qui se propage dans l'animation.</p>
      <p>Le plus long chemin séquentiel devient <code>K + T/K</code>, minimisé quand <code>K ≈ √T</code>
      (ici √48 ≈ 6.9, d'où l'optimum K = 6 ou 8 → 14 pas au lieu de 48) : on passe de O(T) à
      <strong>O(√T)</strong>. Et comme les deux passes restent des <dfn class="term" data-term="gru">GRU</dfn>/<dfn class="term" data-term="lstm">LSTM</dfn> ordinaires — <dfn class="term" data-term="streaming">streamables</dfn>,
      faciles à rendre <dfn class="term" data-term="causal">causaux</dfn> —, ce motif est devenu le pattern des <dfn class="term" data-term="bottleneck">bottlenecks</dfn> temps réel modernes
      (séparation, enhancement : DPRNN-TasNet, puis DPTNet et <dfn class="term" data-term="sepformer">SepFormer</dfn> qui gardent exactement le même pliage,
      en remplaçant les RNN par de l'<dfn class="term" data-term="attention">attention</dfn>).</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)
      let clk = 0; // horloge locale, rejouable (avance avec dt → respecte pause/vitesse)

      const kCtl = stage.addSlider({
        label: 'Taille de chunk K', min: 0, max: KS.length - 1, step: 1, value: 2,
        format: (v) => 'K = ' + KS[v | 0],
        onChange: () => { clk = 0; },          // la grille se recompose : pliage rejoué
      });
      stage.addButton({ label: 'Replier', onClick: () => { clk = 0; } });
      const rfCtl = stage.addToggle({ label: 'Champ réceptif', value: true });

      /* Chip centrée horizontalement (mesure puis dessin) */
      function chipC(str, cx, y, color, alpha, size) {
        const sz = size || 10;
        ctx.save();
        ctx.globalAlpha = U.clamp(alpha, 0, 1);
        ctx.font = '600 ' + sz + 'px ' + U.FONT;
        const w = ctx.measureText(str).width + 14;
        U.chip(ctx, str, cx - w / 2, y, { color, size: sz });
        ctx.restore();
      }

      /* ===========================================================
         HELPERS DE DESSIN — partagés entre desktop et mobile.
         Tout ce qui dépend de la taille passe par fs() ; toutes les
         dimensions de cellule/grille sont planchées avec Math.max pour
         qu'aucune frame ne puisse produire une taille ≤ 0.
         =========================================================== */

      /* Calcule la géométrie d'une grille S×K tenant dans (areaX,gTop,areaW,availH).
         Renvoie cs (côté de cellule, ≥ 5), gw, gh, gx, gy, gap. */
      function gridGeom(areaX, gTop, areaW, availH, K, S) {
        const gap = 2;
        const wByK = (areaW - (K - 1) * gap) / Math.max(1, K);
        const hByS = (availH - (S - 1) * gap) / Math.max(1, S);
        const cs = Math.max(5, Math.min(wByK, hByS));
        const gw = cs * K + gap * (K - 1), gh = cs * S + gap * (S - 1);
        const gx = areaX + Math.max(0, (areaW - gw) / 2);
        const gy = gTop + Math.max(0, (availH - gh) / 2);
        return { cs, gw, gh, gx, gy, gap };
      }

      /* Dessine le bandeau 1D qui se plie en grille + le champ réceptif.
         g = géométrie de grille ; st = état de frame (phases, couvertures…).
         bandeau : (bAreaX, bY, bAreaW, bH). */
      function drawFold(bAreaX, bY, bAreaW, bH, g, st) {
        const { K, folded, phase, rowF, intraCov, interCov, seenCell, showRF } = st;
        const bcw = bAreaW / T;
        for (let i = 0; i < T; i++) {
          const r = (i / K) | 0, c = i % K;
          const f = rowF(r);
          const x = U.lerp(bAreaX + i * bcw, g.gx + c * (g.cs + g.gap), f);
          const y = U.lerp(bY, g.gy + r * (g.cs + g.gap), f);
          const w = U.lerp(Math.max(1, bcw - 1), g.cs, f);
          const h = U.lerp(bH, g.cs, f);
          ctx.fillStyle = cellC[i];
          U.roundRect(ctx, x, y, w, h, 2);
          ctx.fill();

          if (folded) {
            const ic = intraCov(c), rc = interCov(r);
            if (rc || ic) {
              ctx.strokeStyle = rc ? palette.rest : palette.voice;
              ctx.lineWidth = 1.4;
              ctx.globalAlpha = 0.95;
              U.roundRect(ctx, x + 0.5, y + 0.5, w - 1, h - 1, 2);
              ctx.stroke();
              ctx.globalAlpha = 1;
            }
            if (showRF && seenCell(r, c)) {
              ctx.fillStyle = palette.mix;
              ctx.globalAlpha = 0.26;
              U.roundRect(ctx, x, y, w, h, 2);
              ctx.fill();
              ctx.globalAlpha = 1;
            }
          }
        }
      }

      /* Annotations d'axes + cible + glows de balayage sur la grille.
         axisFs = taille des étiquettes d'axe ; showSideAxis : étiquette inter
         (verticale à gauche, si la place existe). */
      function drawGridAnnot(g, st, axisFs, showSideAxis) {
        const { K, S, tr, tc, showRF, phase, sw } = st;
        U.text(ctx, 'K = ' + K + ' frames — intra →', g.gx + g.gw / 2, g.gy - fs(7),
          { size: axisFs, color: palette.voice, align: 'center' });
        if (showSideAxis && g.gx - st.areaX >= 16) {
          ctx.save();
          ctx.translate(g.gx - 11, g.gy + g.gh / 2);
          ctx.rotate(-Math.PI / 2);
          U.text(ctx, 'S = ' + S + ' chunks — inter ↓', 0, 0, { size: axisFs, color: palette.rest, align: 'center' });
          ctx.restore();
        }

        if (showRF) {
          const tx = g.gx + tc * (g.cs + g.gap), ty = g.gy + tr * (g.cs + g.gap);
          ctx.save();
          ctx.strokeStyle = palette.mix;
          ctx.lineWidth = 2.2;
          ctx.shadowColor = palette.mix;
          ctx.shadowBlur = 8;
          U.roundRect(ctx, tx - 1.5, ty - 1.5, g.cs + 3, g.cs + 3, 3);
          ctx.stroke();
          ctx.restore();
          if (g.cs > 9) {
            U.text(ctx, 'cible', tx + g.cs / 2, tr === 0 ? ty + g.cs + fs(11) : ty - fs(4),
              { size: axisFs - 1, color: palette.mix, align: 'center' });
          }
        }

        if (phase === 'intra') {
          const x = g.gx + sw * g.gw;
          ctx.strokeStyle = palette.voice;
          ctx.globalAlpha = 0.3;
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(x, g.gy); ctx.lineTo(x, g.gy + g.gh); ctx.stroke();
          ctx.globalAlpha = 1;
          for (let r = 0; r < S; r++) U.glowDot(ctx, x, g.gy + r * (g.cs + g.gap) + g.cs / 2, 3.2, palette.voice);
        }
        if (phase === 'inter') {
          const y = g.gy + sw * g.gh;
          ctx.strokeStyle = palette.rest;
          ctx.globalAlpha = 0.3;
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(g.gx, y); ctx.lineTo(g.gx + g.gw, y); ctx.stroke();
          ctx.globalAlpha = 1;
          for (let c = 0; c < K; c++) U.glowDot(ctx, g.gx + c * (g.cs + g.gap) + g.cs / 2, y, 3.2, palette.rest);
        }
      }

      /* Comparatif RNN simple vs DPRNN (chaînes de flèches + survie du gradient).
         Dessiné dans la bande (bx, by, bw, bandH). lblFs = taille de texte. */
      function drawCompare(bx, by, bw, bandH, lblFs, st) {
        const { K, S, path } = st;
        ctx.strokeStyle = palette.grid;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + bw, by); ctx.stroke();

        const step = bw / T;
        const l1y = by + lblFs + 3, a1y = by + bandH * 0.34;
        const l2y = by + bandH * 0.60, a2y = by + bandH * 0.84;
        const surv1 = Math.pow(DECAY, T - 1);     // survie du gradient après T pas
        const surv2 = Math.pow(DECAY, path - 1);  // après K + T/K pas

        U.text(ctx, 'RNN simple — T = ' + T + ' pas (×' + DECAY + '/pas)', bx, l1y,
          { size: lblFs, color: palette.dim });
        U.text(ctx, 'survie ≈ ' + surv1.toFixed(2), bx + bw, l1y, { size: lblFs, color: palette.dim, align: 'right' });
        for (let i = 0; i < T; i++) {
          U.arrow(ctx, bx + i * step + 1, a1y, bx + (i + 1) * step - 1, a1y,
            { color: palette.dim, alpha: Math.max(0.045, Math.pow(DECAY, i)), head: 3, lw: 1.2 });
        }

        U.text(ctx, 'DPRNN — K + T/K = ' + K + ' + ' + S + ' = ' + path + ' pas', bx, l2y,
          { size: lblFs, color: palette.dim });
        const safePath = Math.max(1, Math.min(T, path | 0));
        for (let i = 0; i < safePath; i++) {
          U.arrow(ctx, bx + i * step + 1, a2y, bx + (i + 1) * step - 1, a2y,
            { color: i < K ? palette.voice : palette.rest, alpha: Math.max(0.05, Math.pow(DECAY, i)), head: 3, lw: 1.2 });
        }
        U.text(ctx, 'survie ≈ ' + surv2.toFixed(2) + ' — le gradient vit', bx + safePath * step + 8, a2y + 3,
          { size: lblFs, color: palette.green });
      }

      stage.onFrame((t, dt) => {
        clk += dt;
        const W = stage.W, H = stage.H;
        stage.clear();
        const compact = stage.compact;
        const K = KS[kCtl.value | 0];
        const S = T / K;                      // nombre de chunks
        const path = K + S;                   // chemin séquentiel max (vrai calcul)
        const showRF = rfCtl.value;

        /* ---------- machine à phases ---------- */
        let phase, ph = 0, blockIdx = 0;
        if (clk < HOLD) { phase = 'hold'; ph = clk / HOLD; }
        else if (clk < HOLD + FOLD) { phase = 'fold'; ph = (clk - HOLD) / FOLD; }
        else {
          const cyc = clk - HOLD - FOLD;
          blockIdx = Math.floor(cyc / BLOCK);
          const bt = cyc - blockIdx * BLOCK;
          if (bt < SWEEP) { phase = 'intra'; ph = bt / SWEEP; }
          else if (bt < SWEEP + PSE1) { phase = 'p1'; ph = (bt - SWEEP) / PSE1; }
          else if (bt < SWEEP + PSE1 + SWEEP) { phase = 'inter'; ph = (bt - SWEEP - PSE1) / SWEEP; }
          else { phase = 'p2'; ph = (bt - SWEEP - PSE1 - SWEEP) / PSE2; }
        }
        const folded = phase !== 'hold' && phase !== 'fold';
        const B = folded ? blockIdx + (phase === 'p2' ? 1 : 0) : 0;   // blocs intra+inter terminés
        const sw = U.smoothstep(ph);                                   // balayage adouci

        /* cible auto-sélectionnée cycliquement (gcd(11,48)=1 → couvre tout) */
        const tIdx = (5 + blockIdx * 11) % T;
        const tr = (tIdx / K) | 0, tc = tIdx % K;

        /* couverture des balayages */
        const intraCov = (c) => phase === 'intra' ? sw * K >= c + 0.5 : folded && phase !== 'fold';
        const interCov = (r) => phase === 'inter' ? sw * S >= r + 0.5 : phase === 'p2';
        /* champ réceptif : vraie propagation — la ligne de la cible, puis toutes les
           lignes via les colonnes (chaque colonne contient une case déjà vue) */
        const seenCell = (r, c) => {
          if (!folded) return false;
          if (r === tr && c === tc) return true;
          if (phase === 'intra') return r === tr && intraCov(c);
          if (phase === 'p1') return r === tr;
          if (phase === 'inter') return r === tr || interCov(r);
          return true; // p2 : tout
        };
        let seenN = 0;
        if (folded && showRF) {
          for (let i = 0; i < T; i++) if (seenCell((i / K) | 0, i % K)) seenN++;
        }

        /* progression du pliage par ligne (easée, étagée) */
        const stag = S > 1 ? 0.55 / (S - 1) : 0;
        const rowF = (r) => phase === 'fold'
          ? U.ease(U.clamp((ph - r * stag) / 0.45, 0, 1))
          : (folded ? 1 : 0);

        /* état commun passé aux helpers */
        const st = { K, S, path, folded, phase, sw, tr, tc, showRF, rowF, intraCov, interCov, seenCell };

        if (compact) {
          /* ===================================================
             MOBILE — empilement vertical (profite de la hauteur).
             1. titre + statut
             2. bandeau 1D → grille de pliage (en GRAND)
             3. paramètres (T, √T, K/S, chemin, blocs) en pleine largeur
             4. comparatif gradient RNN vs DPRNN
             =================================================== */
          const m = 12;
          const W2 = W - 2 * m;

          /* --- 1. en-tête --- */
          U.text(ctx, 'DPRNN — plier la séquence', W / 2, fs(15), { size: fs(14), bold: true, align: 'center' });
          U.text(ctx, PHASE_LABEL[phase], W / 2, fs(15) + fs(14), { size: fs(11), color: palette.dim, align: 'center' });

          /* --- répartition verticale : grille / params / comparatif --- */
          let y = fs(15) + fs(14) + fs(12);
          const compareH = U.clamp(H * 0.24, 96, 150);   // comparatif gradient en bas
          const paramH = fs(12) * 3 + fs(11) + 30;        // bloc paramètres (mesuré ci-dessous)
          const gridTop = y;
          const gridBottom = H - compareH - paramH - fs(18) * 2;
          const gridAvailH = Math.max(70, gridBottom - gridTop - fs(28));

          /* --- 2. bandeau + grille --- */
          const bH = U.clamp(gridAvailH * 0.10, 14, 22);
          const bY = gridTop + fs(16);
          if (!folded) {
            ctx.save();
            ctx.globalAlpha = phase === 'fold' ? 1 - U.clamp(ph * 2, 0, 1) : 1;
            U.text(ctx, 'Séquence longue — T = ' + T + ' frames', m, bY - fs(6), { size: fs(11), color: palette.dim });
            U.text(ctx, '0', m, bY + bH + fs(12), { size: fs(9), color: palette.faint, mono: true });
            U.text(ctx, '47', W - m, bY + bH + fs(12), { size: fs(9), color: palette.faint, align: 'right', mono: true });
            ctx.restore();
          }
          if (phase === 'fold') {
            U.text(ctx, 'pliage : S = ' + S + ' × K = ' + K, W / 2, bY - fs(6),
              { size: fs(11), color: palette.dim, align: 'center' });
          }

          const gTop = bY + bH + fs(14);
          const g = gridGeom(m, gTop, W2, Math.max(60, gridAvailH - (gTop - gridTop) + fs(16)), K, S);
          st.areaX = m;
          drawFold(m, bY, W2, bH, g, st);
          if (folded) {
            drawGridAnnot(g, st, fs(11), true);
            const cy = Math.min(g.gy + g.gh + fs(18), gridBottom - fs(4));
            if (phase === 'p1' && showRF) {
              chipC('après l’intra : la cible voit sa ligne (' + seenN + '/' + T + ')',
                W / 2, cy, palette.voice, U.ease(ph * 2.5), fs(10.5));
            } else if (phase === 'p2') {
              chipC('1 bloc intra+inter = contexte GLOBAL' + (showRF ? ' (' + seenN + '/' + T + ')' : ''),
                W / 2, cy, palette.mix, U.ease(ph * 2.5), fs(10.5));
            }
          }

          /* --- 3. panneau paramètres pleine largeur --- */
          const py = H - compareH - paramH - fs(14);
          U.frame(ctx, m, py, W2, paramH, 'paramètres');
          let yy = py + fs(20);
          U.text(ctx, 'T = ' + T + ' frames   ·   √T ≈ ' + SQT.toFixed(1), m + 12, yy, { size: fs(12) });
          yy += fs(20);
          let cxp = m + 12;
          cxp += U.chip(ctx, 'K = ' + K + ' intra', cxp, yy, { color: palette.voice, size: fs(10) }) + 8;
          U.chip(ctx, 'S = T/K = ' + S + ' inter', cxp, yy, { color: palette.rest, size: fs(10) });
          yy += fs(22);
          /* auto-ajuste la police à la largeur du panneau (cf. drawCost dans conv2d.js) :
             on mesure le texte mono/bold et on réduit la taille s'il dépasse la zone utile
             (largeur intérieure = W2 moins les 12 px de marge de chaque côté). */
          const pathStr = 'chemin séquentiel max : K + T/K = ' + K + ' + ' + S + ' = ' + path + ' pas';
          const pathMaxW = Math.max(40, W2 - 24);
          let pathFz = fs(12);
          ctx.font = '600 ' + pathFz + 'px ' + U.MONO;
          const pathW = ctx.measureText(pathStr).width;
          if (pathW > pathMaxW) pathFz = Math.max(fs(8.5), pathFz * pathMaxW / pathW);
          U.text(ctx, pathStr, m + 12, yy, { size: pathFz, bold: true, color: palette.mix, mono: true });
          yy += fs(16);
          U.text(ctx, 'min vers K ≈ √T ≈ ' + SQT.toFixed(1) + '   ·   blocs B = ' + B,
            m + 12, yy, { size: fs(10.5), color: palette.faint });

          /* --- 4. comparatif gradient pleine largeur --- */
          drawCompare(m, H - compareH, W2, compareH, fs(11), st);
        } else {
          /* ===================================================
             DESKTOP / TABLETTE (≥ 560) — disposition d'origine :
             grille à gauche, panneau d'infos à droite, comparatif en bas.
             =================================================== */
          const m = 10;
          const headY = 24;
          const botH = U.clamp(H * 0.27, 76, 122);
          const mainY = headY + 6;
          const mainH = H - mainY - botH - 10;
          const panelW = U.clamp(W * 0.30, 185, 255);
          const areaX = m, areaW = W - 2 * m - (panelW + 10);

          const bH = U.clamp(mainH * 0.10, 12, 20);       // bandeau 1D
          const bY = mainY + 16;

          const gTop = bY + bH + 10;
          const gAvailH = mainY + mainH - gTop - 26;       // 26 px réservés aux chips
          const g = gridGeom(areaX, gTop, areaW, gAvailH, K, S);
          st.areaX = areaX;

          /* en-tête */
          U.text(ctx, 'DPRNN — plier la séquence, alterner intra ⇄ inter',
            m, 16, { size: 13, bold: true });
          U.text(ctx, PHASE_LABEL[phase], W - m, 16, { size: 11, color: palette.dim, align: 'right' });

          /* étiquette du bandeau (s'efface pendant le pliage) */
          if (!folded) {
            ctx.save();
            ctx.globalAlpha = phase === 'fold' ? 1 - U.clamp(ph * 2, 0, 1) : 1;
            U.text(ctx, 'Séquence longue — T = ' + T + ' frames', areaX, bY - 6, { size: 11, color: palette.dim });
            U.text(ctx, '0', areaX, bY + bH + 11, { size: 9, color: palette.faint, mono: true });
            U.text(ctx, '47', areaX + areaW, bY + bH + 11, { size: 9, color: palette.faint, align: 'right', mono: true });
            ctx.restore();
          }
          if (phase === 'fold') {
            U.text(ctx, 'pliage : S = ' + S + ' chunks × K = ' + K + ' frames',
              areaX + areaW / 2, bY - 6, { size: 11, color: palette.dim, align: 'center' });
          }

          drawFold(areaX, bY, areaW, bH, g, st);

          if (folded) {
            drawGridAnnot(g, st, 10, true);
            const cy = Math.min(g.gy + g.gh + 16, mainY + mainH - 6);
            if (phase === 'p1' && showRF) {
              chipC('après l’intra : la cible voit sa ligne (' + seenN + '/' + T + ' cases)',
                g.gx + g.gw / 2, cy, palette.voice, U.ease(ph * 2.5), 10);
            } else if (phase === 'p2') {
              chipC('1 bloc intra+inter = contexte GLOBAL' + (showRF ? ' (' + seenN + '/' + T + ')' : ''),
                g.gx + g.gw / 2, cy, palette.mix, U.ease(ph * 2.5), 10);
            }
          }

          /* panneau d'infos */
          const px = W - m - panelW, py = mainY + 4, pH = mainH - 8;
          U.frame(ctx, px, py, panelW, pH, 'paramètres');
          let yy = py + 22;
          U.text(ctx, 'T = ' + T + ' frames   ·   √T ≈ ' + SQT.toFixed(1), px + 12, yy, { size: 11 });
          yy += 22;
          let cxp = px + 12;
          cxp += U.chip(ctx, 'K = ' + K + ' intra', cxp, yy, { color: palette.voice }) + 8;
          U.chip(ctx, 'S = T/K = ' + S + ' inter', cxp, yy, { color: palette.rest });
          yy += 24;
          U.text(ctx, 'chemin séquentiel max :', px + 12, yy, { size: 10, color: palette.dim });
          yy += 17;
          U.text(ctx, 'K + T/K = ' + K + ' + ' + S + ' = ' + path + ' pas', px + 12, yy,
            { size: 12, bold: true, color: palette.mix, mono: true });
          if (pH > 150) {
            yy += 17;
            U.text(ctx, PATHS, px + 12, yy, { size: 9, color: palette.faint, mono: true });
            yy += 14;
            U.text(ctx, 'minimum autour de K ≈ √T ≈ ' + SQT.toFixed(1), px + 12, yy,
              { size: 9, color: palette.faint });
          }
          yy += 22;
          U.text(ctx, 'blocs traités : B = ' + B, px + 12, yy, { size: 11, bold: true });
          if (pH > 215) {
            yy += 24;
            const leg = [
              [palette.voice, 'passe intra — locale, par chunk'],
              [palette.rest, 'passe inter — globale, entre chunks'],
              [palette.mix, 'champ réceptif de la cible'],
            ];
            for (const [col, lab] of leg) {
              ctx.fillStyle = col;
              ctx.globalAlpha = 0.85;
              ctx.fillRect(px + 12, yy - 8, 9, 9);
              ctx.globalAlpha = 1;
              U.text(ctx, lab, px + 27, yy, { size: 10, color: palette.dim });
              yy += 16;
            }
          }

          /* comparatif bas */
          drawCompare(m, H - botH, W - 2 * m, botH, 10, st);
        }
      });
    },
  });
})();
