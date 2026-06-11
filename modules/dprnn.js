/* ============================================================
   Audio AI Atlas — module DPRNN (dual-path pour longues séquences)
   Pliage 1D → 2D, passes RNN intra-chunk / inter-chunk alternées,
   champ réceptif et chemins de gradient en O(√T).
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
      <p><strong>DPRNN</strong> (<em>Dual-Path RNN</em>, Luo et al. 2020) répond à un problème très concret :
      en séparation de sources « bout en bout » (TasNet), l'encodeur produit des milliers de frames par seconde.
      Un RNN unique doit alors propager l'information — et le gradient — sur <code>T</code> pas consécutifs :
      le chemin séquentiel est en <strong>O(T)</strong>, et le gradient s'évanouit bien avant la fin
      (chaîne rouge en bas de l'animation).</p>
      <p>L'idée dual-path : <strong>plier</strong> la séquence en <code>S = T/K</code> chunks de <code>K</code> frames,
      posés en matrice 2D. On alterne ensuite deux petits RNN : la passe <strong>intra-chunk</strong> (teal)
      parcourt chaque ligne et modélise la structure locale fine ; la passe <strong>inter-chunk</strong> (ambre)
      parcourt chaque colonne et relie les chunks entre eux. Après <strong>un seul bloc intra+inter</strong>,
      chaque frame a accès à toute la séquence : sa ligne d'abord, puis toutes les lignes via les colonnes —
      c'est le champ réceptif violet qui se propage dans l'animation.</p>
      <p>Le plus long chemin séquentiel devient <code>K + T/K</code>, minimisé quand <code>K ≈ √T</code>
      (ici √48 ≈ 6.9, d'où l'optimum K = 6 ou 8 → 14 pas au lieu de 48) : on passe de O(T) à
      <strong>O(√T)</strong>. Et comme les deux passes restent des GRU/LSTM ordinaires — streamables,
      faciles à rendre causaux —, ce motif est devenu le pattern des bottlenecks temps réel modernes
      (séparation, enhancement : DPRNN-TasNet, puis DPTNet et SepFormer qui gardent exactement le même pliage,
      en remplaçant les RNN par de l'attention).</p>`,

    init(stage) {
      const ctx = stage.ctx;
      let clk = 0; // horloge locale, rejouable (avance avec dt → respecte pause/vitesse)

      const kCtl = stage.addSlider({
        label: 'Taille de chunk K', min: 0, max: KS.length - 1, step: 1, value: 2,
        format: (v) => 'K = ' + KS[v | 0],
        onChange: () => { clk = 0; },          // la grille se recompose : pliage rejoué
      });
      stage.addButton({ label: 'Replier', onClick: () => { clk = 0; } });
      const rfCtl = stage.addToggle({ label: 'Champ réceptif', value: true });

      /* Chip centrée horizontalement (mesure puis dessin) */
      function chipC(str, cx, y, color, alpha) {
        ctx.save();
        ctx.globalAlpha = U.clamp(alpha, 0, 1);
        ctx.font = '600 10px ' + U.FONT;
        const w = ctx.measureText(str).width + 14;
        U.chip(ctx, str, cx - w / 2, y, { color });
        ctx.restore();
      }

      stage.onFrame((t, dt) => {
        clk += dt;
        const W = stage.W, H = stage.H;
        stage.clear();
        const narrow = W < 560;
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

        /* ---------- mise en page (recalculée chaque frame) ---------- */
        const m = 10;
        const headY = 24;
        const botH = U.clamp(H * 0.27, 76, 122);
        const mainY = headY + 6;
        const mainH = H - mainY - botH - 10;
        const panelW = narrow ? 0 : U.clamp(W * 0.30, 185, 255);
        const areaX = m, areaW = W - 2 * m - (panelW ? panelW + 10 : 0);

        const bH = U.clamp(mainH * 0.10, 12, 20);       // bandeau 1D
        const bY = mainY + 16;
        const bcw = areaW / T;

        const gap = 2;
        const gTop = bY + bH + 10;
        const gAvailH = mainY + mainH - gTop - 26;       // 26 px réservés aux chips
        const cs = Math.max(5, Math.min((areaW - (K - 1) * gap) / K, (gAvailH - (S - 1) * gap) / S));
        const gw = cs * K + gap * (K - 1), gh = cs * S + gap * (S - 1);
        const gx = areaX + (areaW - gw) / 2;
        const gy = gTop + Math.max(0, (gAvailH - gh) / 2);

        /* ---------- en-tête ---------- */
        U.text(ctx, narrow ? 'DPRNN' : 'DPRNN — plier la séquence, alterner intra ⇄ inter',
          m, 16, { size: 13, bold: true });
        if (narrow) {
          U.text(ctx, 'K=' + K + ' · ' + S + '×' + K + ' · chemin ' + path + ' · B=' + B,
            W - m, 16, { size: 9, color: palette.dim, align: 'right', mono: true });
        } else {
          U.text(ctx, PHASE_LABEL[phase], W - m, 16, { size: 11, color: palette.dim, align: 'right' });
        }

        /* ---------- pliage : progression par ligne (easée, étagée) ---------- */
        const stag = S > 1 ? 0.55 / (S - 1) : 0;
        const rowF = (r) => phase === 'fold'
          ? U.ease(U.clamp((ph - r * stag) / 0.45, 0, 1))
          : (folded ? 1 : 0);

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

        /* ---------- les 48 cases (interpolation bandeau → grille) ---------- */
        for (let i = 0; i < T; i++) {
          const r = (i / K) | 0, c = i % K;
          const f = rowF(r);
          const x = U.lerp(areaX + i * bcw, gx + c * (cs + gap), f);
          const y = U.lerp(bY, gy + r * (cs + gap), f);
          const w = U.lerp(Math.max(1, bcw - 1), cs, f);
          const h = U.lerp(bH, cs, f);
          ctx.fillStyle = cellC[i];
          U.roundRect(ctx, x, y, w, h, 2);
          ctx.fill();

          if (folded) {
            /* liserés gagnés au passage des glows */
            const ic = intraCov(c), rc = interCov(r);
            if (rc || ic) {
              ctx.strokeStyle = rc ? palette.rest : palette.voice;
              ctx.lineWidth = 1.4;
              ctx.globalAlpha = 0.95;
              U.roundRect(ctx, x + 0.5, y + 0.5, w - 1, h - 1, 2);
              ctx.stroke();
              ctx.globalAlpha = 1;
            }
            /* champ réceptif (violet) */
            if (showRF && seenCell(r, c)) {
              ctx.fillStyle = palette.mix;
              ctx.globalAlpha = 0.26;
              U.roundRect(ctx, x, y, w, h, 2);
              ctx.fill();
              ctx.globalAlpha = 1;
            }
          }
        }

        /* ---------- annotations d'axes + cible + glows ---------- */
        if (folded) {
          U.text(ctx, 'K = ' + K + ' frames — intra →', gx + gw / 2, gy - 7,
            { size: 10, color: palette.voice, align: 'center' });
          if (!narrow && gx - areaX >= 16) {
            ctx.save();
            ctx.translate(gx - 11, gy + gh / 2);
            ctx.rotate(-Math.PI / 2);
            U.text(ctx, 'S = ' + S + ' chunks — inter ↓', 0, 0, { size: 10, color: palette.rest, align: 'center' });
            ctx.restore();
          }

          if (showRF) {
            const tx = gx + tc * (cs + gap), ty = gy + tr * (cs + gap);
            ctx.save();
            ctx.strokeStyle = palette.mix;
            ctx.lineWidth = 2.2;
            ctx.shadowColor = palette.mix;
            ctx.shadowBlur = 8;
            U.roundRect(ctx, tx - 1.5, ty - 1.5, cs + 3, cs + 3, 3);
            ctx.stroke();
            ctx.restore();
            if (cs > 9) {
              U.text(ctx, 'cible', tx + cs / 2, tr === 0 ? ty + cs + 11 : ty - 4,
                { size: 9, color: palette.mix, align: 'center' });
            }
          }

          if (phase === 'intra') {
            const x = gx + sw * gw;
            ctx.strokeStyle = palette.voice;
            ctx.globalAlpha = 0.3;
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x, gy + gh); ctx.stroke();
            ctx.globalAlpha = 1;
            for (let r = 0; r < S; r++) U.glowDot(ctx, x, gy + r * (cs + gap) + cs / 2, 3.2, palette.voice);
          }
          if (phase === 'inter') {
            const y = gy + sw * gh;
            ctx.strokeStyle = palette.rest;
            ctx.globalAlpha = 0.3;
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx + gw, y); ctx.stroke();
            ctx.globalAlpha = 1;
            for (let c = 0; c < K; c++) U.glowDot(ctx, gx + c * (cs + gap) + cs / 2, y, 3.2, palette.rest);
          }

          /* chips conclusives sous la grille */
          const cy = Math.min(gy + gh + 16, mainY + mainH - 6);
          if (phase === 'p1' && showRF) {
            chipC('après l’intra : la cible voit sa ligne (' + seenN + '/' + T + ' cases)',
              gx + gw / 2, cy, palette.voice, U.ease(ph * 2.5));
          } else if (phase === 'p2') {
            chipC('1 bloc intra+inter = contexte GLOBAL' + (showRF ? ' (' + seenN + '/' + T + ')' : ''),
              gx + gw / 2, cy, palette.mix, U.ease(ph * 2.5));
          }
        }

        /* ---------- panneau d'infos (large uniquement) ---------- */
        if (panelW) {
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
        }

        /* ---------- comparatif bas : RNN simple vs DPRNN ---------- */
        const by = H - botH;
        ctx.strokeStyle = palette.grid;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(m, by); ctx.lineTo(W - m, by); ctx.stroke();

        const aw = W - 2 * m, step = aw / T;
        const fs = narrow ? 9 : 10;
        const l1y = by + 14, a1y = by + botH * 0.34;
        const l2y = by + botH * 0.60, a2y = by + botH * 0.84;
        const surv1 = Math.pow(DECAY, T - 1);     // survie du gradient après T pas
        const surv2 = Math.pow(DECAY, path - 1);  // après K + T/K pas

        U.text(ctx, 'RNN simple — T = ' + T + ' pas séquentiels (×' + DECAY + ' par pas)', m, l1y,
          { size: fs, color: palette.dim });
        U.text(ctx, 'survie ≈ ' + surv1.toFixed(2), W - m, l1y, { size: fs, color: palette.red, align: 'right' });
        for (let i = 0; i < T; i++) {
          U.arrow(ctx, m + i * step + 1, a1y, m + (i + 1) * step - 1, a1y,
            { color: palette.red, alpha: Math.max(0.045, Math.pow(DECAY, i)), head: 3, lw: 1.2 });
        }

        U.text(ctx, 'DPRNN — chemin max = K + T/K = ' + K + ' + ' + S + ' = ' + path + ' pas', m, l2y,
          { size: fs, color: palette.dim });
        for (let i = 0; i < path; i++) {
          U.arrow(ctx, m + i * step + 1, a2y, m + (i + 1) * step - 1, a2y,
            { color: i < K ? palette.voice : palette.rest, alpha: Math.max(0.05, Math.pow(DECAY, i)), head: 3, lw: 1.2 });
        }
        U.text(ctx, 'survie ≈ ' + surv2.toFixed(2) + ' — le gradient vit', m + path * step + 8, a2y + 3,
          { size: fs, color: palette.green });
      });
    },
  });
})();
