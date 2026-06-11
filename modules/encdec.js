/* =====================================================================
   Audio AI Atlas — module « Encoder–Decoder & U-Net »
   Diagramme en U animé : encodeur (compression), bottleneck récurrent,
   décodeur (reconstruction), skip connections. Démo clé : sans skips,
   la sortie devient spectralement floue (moyenne sur 5 bins).
   ===================================================================== */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  const SR = 16000;        // taux d'échantillonnage des générateurs
  const NFFT = 256;        // FFT de l'aperçu spectrogramme
  const NBINS = NFFT / 2;  // 128 bins poussés dans les Scrollers
  const COL_DT = 1 / 32;   // 32 colonnes de spectrogramme / seconde
  const PERIOD = 4;        // durée d'un tour complet du pulse (s)
  const MAXD = 4;          // profondeur maximale du slider

  AtlasRegister({
    id: 'encdec',
    title: 'Encoder–Decoder & U-Net',
    category: 'archi',
    icon: '⋈',
    summary: 'Compression en U, bottleneck récurrent et skips : le QUOI passe par le fond, le OÙ par les skips.',
    explain: `
      <p>L'architecture <strong>encodeur–décodeur</strong> (et sa variante <strong>U-Net</strong>) domine
      le traitement audio neuronal — DeepFilterNet (débruitage), Demucs (séparation de sources), la plupart
      des modèles de speech enhancement. L'encodeur empile des convolutions à <em>stride&nbsp;2</em> :
      à chaque étage, la résolution temps–fréquence est divisée par 2 dans chaque dimension pendant que le
      nombre de canaux augmente (F=256→128→64→32, C=2→16→32→64). Au <strong>bottleneck</strong>, chaque
      « pixel » résume une grande région du spectrogramme : le réseau y encode le <strong>QUOI</strong> —
      présence de voix, timbre, phonème — mais il a perdu le <strong>OÙ</strong> précis (le bin exact,
      la micro-transition).</p>
      <p>Placer un module récurrent (<code>GRU ×2</code>) au fond du U est un choix d'efficacité : la
      récurrence, séquentielle par nature, coûte cher à chaque pas de temps. Au bottleneck, le temps est
      compressé ×8 et la fréquence ×8 — c'est l'endroit le moins cher pour modéliser la
      <strong>dynamique temporelle longue</strong> (prosodie, attaques et tenues, continuité de la voix),
      exactement la recette des CRN (<em>convolutional recurrent networks</em>) comme DeepFilterNet.</p>
      <p>Les <strong>skip connections</strong> copient les activations de chaque étage d'encodeur vers
      l'étage symétrique du décodeur (concaténation). Le détail spectral fin court-circuite ainsi le
      bottleneck : c'est par là que passe le <strong>OÙ</strong>. Bonus : ce sont aussi des autoroutes de
      gradient qui stabilisent l'entraînement profond. Coupez le toggle pour le constater — sans skips, le
      décodeur ne peut reconstruire qu'une version lissée du spectre (ici, une vraie moyenne glissante sur
      5 bins), fidèle à la perte réelle de localisation fréquentielle.</p>
      <p>C'est crucial pour le <strong>masking</strong> : débruiter ou séparer revient à prédire un masque
      appliqué bin par bin au spectrogramme d'entrée. Sans la précision portée par les skips, le masque
      « baverait » sur les harmoniques voisines — la reconstruction fine à pleine résolution est précisément
      ce que l'encodeur–décodeur à skips garantit.</p>`,

    init(stage) {
      /* ---------- buffers pré-alloués (jamais de gros new dans onFrame) ---------- */
      const sig = new Float32Array(NFFT);      // fenêtre signal
      const sharp = new Float32Array(NBINS);   // colonne nette
      const blurC = new Float32Array(NBINS);   // colonne floutée (5 bins)
      const outCol = new Float32Array(NBINS);  // colonne effectivement poussée en sortie
      const scrIn = new U.Scroller(90, 54, U.magmaRGB);
      const scrOut = new U.Scroller(90, 54, U.magmaRGB);
      const NWP = 2 * MAXD + 3;                // waypoints du pulse
      const wpX = new Float64Array(NWP), wpY = new Float64Array(NWP);
      const encX = new Float64Array(MAXD), encW = new Float64Array(MAXD), levY = new Float64Array(MAXD);
      let lastCol = -1;                        // index de la dernière colonne poussée
      let blurMix = 0;                         // 0 = skips ON (net) → 1 = OFF (flou), interpolé

      /* ---------- contrôles ---------- */
      const ctlDepth = stage.addSlider({
        label: 'Profondeur', min: 2, max: MAXD, step: 1, value: 3,
        format: (v) => v + ' étages',
      });
      const ctlSkip = stage.addToggle({ label: 'Skip connections', value: true });
      const ctlSrc = stage.addSelect({
        label: 'Source',
        options: [{ value: 'speech', label: 'Voix' }, { value: 'music', label: 'Musique' }],
        value: 'speech',
      });

      /* Pousse une colonne : entrée nette ; sortie = mêmes bins, floutés si skips OFF */
      function pushColumn(t0) {
        const fn = ctlSrc.value === 'music' ? U.gen.music : U.gen.speech;
        for (let i = 0; i < NFFT; i++) sig[i] = fn(t0 + i / SR);
        const mag = U.rfftMag(sig);
        for (let i = 0; i < NBINS; i++) sharp[i] = U.clamp(Math.pow(mag[i] * 2.6, 0.6), 0, 1);
        for (let i = 0; i < NBINS; i++) {      // vrai flou spectral : moyenne sur 5 bins
          let s = 0, n = 0;
          for (let k = -2; k <= 2; k++) {
            const j = i + k;
            if (j >= 0 && j < NBINS) { s += sharp[j]; n++; }
          }
          blurC[i] = s / n;
        }
        const bm = U.smoothstep(blurMix);
        for (let i = 0; i < NBINS; i++) outCol[i] = U.lerp(sharp[i], blurC[i], bm);
        scrIn.push(sharp);
        scrOut.push(outCol);
      }

      /* point d'une courbe quadratique (arcs de skip), composante par composante */
      function qPt(a, c, b, f) { const g = 1 - f; return g * g * a + 2 * g * f * c + f * f * b; }

      /* ligne de readout multicolore, centrée en cx */
      function readout(ctx, parts, cx, y, size) {
        let total = 0;
        for (let i = 0; i < parts.length; i++) {
          ctx.font = `${parts[i].b ? '600 ' : ''}${size}px ${U.FONT}`;
          parts[i].w = ctx.measureText(parts[i].s).width;
          total += parts[i].w;
        }
        let x = cx - total / 2;
        for (let i = 0; i < parts.length; i++) {
          U.text(ctx, parts[i].s, x, y, { size, color: parts[i].c, bold: parts[i].b });
          x += parts[i].w;
        }
      }

      stage.onFrame((t, dt) => {
        const ctx = stage.ctx, W = stage.W, H = stage.H;
        stage.clear();
        const compact = W < 560;
        const d = U.clamp(Math.round(ctlDepth.value), 2, MAXD);

        /* transition continue net↔flou (réagit aux contrôles même en pause) */
        const uiDt = dt > 0 ? dt : 1 / 60;
        blurMix += ((ctlSkip.value ? 0 : 1) - blurMix) * Math.min(1, uiDt * 5);
        const skipAlpha = U.lerp(0.9, 0.05, U.smoothstep(blurMix));

        /* colonnes des spectrogrammes (gelées en pause : t n'avance plus) */
        const colIdx = Math.floor(t / COL_DT);
        if (lastCol < 0 || colIdx - lastCol > 3) lastCol = colIdx - 1;
        while (lastCol < colIdx) { lastCol++; pushColumn(lastCol * COL_DT); }

        /* dimensions RÉELLES le long du U : F divisé par 2, C augmente, T divisé par 2 */
        const F = [256], C = [2];
        for (let i = 1; i <= d; i++) { F.push(256 >> i); C.push(8 << i); }
        const tDiv = 1 << d;                   // compression temporelle (stride 2 / étage)
        const comp = (F[0] / F[d]) * tDiv;     // cellules temps-fréquence : ×4^profondeur

        /* ---------- layout, recalculé chaque frame depuis W/H ---------- */
        const m = compact ? 8 : 16;
        const sw = U.clamp(W * 0.2, 72, 140), sh = sw * 0.6;
        const specY = m + (compact ? 13 : 17);
        const inX = m, outX = W - m - sw;
        const nodeH = compact ? 26 : 34;
        const wMax = U.clamp(W * 0.21, 68, 150);
        const bnW = U.clamp(W * 0.17, 80, 130), bnH = compact ? 30 : 38;
        const yBN = H - (compact ? 26 : 32) - bnH / 2;
        const yTop = specY + sh + (compact ? 24 : 44);
        const yLow = Math.max(yBN - bnH / 2 - nodeH / 2 - (compact ? 12 : 18), yTop + 6);

        for (let i = 0; i < d; i++) {
          const fy = i / (d - 1);
          levY[i] = U.lerp(yTop, yLow, fy);
          encW[i] = wMax * (0.5 + 0.5 * (F[i + 1] / F[1])); // largeur ∝ bandes restantes
          encX[i] = U.lerp(inX + wMax / 2, W * 0.345, fy);
        }
        const decXi = (i) => W - encX[i];

        /* waypoints : entrée → encodeurs → bottleneck → décodeurs → sortie */
        const nSeg = 2 * d + 2;
        wpX[0] = inX + sw / 2; wpY[0] = specY + sh;
        for (let i = 0; i < d; i++) { wpX[i + 1] = encX[i]; wpY[i + 1] = levY[i]; }
        wpX[d + 1] = W / 2; wpY[d + 1] = yBN;
        for (let j = 0; j < d; j++) { const i = d - 1 - j; wpX[d + 2 + j] = decXi(i); wpY[d + 2 + j] = levY[i]; }
        wpX[nSeg] = outX + sw / 2; wpY[nSeg] = specY + sh;

        const u = (t % PERIOD) / PERIOD * nSeg;            // position du pulse, easée par segment
        const seg = Math.min(Math.floor(u), nSeg - 1);
        const ff = U.smoothstep(u - seg);
        const px = U.lerp(wpX[seg], wpX[seg + 1], ff);
        const py = U.lerp(wpY[seg], wpY[seg + 1], ff);
        const prox = (k) => U.clamp(1 - Math.abs(u - k) / 0.7, 0, 1);

        /* ---------- titre ---------- */
        U.text(ctx, compact ? `U-Net — ${d} étages` : `U-Net audio — ${d} étages, bottleneck récurrent`,
          W / 2, m + 6, { size: compact ? 11 : 13, bold: true, align: 'center', baseline: 'middle' });

        /* ---------- flèches de liaison (sous les nodes) ---------- */
        const aCol = palette.faint;
        U.arrow(ctx, wpX[0], wpY[0] + 2, encX[0], levY[0] - nodeH / 2, { color: aCol, alpha: 0.8 });
        for (let i = 0; i < d - 1; i++)
          U.arrow(ctx, encX[i], levY[i] + nodeH / 2, encX[i + 1], levY[i + 1] - nodeH / 2, { color: aCol, alpha: 0.8 });
        U.arrow(ctx, encX[d - 1], levY[d - 1] + nodeH / 2, W / 2 - bnW * 0.28, yBN - bnH / 2, { color: aCol, alpha: 0.8 });
        U.arrow(ctx, W / 2 + bnW * 0.28, yBN - bnH / 2, decXi(d - 1), levY[d - 1] + nodeH / 2, { color: aCol, alpha: 0.8 });
        for (let i = d - 1; i > 0; i--)
          U.arrow(ctx, decXi(i), levY[i] - nodeH / 2, decXi(i - 1), levY[i - 1] + nodeH / 2, { color: aCol, alpha: 0.8 });
        U.arrow(ctx, decXi(0), levY[0] - nodeH / 2, wpX[nSeg], wpY[nSeg] + 2, { color: aCol, alpha: 0.8 });

        /* ---------- skip connections : arcs pointillés teal + paquets rapides ---------- */
        const lift = compact ? 9 : 14;
        ctx.save();
        for (let i = 0; i < d; i++) {
          const x0 = encX[i] + encW[i] / 2 + 4, x1 = decXi(i) - encW[i] / 2 - 4, y = levY[i];
          ctx.strokeStyle = palette.voice;
          ctx.lineWidth = 1.3;
          ctx.globalAlpha = skipAlpha;
          ctx.setLineDash([5, 4]);
          ctx.lineDashOffset = -((t * 16) % 9);
          ctx.beginPath();
          ctx.moveTo(x0, y);
          ctx.quadraticCurveTo(W / 2, y - lift, x1, y);
          ctx.stroke();
          ctx.setLineDash([]);
          /* paquet : part quand le pulse quitte l'encodeur i, arrive AVANT lui au décodeur i */
          const q = u - (i + 1.25);
          if (skipAlpha > 0.15 && q > 0 && q < 1) {
            const e = U.ease(q);
            ctx.globalAlpha = 1;
            U.glowDot(ctx, qPt(x0, W / 2, x1, e), qPt(y, y - lift, y, e), compact ? 2.4 : 3, palette.voice);
          }
        }
        if (!compact && skipAlpha > 0.3) {
          ctx.globalAlpha = skipAlpha;
          U.text(ctx, 'skips : le détail contourne le goulot', W / 2, levY[0] - lift - 7,
            { size: 10, color: palette.voice, align: 'center' });
        }
        ctx.restore();

        /* ---------- nodes encodeur / décodeur / bottleneck ---------- */
        for (let i = 0; i < d; i++) {
          U.node(ctx, encX[i] - encW[i] / 2, levY[i] - nodeH / 2, encW[i], nodeH, {
            title: 'Conv ↓2', sub: `F=${F[i + 1]} C=${C[i + 1]}`,
            color: palette.blue, active: prox(i + 1) > 0.35, size: compact ? 11 : 12,
          });
          U.node(ctx, decXi(i) - encW[i] / 2, levY[i] - nodeH / 2, encW[i], nodeH, {
            title: 'TConv ↑2', sub: `F=${F[i]} C=${C[i]}`,
            color: palette.blue, active: prox(2 * d + 1 - i) > 0.35, size: compact ? 11 : 12,
          });
        }
        U.node(ctx, W / 2 - bnW / 2, yBN - bnH / 2, bnW, bnH, {
          title: 'GRU ×2', sub: `F=${F[d]} · T÷${tDiv} · C=${C[d]}`,
          color: palette.mix, active: true, fill: palette.panel2, size: compact ? 11 : 12,
        });
        if (!compact)
          U.text(ctx, `×${comp}`, W / 2 + bnW / 2 + 8, yBN + 4, { size: 12, bold: true, color: palette.mix });

        /* ---------- spectrogrammes entrée / sortie ---------- */
        scrIn.draw(ctx, inX, specY, sw, sh);
        U.frame(ctx, inX, specY, sw, sh, compact ? 'Entrée' : 'Entrée — |STFT|');
        scrOut.draw(ctx, outX, specY, sw, sh);
        U.frame(ctx, outX, specY, sw, sh, compact ? 'Sortie' : 'Sortie reconstruite');
        if (blurMix > 0.4) {
          ctx.save();
          ctx.globalAlpha = U.smoothstep(blurMix);
          ctx.strokeStyle = palette.red;
          ctx.lineWidth = 1.4;
          U.roundRect(ctx, outX, specY, sw, sh, 6);
          ctx.stroke();
          U.text(ctx, 'floue (moy. 5 bins)', outX + sw / 2, specY + sh + 12,
            { size: 10, color: palette.red, align: 'center' });
          ctx.restore();
        }
        if (!compact) {
          U.text(ctx, `F=${F[0]} · C=${C[0]}`, inX + sw + 6, specY + 10,
            { size: 9, color: palette.dim, mono: true });
          U.text(ctx, 'Encodeur — compression', inX, specY + sh + 26, { size: 10, color: palette.dim });
          U.text(ctx, 'Décodeur — reconstruction', W - m, specY + sh + 26,
            { size: 10, color: palette.dim, align: 'right' });
        }

        /* ---------- pulse principal (le QUOI, par le fond) + traîne ---------- */
        const u2 = Math.max(u - 0.22, 0);
        const s2 = Math.min(Math.floor(u2), nSeg - 1), f2 = U.smoothstep(u2 - s2);
        ctx.save();
        ctx.globalAlpha = 0.45;
        U.glowDot(ctx, U.lerp(wpX[s2], wpX[s2 + 1], f2), U.lerp(wpY[s2], wpY[s2 + 1], f2),
          compact ? 2.2 : 2.8, palette.rest);
        ctx.restore();
        U.glowDot(ctx, px, py, compact ? 3.5 : 4.5, palette.rest);

        /* ---------- readout (nombres réellement calculés) ---------- */
        const ry = H - (compact ? 8 : 12), rs = compact ? 10 : 11.5;
        if (blurMix < 0.5) {
          readout(ctx, compact ? [
            { s: `×${comp} au fond — `, c: palette.dim },
            { s: 'QUOI', c: palette.rest, b: true },
            { s: ' par le fond, ', c: palette.dim },
            { s: 'OÙ', c: palette.voice, b: true },
            { s: ' par les skips', c: palette.dim },
          ] : [
            { s: `compression ×${comp} au bottleneck — le `, c: palette.dim },
            { s: 'QUOI', c: palette.rest, b: true },
            { s: ' passe par le fond, le ', c: palette.dim },
            { s: 'OÙ', c: palette.voice, b: true },
            { s: ' passe par les skips', c: palette.dim },
          ], W / 2, ry, rs);
        } else {
          readout(ctx, compact ? [
            { s: 'sans skips : ', c: palette.dim },
            { s: 'sortie floue', c: palette.red, b: true },
            { s: ' (moy. 5 bins)', c: palette.dim },
          ] : [
            { s: 'sans skips : ', c: palette.dim },
            { s: 'sortie floue', c: palette.red, b: true },
            { s: ` — le OÙ précis se perd dans la compression ×${comp}`, c: palette.dim },
          ], W / 2, ry, rs);
        }
      });
    },
  });
})();
