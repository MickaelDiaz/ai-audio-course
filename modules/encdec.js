/* =====================================================================
   Audio AI Atlas — module « Encoder–Decoder & U-Net »
   Diagramme en U animé : encodeur (compression), bottleneck récurrent,
   décodeur (reconstruction), skip connections. Démo clé : sans skips,
   la sortie devient spectralement floue (moyenne sur 5 bins).

   Mise en page responsive : sur mobile (stage.compact), le U est dessiné
   en GRAND dans la hauteur disponible (spectros en haut, encodeur à
   gauche, décodeur à droite, bottleneck en bas), toutes les polices
   passent par stage.fs(...) et rien n'est masqué (axes, ×compression,
   légendes, readout). Sur desktop (≥ 560), disposition d'origine.
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
      <p>L'architecture <dfn class="term" data-term="encoder-decoder"><strong>encodeur–décodeur</strong></dfn> (et sa variante <strong>U-Net</strong>) domine
      le traitement audio neuronal — DeepFilterNet (débruitage), Demucs (séparation de sources), la plupart
      des modèles de speech enhancement. L'encodeur empile des <dfn class="term" data-term="convolution">convolutions</dfn> à <dfn class="term" data-term="stride"><em>stride&nbsp;2</em></dfn> :
      à chaque étage, la résolution temps–fréquence est divisée par 2 dans chaque dimension pendant que le
      nombre de <dfn class="term" data-term="channel">canaux</dfn> augmente (F=256→128→64→32, C=2→16→32→64). Au <dfn class="term" data-term="bottleneck"><strong>bottleneck</strong></dfn>, chaque
      « pixel » résume une grande région du <dfn class="term" data-term="spectrogram">spectrogramme</dfn> : le réseau y encode le <strong>QUOI</strong> —
      présence de voix, timbre, <dfn class="term" data-term="phoneme">phonème</dfn> — mais il a perdu le <strong>OÙ</strong> précis (le <dfn class="term" data-term="fft-bin">bin</dfn> exact,
      la micro-transition).</p>
      <p>Placer un module récurrent (<dfn class="term" data-term="gru"><code>GRU ×2</code></dfn>) au fond du U est un choix d'efficacité : la
      <dfn class="term" data-term="rnn">récurrence</dfn>, séquentielle par nature, coûte cher à chaque pas de temps. Au bottleneck, le temps est
      compressé ×8 et la fréquence ×8 — c'est l'endroit le moins cher pour modéliser la
      <strong>dynamique temporelle longue</strong> (prosodie, attaques et tenues, continuité de la voix),
      exactement la recette des <dfn class="term" data-term="crn">CRN</dfn> (<em>convolutional recurrent networks</em>) comme DeepFilterNet.</p>
      <p>Les <dfn class="term" data-term="skip-connection"><strong>skip connections</strong></dfn> copient les activations de chaque étage d'encodeur vers
      l'étage symétrique du décodeur (concaténation). Le détail spectral fin court-circuite ainsi le
      bottleneck : c'est par là que passe le <strong>OÙ</strong>. Bonus : ce sont aussi des autoroutes de
      <dfn class="term" data-term="gradient-descent">gradient</dfn> qui stabilisent l'entraînement profond. Coupez le toggle pour le constater — sans skips, le
      décodeur ne peut reconstruire qu'une version lissée du spectre (ici, une vraie moyenne glissante sur
      5 bins), fidèle à la perte réelle de localisation fréquentielle.</p>
      <p>C'est crucial pour le <dfn class="term" data-term="masking"><strong>masking</strong></dfn> : débruiter ou séparer revient à prédire un masque
      appliqué bin par bin au spectrogramme d'entrée. Sans la précision portée par les skips, le masque
      « baverait » sur les <dfn class="term" data-term="harmonique">harmoniques</dfn> voisines — la reconstruction fine à pleine résolution est précisément
      ce que l'encodeur–décodeur à skips garantit.</p>`,

    init(stage) {
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

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
          blurC[i] = n > 0 ? s / n : sharp[i];
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

      /* =================================================================
         Helpers de dessin partagés par les deux mises en page.
         Toute la GÉOMÉTRIE est passée en argument (objet g) : seules les
         positions / tailles changent entre compact et desktop.
         ================================================================= */

      /* g = {
           inX, outX, specY, sw, sh,          // spectrogrammes
           encX[], decX[], levY[], encW[], nodeH,
           bnX, bnY, bnW, bnH,                // bottleneck
           wpX/wpY (waypoints), nSeg, d,
           F[], C[], tDiv, comp, lift, dotR, nodeFs,
           showAxes, showCaptions, showCompChip, showSkipLabel
         } */

      /* ---------- flèches de liaison (squelette du U) ---------- */
      function drawArrows(ctx, g) {
        const aCol = palette.faint, nh = g.nodeH;
        U.arrow(ctx, g.wpX[0], g.wpY[0] + 2, g.encX[0], g.levY[0] - nh / 2, { color: aCol, alpha: 0.8 });
        for (let i = 0; i < g.d - 1; i++)
          U.arrow(ctx, g.encX[i], g.levY[i] + nh / 2, g.encX[i + 1], g.levY[i + 1] - nh / 2, { color: aCol, alpha: 0.8 });
        U.arrow(ctx, g.encX[g.d - 1], g.levY[g.d - 1] + nh / 2, g.bnX - g.bnW * 0.28, g.bnY - g.bnH / 2, { color: aCol, alpha: 0.8 });
        U.arrow(ctx, g.bnX + g.bnW * 0.28, g.bnY - g.bnH / 2, g.decX[g.d - 1], g.levY[g.d - 1] + nh / 2, { color: aCol, alpha: 0.8 });
        for (let i = g.d - 1; i > 0; i--)
          U.arrow(ctx, g.decX[i], g.levY[i] - nh / 2, g.decX[i - 1], g.levY[i - 1] + nh / 2, { color: aCol, alpha: 0.8 });
        U.arrow(ctx, g.decX[0], g.levY[0] - nh / 2, g.wpX[g.nSeg], g.wpY[g.nSeg] + 2, { color: aCol, alpha: 0.8 });
      }

      /* ---------- skip connections : arcs pointillés + paquets ---------- */
      function drawSkips(ctx, g, t, u, skipAlpha) {
        ctx.save();
        const midX = g.bnX;
        for (let i = 0; i < g.d; i++) {
          const x0 = g.encX[i] + g.encW[i] / 2 + 4, x1 = g.decX[i] - g.encW[i] / 2 - 4, y = g.levY[i];
          ctx.strokeStyle = palette.voice;
          ctx.lineWidth = 1.3;
          ctx.globalAlpha = skipAlpha;
          ctx.setLineDash([5, 4]);
          ctx.lineDashOffset = -((t * 16) % 9);
          ctx.beginPath();
          ctx.moveTo(x0, y);
          ctx.quadraticCurveTo(midX, y - g.lift, x1, y);
          ctx.stroke();
          ctx.setLineDash([]);
          const q = u - (i + 1.25);
          if (skipAlpha > 0.15 && q > 0 && q < 1) {
            const e = U.ease(q);
            ctx.globalAlpha = 1;
            U.glowDot(ctx, qPt(x0, midX, x1, e), qPt(y, y - g.lift, y, e), g.dotR.skip, palette.voice);
          }
        }
        if (g.showSkipLabel && skipAlpha > 0.3) {
          ctx.globalAlpha = skipAlpha;
          U.text(ctx, 'skips : le détail contourne le goulot', midX, g.levY[0] - g.lift - fs(7),
            { size: g.skipLabelFs, color: palette.voice, align: 'center' });
        }
        ctx.restore();
      }

      /* ---------- nodes encodeur / décodeur / bottleneck ---------- */
      function drawNodes(ctx, g, prox) {
        const nh = g.nodeH;
        for (let i = 0; i < g.d; i++) {
          U.node(ctx, g.encX[i] - g.encW[i] / 2, g.levY[i] - nh / 2, g.encW[i], nh, {
            title: 'Conv ↓2', sub: `F=${g.F[i + 1]} C=${g.C[i + 1]}`,
            color: palette.blue, active: prox(i + 1) > 0.35, size: g.nodeFs,
          });
          U.node(ctx, g.decX[i] - g.encW[i] / 2, g.levY[i] - nh / 2, g.encW[i], nh, {
            title: 'TConv ↑2', sub: `F=${g.F[i]} C=${g.C[i]}`,
            color: palette.blue, active: prox(2 * g.d + 1 - i) > 0.35, size: g.nodeFs,
          });
        }
        U.node(ctx, g.bnX - g.bnW / 2, g.bnY - g.bnH / 2, g.bnW, g.bnH, {
          title: 'GRU ×2', sub: `F=${g.F[g.d]} · T÷${g.tDiv} · C=${g.C[g.d]}`,
          color: palette.mix, active: true, fill: palette.panel2, size: g.nodeFs,
        });
        if (g.showCompChip) {
          U.text(ctx, `×${g.comp}`, g.bnX + g.bnW / 2 + 8, g.bnY + 4,
            { size: g.compFs, bold: true, color: palette.mix });
        }
      }

      /* ---------- spectrogrammes entrée / sortie ---------- */
      function drawSpectros(ctx, g, blurMix) {
        scrIn.draw(ctx, g.inX, g.specY, g.sw, g.sh);
        U.frame(ctx, g.inX, g.specY, g.sw, g.sh, g.inLabel);
        scrOut.draw(ctx, g.outX, g.specY, g.sw, g.sh);
        U.frame(ctx, g.outX, g.specY, g.sw, g.sh, g.outLabel);
        if (blurMix > 0.4) {
          ctx.save();
          ctx.globalAlpha = U.smoothstep(blurMix);
          ctx.strokeStyle = palette.red;
          ctx.lineWidth = 1.4;
          U.roundRect(ctx, g.outX, g.specY, g.sw, g.sh, 6);
          ctx.stroke();
          U.text(ctx, 'floue (moy. 5 bins)', g.outX + g.sw / 2, g.specY + g.sh + fs(12),
            { size: g.specCapFs, color: palette.red, align: 'center' });
          ctx.restore();
        }
      }

      stage.onFrame((t, dt) => {
        const ctx = stage.ctx, W = stage.W, H = stage.H;
        stage.clear();
        const compact = stage.compact;
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

        /* objet géométrie partagé (rempli différemment selon compact) */
        const g = {
          d, F, C, tDiv, comp,
          wpX, wpY, encX, encW, levY,
          decX: [],
        };

        if (compact) {
          /* ===================================================================
             MOBILE — le U dessiné en GRAND dans la hauteur portrait.
             Spectros côte à côte en haut, encodeur descend à gauche, décodeur
             remonte à droite, bottleneck en bas. Tout passe par fs(...).
             =================================================================== */
          const m = 12;
          const titleY = fs(15);
          /* spectrogrammes : larges (≈ 40 % de la largeur chacun) */
          const sw = U.clamp((W - 2 * m - 14) / 2, 96, 168);
          const sh = U.clamp(sw * 0.6, 40, 110);
          const specY = titleY + fs(16);
          const inX = m, outX = W - m - sw;

          /* nodes : généreux verticalement, on a de la place.
             Largeur plafonnée pour que l'encodeur (gauche) et le décodeur
             (droite, miroir) ne se chevauchent jamais : le node le plus large
             a sa moitié droite à encXmax + wMax/2, qui doit rester < W/2. */
          const nodeH = U.clamp(fs(30), 30, 48);
          const nodeFs = fs(11.5);
          const encXmax = W * 0.30;                       // centre du node le plus bas/large
          const wMax = U.clamp(2 * (W / 2 - encXmax - 6), 60, 150);
          const bnW = U.clamp(W * 0.46, 110, 220), bnH = U.clamp(fs(34), 34, 52);

          /* readout occupe le bas ; le bottleneck juste au-dessus */
          const readoutH = fs(34);
          const bnY = H - readoutH - bnH / 2 - fs(10);
          /* première rangée d'étages sous les spectros (laisse place aux légendes) */
          const yTop = specY + sh + fs(34);
          /* dernière rangée juste au-dessus du bottleneck */
          const yLow = Math.max(bnY - bnH / 2 - nodeH / 2 - fs(14), yTop + fs(8));

          for (let i = 0; i < d; i++) {
            const fy = d > 1 ? i / (d - 1) : 0;
            levY[i] = U.lerp(yTop, yLow, fy);
            encW[i] = wMax * (0.55 + 0.45 * (F[i + 1] / F[1]));
            /* encodeur : du bord gauche vers le centre-gauche en descendant */
            encX[i] = U.lerp(inX + wMax * 0.55, W * 0.30, fy);
          }
          for (let i = 0; i < d; i++) g.decX.push(W - encX[i]);

          Object.assign(g, {
            inX, outX, specY, sw, sh,
            nodeH, nodeFs, wMax,
            bnX: W / 2, bnY, bnW, bnH,
            lift: fs(13), nSeg: 2 * d + 2,
            dotR: { skip: fs(3), trail: fs(2.6), pulse: fs(4) },
            compFs: fs(12), skipLabelFs: fs(10.5), specCapFs: fs(11),
            inLabel: 'Entrée', outLabel: 'Sortie',
            showAxes: true, showCaptions: true, showCompChip: false, showSkipLabel: false,
          });

          /* titre */
          U.text(ctx, `U-Net — ${d} étages`, W / 2, titleY,
            { size: fs(14), bold: true, align: 'center', baseline: 'middle' });
        } else {
          /* ===================================================================
             DESKTOP — disposition d'origine (inchangée).
             =================================================================== */
          const m = 16;
          const sw = U.clamp(W * 0.2, 72, 140), sh = sw * 0.6;
          const specY = m + 17;
          const inX = m, outX = W - m - sw;
          const nodeH = 34;
          const wMax = U.clamp(W * 0.21, 68, 150);
          const bnW = U.clamp(W * 0.17, 80, 130), bnH = 38;
          const bnY = H - 32 - bnH / 2;
          const yTop = specY + sh + 44;
          const yLow = Math.max(bnY - bnH / 2 - nodeH / 2 - 18, yTop + 6);

          for (let i = 0; i < d; i++) {
            const fy = d > 1 ? i / (d - 1) : 0;
            levY[i] = U.lerp(yTop, yLow, fy);
            encW[i] = wMax * (0.5 + 0.5 * (F[i + 1] / F[1]));
            encX[i] = U.lerp(inX + wMax / 2, W * 0.345, fy);
          }
          for (let i = 0; i < d; i++) g.decX.push(W - encX[i]);

          Object.assign(g, {
            inX, outX, specY, sw, sh,
            nodeH, nodeFs: 12, wMax,
            bnX: W / 2, bnY, bnW, bnH,
            lift: 14, nSeg: 2 * d + 2,
            dotR: { skip: 3, trail: 2.8, pulse: 4.5 },
            compFs: 12, skipLabelFs: 10, specCapFs: 10,
            inLabel: 'Entrée — |STFT|', outLabel: 'Sortie reconstruite',
            showAxes: false, showCaptions: false, showCompChip: true, showSkipLabel: true,
          });

          U.text(ctx, `U-Net audio — ${d} étages, bottleneck récurrent`,
            W / 2, m + 6, { size: 13, bold: true, align: 'center', baseline: 'middle' });
        }

        /* ---------- waypoints : entrée → encodeurs → bottleneck → décodeurs → sortie ---------- */
        const nSeg = g.nSeg;
        wpX[0] = g.inX + g.sw / 2; wpY[0] = g.specY + g.sh;
        for (let i = 0; i < d; i++) { wpX[i + 1] = encX[i]; wpY[i + 1] = levY[i]; }
        wpX[d + 1] = g.bnX; wpY[d + 1] = g.bnY;
        for (let j = 0; j < d; j++) { const i = d - 1 - j; wpX[d + 2 + j] = g.decX[i]; wpY[d + 2 + j] = levY[i]; }
        wpX[nSeg] = g.outX + g.sw / 2; wpY[nSeg] = g.specY + g.sh;

        const u = (t % PERIOD) / PERIOD * nSeg;            // position du pulse, easée par segment
        const seg = U.clamp(Math.floor(u), 0, nSeg - 1);
        const ff = U.smoothstep(u - seg);
        const px = U.lerp(wpX[seg], wpX[seg + 1], ff);
        const py = U.lerp(wpY[seg], wpY[seg + 1], ff);
        const prox = (k) => U.clamp(1 - Math.abs(u - k) / 0.7, 0, 1);

        /* ---------- squelette + skips + nodes + spectros ---------- */
        drawArrows(ctx, g);
        drawSkips(ctx, g, t, u, skipAlpha);
        drawNodes(ctx, g, prox);
        drawSpectros(ctx, g, blurMix);

        /* ---------- légendes / axes ---------- */
        if (g.showCaptions) {
          /* mobile : légendes en GRAND sous les spectros (rien de minuscule).
             Les étiquettes de dimensions sont placées hors des nodes :
             « F=…·C=… » sous le spectrogramme d'entrée (mesurée pour ne pas
             empiéter sur « Encodeur ↓ »), « ×… au fond » sous le bottleneck. */
          const capY = g.specY + g.sh + fs(15);
          U.text(ctx, 'Encodeur ↓', g.inX, capY,
            { size: fs(11), color: palette.blue, bold: true });
          U.text(ctx, 'Décodeur ↑', W - 12, capY,
            { size: fs(11), color: palette.blue, bold: true, align: 'right' });
          /* dimensions d'entrée : sous le spectro d'entrée, alignées à droite sur
             son bord ; police réduite si elle risque de toucher « Encodeur ↓ » */
          const dimStr = `F=${F[0]}·C=${C[0]}`;
          ctx.font = `${fs(10)}px ${U.MONO}`;
          const dimW = ctx.measureText(dimStr).width;
          ctx.font = `600 ${fs(11)}px ${U.FONT}`;
          const encW0 = ctx.measureText('Encodeur ↓').width;
          const dimRight = g.inX + g.sw;
          const avail = Math.max(20, dimRight - (g.inX + encW0 + fs(8)));
          const dimFs = dimW > avail ? Math.max(fs(8), fs(10) * avail / dimW) : fs(10);
          U.text(ctx, dimStr, dimRight, capY,
            { size: dimFs, color: palette.dim, mono: true, align: 'right' });
          /* compression au fond : zone libre juste sous le node bottleneck */
          U.text(ctx, `×${comp} au fond`, g.bnX, g.bnY + g.bnH / 2 + fs(12),
            { size: fs(10), color: palette.mix, mono: true, align: 'center' });
        } else {
          U.text(ctx, `F=${F[0]} · C=${C[0]}`, g.inX + g.sw + 6, g.specY + 10,
            { size: 9, color: palette.dim, mono: true });
          U.text(ctx, 'Encodeur — compression', g.inX, g.specY + g.sh + 26, { size: 10, color: palette.dim });
          U.text(ctx, 'Décodeur — reconstruction', W - (compact ? 12 : 16), g.specY + g.sh + 26,
            { size: 10, color: palette.dim, align: 'right' });
        }

        /* ---------- pulse principal (le QUOI, par le fond) + traîne ---------- */
        const u2 = Math.max(u - 0.22, 0);
        const s2 = U.clamp(Math.floor(u2), 0, nSeg - 1), f2 = U.smoothstep(u2 - s2);
        ctx.save();
        ctx.globalAlpha = 0.45;
        U.glowDot(ctx, U.lerp(wpX[s2], wpX[s2 + 1], f2), U.lerp(wpY[s2], wpY[s2 + 1], f2),
          g.dotR.trail, palette.rest);
        ctx.restore();
        U.glowDot(ctx, px, py, g.dotR.pulse, palette.rest);

        /* ---------- readout (nombres réellement calculés) ---------- */
        const ry = H - (compact ? fs(10) : 12), rs = compact ? fs(11.5) : 11.5;
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
