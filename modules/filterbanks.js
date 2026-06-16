/* ============================================================
   Audio AI Atlas — « Mel & ERB — l'oreille comme prior »
   Banque de filtres perceptuelle appliquée à un VRAI spectre FFT :
   512 bins linéaires compressés en N bandes (Mel / ERB / linéaire),
   énergies = vraie multiplication banque × spectre.
   Mise en page responsive : sur mobile (stage.compact), tout est empilé
   verticalement et agrandi via stage.fs(...) ; rien n'est masqué.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  /* ---------- Constantes signal ---------- */
  const SR = 16000;          // sr 16 kHz → axe 0–8 kHz
  const NFFT = 1024;         // FFT 1024 → 512 bins
  const BINS = NFFT / 2;     // 512
  const NYQ = SR / 2;        // 8000 Hz
  const MAXB = 64;           // nombre max de bandes
  const PERIOD = 1.5;        // s — cycle d'illumination d'une bande

  /* ---------- Échelles perceptuelles (formules vraies) ---------- */
  const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
  const melToHz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
  const hzToErb = (f) => 21.4 * Math.log10(1 + 0.00437 * f);
  const erbToHz = (e) => (Math.pow(10, e / 21.4) - 1) / 0.00437;
  const ident = (f) => f;

  AtlasRegister({
    id: 'filterbanks',
    title: 'Mel & ERB — l’oreille comme prior',
    category: 'signal',
    icon: '⩙',
    summary: 'Compresser 512 bins FFT en quelques bandes perceptuelles : Mel, ERB, et pourquoi les réseaux y gagnent.',
    explain: `
      <p>L'oreille n'analyse pas le son sur une grille linéaire : la <dfn class="term" data-term="cochlee">cochlée</dfn>
      déroule les <dfn class="term" data-term="frequence">fréquences</dfn> le long de la membrane basilaire avec une résolution très fine dans les graves
      et de plus en plus grossière vers les aigus. Les échelles <dfn class="term" data-term="mel">Mel</dfn>
      (<code>mel = 2595·log₁₀(1 + f/700)</code>) et <dfn class="term" data-term="erb">ERB</dfn>
      (<em>Equivalent Rectangular Bandwidth</em>, <code>≈ 21,4·log₁₀(1 + 0,00437·f)</code>)
      modélisent cette courbe : des <dfn class="term" data-term="filtre">filtres</dfn> serrés sous 1&nbsp;kHz, de plus en plus larges au-dessus.
      Ici le <dfn class="term" data-term="spectre">spectre</dfn> est calculé à 16&nbsp;kHz (axe 0–8&nbsp;kHz) ; les systèmes <dfn class="term" data-term="full-band">full-band</dfn>
      appliquent exactement la même logique à 48&nbsp;kHz (axe 0–24&nbsp;kHz).</p>
      <p>Pourquoi compresser 512 <dfn class="term" data-term="fft-bin">bins</dfn> en N bandes ? Parce qu'un réseau qui prédit un
      <dfn class="term" data-term="gain-par-bande">gain par bande</dfn> n'a plus que N sorties au lieu de 512 : moins de <dfn class="term" data-term="parameters">paramètres</dfn>,
      moins de variance d'estimation, et une cible alignée sur ce que l'oreille perçoit réellement.
      Chaque énergie de bande est une <dfn class="term" data-term="dot-product">somme pondérée</dfn> de dizaines de bins voisins —
      la <dfn class="term" data-term="filterbank">banque de filtres</dfn> agit comme un lissage structuré qui <strong>stabilise l'apprentissage</strong> :
      une erreur sur une bande large coûte peu perceptivement, alors qu'un gain erratique bin par
      bin produit des artefacts musicaux audibles.</p>
      <p>C'est le pattern récurrent « <strong>gains grossiers perceptuels + raffinement fin
      ailleurs</strong> » : <strong>DeepFilterNet</strong> prédit 32 gains ERB pour sculpter
      l'<dfn class="term" data-term="enveloppe-spectrale">enveloppe spectrale</dfn> sur tout le spectre, puis un étage de <dfn class="term" data-term="deep-filtering">deep filtering</dfn> raffine la
      structure fine des seules basses fréquences (<dfn class="term" data-term="harmonique">harmoniques</dfn> de la voix), là où l'oreille exige de
      la précision. On dépense les paramètres là où ils comptent, et on laisse le <dfn class="term" data-term="prior-perceptuel">prior perceptuel</dfn>
      faire le reste.</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

      /* ---------- Contrôles ---------- */
      const selSrc = stage.addSelect({
        label: 'Source',
        options: [
          { value: 'speech', label: 'Parole' },
          { value: 'music', label: 'Musique' },
          { value: 'mix', label: 'Mix' },
        ],
        value: 'speech',
      });
      const selScale = stage.addSelect({
        label: 'Échelle',
        options: [
          { value: 'mel', label: 'Mel (triangles)' },
          { value: 'erb', label: 'ERB (arrondies)' },
          { value: 'lin', label: 'Linéaire (uniformes)' },
        ],
        value: 'erb',
      });
      const slBands = stage.addSlider({
        label: 'Bandes',
        min: 8, max: 64, step: 1, value: 32,
        format: (v) => v + ' bandes',
      });

      /* ---------- Buffers pré-alloués (jamais dans onFrame) ---------- */
      const sig = new Float32Array(NFFT);          // fenêtre temporelle
      const spec = new Float32Array(BINS);         // magnitudes lissées
      const dispSpec = new Float32Array(BINS);     // spectre normalisé affiché
      const weights = new Float32Array(MAXB * BINS); // banque de filtres
      const loBin = new Int32Array(MAXB);          // support de chaque filtre
      const hiBin = new Int32Array(MAXB);
      const fLo = new Float32Array(MAXB);          // bords / centre en Hz (vrais)
      const fC = new Float32Array(MAXB);
      const fHi = new Float32Array(MAXB);
      const energy = new Float32Array(MAXB);       // banque × spectre (vraie)
      const eSmooth = new Float32Array(MAXB);
      const dispBand = new Float32Array(MAXB);
      let fbKey = '';                              // cache (échelle, n)
      let specMax = 1e-3, eMax = 1e-3;             // auto-gain d'affichage

      /* ---------- Construction de la banque (au changement seulement) ---- */
      function buildBank(scale, n) {
        const warp = scale === 'mel' ? hzToMel : scale === 'erb' ? hzToErb : ident;
        const unwarp = scale === 'mel' ? melToHz : scale === 'erb' ? erbToHz : ident;
        const wMax = warp(NYQ);
        weights.fill(0);
        for (let i = 0; i < n; i++) {
          const c0 = (i / (n + 1)) * wMax;       // bord gauche (échelle warpée)
          const c1 = ((i + 1) / (n + 1)) * wMax; // centre
          const c2 = ((i + 2) / (n + 1)) * wMax; // bord droit
          fLo[i] = unwarp(c0); fC[i] = unwarp(c1); fHi[i] = unwarp(c2);
          let lo = -1, hi = -1;
          for (let b = 0; b < BINS; b++) {
            const w = warp((b * SR) / NFFT);     // fréquence vraie du bin b
            if (w <= c0 || w >= c2) continue;
            const x = w < c1 ? (w - c0) / (c1 - c0) : (c2 - w) / (c2 - c1);
            // Mel & linéaire : triangles ; ERB : bandes arrondies (smoothstep)
            weights[i * BINS + b] = scale === 'erb' ? U.smoothstep(x) : x;
            if (lo < 0) lo = b;
            hi = b;
          }
          if (lo < 0) { // filtre plus étroit qu'un bin : rattacher le bin le plus proche
            lo = hi = U.clamp(Math.round((fC[i] / NYQ) * BINS), 0, BINS - 1);
            weights[i * BINS + lo] = 1;
          }
          loBin[i] = lo; hiBin[i] = hi;
        }
      }

      /* ============================================================
         HELPERS DE DESSIN — partagés entre mobile et desktop.
         Ils ne supposent jamais un ratio ; tout vient des arguments.
         ============================================================ */

      /* ---- Panneau du haut : spectre linéaire + banque de filtres superposée ---- */
      function drawSpectrum(box, env, sizes) {
        const { x, y, w, h } = box;
        const { srcColor, srcName, scale, n, k, flash } = env;
        const axisY = y + h;
        const xOfBin = (b) => x + ((b + 0.5) / BINS) * w;

        U.frame(ctx, x, y, w, h);
        U.text(ctx, sizes.compact ? 'Spectre — 512 bins'
          : 'Spectre linéaire — 512 bins (0–8 kHz, sr 16 kHz)',
          x + 8, y - 6, { size: sizes.label, color: palette.dim });
        U.text(ctx, 'source : ' + srcName, x + w - 2, y - 6,
          { size: sizes.src, color: srcColor, align: 'right', bold: true });

        U.bars(ctx, dispSpec, x, y, w, h, { color: srcColor, alpha: 0.38, gap: 0 });

        // formule d'espacement vraie, affichée (lisible aussi sur mobile)
        const formula = scale === 'mel' ? 'mel = 2595·log₁₀(1 + f/700)'
          : scale === 'erb' ? 'ERB ≈ 21,4·log₁₀(1 + 0,00437·f)'
          : 'espacement linéaire uniforme';
        U.text(ctx, formula, x + w - 8, y + sizes.formula + 4,
          { size: sizes.formula, color: palette.faint, align: 'right', mono: true });

        // Filtres colorés (viridis), remplis + arête supérieure
        const hCurve = h * 0.92;
        for (let i = 0; i < n; i++) {
          const off = i * BINS;
          const col = U.viridis(i / Math.max(n - 1, 1));
          const isK = i === k;
          ctx.beginPath();
          ctx.moveTo(xOfBin(loBin[i]), axisY);
          for (let b = loBin[i]; b <= hiBin[i]; b++) {
            ctx.lineTo(xOfBin(b), axisY - weights[off + b] * hCurve);
          }
          ctx.lineTo(xOfBin(hiBin[i]), axisY);
          ctx.closePath();
          ctx.fillStyle = col;
          ctx.globalAlpha = 0.35 * (isK ? 1 + 0.9 * flash : 1);
          ctx.fill();
          ctx.strokeStyle = col;
          ctx.globalAlpha = isK ? 0.55 + 0.45 * flash : 0.55;
          ctx.lineWidth = isK ? 1 + flash : 1;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;

        // Axe des fréquences (Hz), annoté
        ctx.strokeStyle = palette.grid;
        ctx.lineWidth = 1;
        const tickStep = sizes.compact || w < 360 ? 2000 : 1000;
        for (let f = 0; f <= NYQ; f += tickStep) {
          const tx = x + (f / NYQ) * w;
          ctx.beginPath(); ctx.moveTo(tx, axisY); ctx.lineTo(tx, axisY + 4); ctx.stroke();
          U.text(ctx, U.fmt.hz(f), tx, axisY + sizes.axis + 5, {
            size: sizes.axis, color: palette.faint, mono: true,
            align: f === 0 ? 'left' : f === NYQ ? 'right' : 'center',
          });
        }
        return { xOfBin, axisY };
      }

      /* ---- Panneau du bas : énergies par bande (banque × spectre) ---- */
      function drawBands(box, env, sizes) {
        const { x, y, w, h } = box;
        const { n, k, flash, colK } = env;
        U.frame(ctx, x, y, w, h);
        U.text(ctx, sizes.compact ? 'Énergies par bande' : 'Énergies par bande — banque × spectre',
          x + 8, y - 6, { size: sizes.label, color: palette.dim });
        const ratio = (BINS / n).toFixed(1).replace(/\.0$/, '');
        const headTxt = `512 bins → ${n} bandes (÷${ratio})`;
        if (sizes.compact) {
          // Mobile : la mesure de bins occupe SA PROPRE ligne, au-dessus du titre,
          // pour ne jamais chevaucher « Énergies par bande ». Réduite si trop large.
          let headSz = sizes.head;
          ctx.font = headSz + 'px ' + U.MONO;
          const avail = Math.max(w - 16, 1);
          const tw = ctx.measureText(headTxt).width;
          if (tw > avail) headSz = Math.max(headSz * avail / tw, fs(8));
          U.text(ctx, headTxt, x + w - 2, y - 6 - sizes.label - fs(4),
            { size: headSz, color: palette.text, align: 'right', mono: true });
        } else {
          U.text(ctx, headTxt, x + w - 2, y - 6,
            { size: sizes.head, color: palette.text, align: 'right', mono: true });
        }

        const bw = w / n;
        for (let i = 0; i < n; i++) {
          const v = dispBand[i];
          const bh = Math.max(v * (h - 4), 1);
          const bx = x + i * bw;
          const isK = i === k;
          ctx.save();
          if (isK && flash > 0.05) { ctx.shadowColor = colK; ctx.shadowBlur = 10 * flash; }
          ctx.fillStyle = U.viridis(i / Math.max(n - 1, 1));
          ctx.globalAlpha = isK ? 0.7 + 0.3 * flash : 0.8;
          ctx.fillRect(bx + 1, y + h - 2 - bh, Math.max(bw - 2, 1), bh);
          ctx.restore();
        }
        ctx.globalAlpha = 1;
        return { bandX: (i) => x + ((i + 0.5) / n) * w, bandTop: (i) => y + h * (1 - dispBand[i]) };
      }

      /* ---- Lignes de compression : bins du spectre → bande active ---- */
      function drawCompression(env, spc, bnd, sizes) {
        const { k, n, flash, grow, colK } = env;
        const lo = loBin[k], hi = hiBin[k];
        const x2 = bnd.bandX(k);
        const y2 = bnd.bandTop(k);
        if (flash <= 0.02) return;
        const nLines = Math.min(9, hi - lo + 1);
        if (nLines < 1) return;
        ctx.strokeStyle = colK;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.55 * flash;
        for (let j = 0; j < nLines; j++) {
          const b = nLines === 1 ? lo : lo + Math.round((j * (hi - lo)) / (nLines - 1));
          const x1 = spc.xOfBin(b);
          const y1 = spc.specY + spc.specH * (1 - dispSpec[b]);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(U.lerp(x1, x2, grow), U.lerp(y1, y2, grow));
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        if (grow > 0.95) U.glowDot(ctx, x2, y2, 2.5 + 1.5 * flash, colK);
      }

      /* ---------- Boucle de rendu ---------- */
      stage.onFrame((t, dt) => {
        stage.clear();
        const W = stage.W, H = stage.H;
        const compact = stage.compact;

        /* --- Lecture des contrôles + reconstruction éventuelle --- */
        const scale = selScale.value;
        const n = U.clamp(Math.round(slBands.value), 8, MAXB);
        const key = scale + ':' + n;
        if (key !== fbKey) { buildBank(scale, n); fbKey = key; }

        const srcKey = selSrc.value;
        const srcColor = srcKey === 'speech' ? palette.voice
          : srcKey === 'music' ? palette.rest : palette.mix;
        const srcName = srcKey === 'speech' ? 'Parole'
          : srcKey === 'music' ? 'Musique' : 'Mix';

        /* --- Signal réel → FFT réelle (512 magnitudes vraies) --- */
        for (let i = 0; i < NFFT; i++) {
          const ts = t + i / SR;
          sig[i] = srcKey === 'speech' ? U.gen.speech(ts)
            : srcKey === 'music' ? U.gen.music(ts)
            : 0.6 * U.gen.speech(ts) + 0.6 * U.gen.music(ts);
        }
        const mag = U.rfftMag(sig);

        /* --- Lissage temporel + auto-gain --- */
        const kS = 1 - Math.exp(-dt * 10);
        let mMax = 1e-4;
        for (let b = 0; b < BINS; b++) {
          spec[b] += (mag[b] - spec[b]) * kS;
          if (spec[b] > mMax) mMax = spec[b];
        }
        specMax = Math.max(specMax * (1 - 0.5 * dt), mMax, 1e-4);
        for (let b = 0; b < BINS; b++) {
          dispSpec[b] = Math.pow(U.clamp(spec[b] / specMax, 0, 1), 0.62);
        }

        /* --- Énergies par bande : VRAIE somme pondérée banque × spectre --- */
        let eM = 1e-5;
        for (let i = 0; i < n; i++) {
          let e = 0;
          const off = i * BINS;
          for (let b = loBin[i]; b <= hiBin[i]; b++) e += weights[off + b] * spec[b];
          energy[i] = e;
          eSmooth[i] += (e - eSmooth[i]) * kS;
          if (eSmooth[i] > eM) eM = eSmooth[i];
        }
        eMax = Math.max(eMax * (1 - 0.5 * dt), eM, 1e-5);
        for (let i = 0; i < n; i++) {
          dispBand[i] = Math.pow(U.clamp(eSmooth[i] / eMax, 0, 1), 0.7);
        }

        /* --- Cycle d'illumination : une bande toutes les ~1,5 s, easée --- */
        const cyc = Math.floor(t / PERIOD);
        const k = ((cyc % n) + n) % n;
        const p = t / PERIOD - cyc;
        const flash = U.smoothstep(p / 0.14) * (1 - U.smoothstep((p - 0.76) / 0.24));
        const grow = U.ease(Math.min(p / 0.45, 1));
        const colK = U.viridis(k / Math.max(n - 1, 1));

        const env = { srcColor, srcName, scale, n, k, flash, grow, colK };

        if (compact) {
          /* ===== MOBILE : tout empilé verticalement, agrandi, rien masqué ===== */
          const M = 12;
          const innerW = W - 2 * M;

          // Titre
          U.text(ctx, 'Mel & ERB — l’oreille comme prior', W / 2, fs(15),
            { size: fs(13), bold: true, align: 'center' });

          // Sous-titre explicatif (ce que le module montre)
          U.text(ctx, `${scale === 'mel' ? 'Mel' : scale === 'erb' ? 'ERB' : 'Linéaire'} · ${n} bandes`,
            W / 2, fs(15) + fs(15), { size: fs(11), color: palette.dim, align: 'center' });

          const top = fs(15) + fs(15) + fs(10);

          // Hauteurs : spectre + zone de compression + bandes, en exploitant la hauteur.
          const sizesSpec = {
            compact: true, label: fs(11), src: fs(11.5),
            formula: fs(10.5), axis: fs(10),
          };
          const sizesBands = { compact: true, label: fs(11), head: fs(11.5) };

          // Réserves verticales (toujours positives même si H petit)
          const headSpec = fs(16);          // place au-dessus du cadre spectre (label + source)
          const axisSpace = fs(18);         // ticks Hz sous le spectre
          // Deux lignes empilées au-dessus du cadre bandes : « 512 bins → … »
          // puis « Énergies par bande ». Réserve la hauteur des deux + interligne.
          const headBands = fs(11) + fs(11.5) + fs(10); // = label + head + interligne
          const midH = Math.max(fs(70), Math.round(H * 0.20)); // zone de compression + annotations
          const botMargin = M;

          const available = H - top - headSpec - axisSpace - headBands - midH - botMargin;
          // Répartit la place restante entre spectre (haut) et bandes (bas).
          // Plancher garanti (>0) même si H est très réduit : aucune taille négative.
          const safeAvail = Math.max(fs(120), available);
          const specH = Math.max(fs(70), Math.round(safeAvail * 0.55));
          const bandsH = Math.max(fs(48), safeAvail - specH);

          const specBox = { x: M, y: top + headSpec, w: innerW, h: specH };
          const midTop = specBox.y + specBox.h + axisSpace;
          // La base des bandes ne dépasse jamais le canvas (clamp de sécurité)
          let bandsY = midTop + midH + headBands;
          bandsY = Math.min(bandsY, H - botMargin - bandsH);
          const bandsBox = { x: M, y: Math.max(bandsY, midTop + headBands), w: innerW, h: bandsH };

          const spc = drawSpectrum(specBox, env, sizesSpec);
          const bnd = drawBands(bandsBox, env, sizesBands);

          // Zone médiane : annotations en GRAND (bande active + bornes Hz)
          const midCenter = midTop + midH / 2;
          // Lignes de compression entre les deux panneaux
          drawCompression(env,
            { xOfBin: spc.xOfBin, specY: specBox.y, specH: specBox.h }, bnd, sizesSpec);

          if (flash > 0.18) {
            const chipTxt = `bande ${k + 1}/${n} : ${U.fmt.hz(fLo[k])} – ${U.fmt.hz(fHi[k])}`;
            ctx.save();
            ctx.globalAlpha = U.smoothstep((flash - 0.18) / 0.4);
            const chW = U.chip(ctx, chipTxt, 0, -999, { color: colK, size: fs(11) });
            U.chip(ctx, chipTxt, U.clamp(W / 2 - chW / 2, M, W - M - chW),
              midCenter, { color: colK, size: fs(11) });
            ctx.restore();
          }
          if (scale === 'erb' && n === 32) {
            // Note ERB placée AU-DESSUS de l'en-tête empilé du cadre bandes
            // (interligne fs(4)) pour ne toucher ni le titre ni « 512 bins → … ».
            const erbY = Math.max(midCenter + fs(8), bandsBox.y - headBands - fs(4));
            U.text(ctx, '32 bandes ERB = le choix de DeepFilterNet',
              W / 2, erbY,
              { size: fs(10.5), color: palette.green, align: 'center', bold: true });
          }
        } else {
          /* ===== DESKTOP / tablette : disposition d'origine (≥ 560 px) ===== */
          const M = 10;
          const specX = M, specW = W - 2 * M;
          const specY = M + 14;
          const specH = Math.round(H * 0.42);
          const axisY = specY + specH;
          const bandsH = Math.round(H * 0.26);
          const bandsY = H - M - bandsH;
          const bandsX = specX, bandsW = specW;
          const gapTop = axisY + 18, gapBot = bandsY - 16;

          const sizesSpec = { compact: false, label: 11, src: 11, formula: 10, axis: 9 };
          const sizesBands = { compact: false, label: 11, head: 11 };

          const specBox = { x: specX, y: specY, w: specW, h: specH };
          const bandsBox = { x: bandsX, y: bandsY, w: bandsW, h: bandsH };

          const spc = drawSpectrum(specBox, env, sizesSpec);
          const bnd = drawBands(bandsBox, env, sizesBands);

          drawCompression(env,
            { xOfBin: spc.xOfBin, specY, specH }, bnd, sizesSpec);

          /* Annotations de la zone médiane (desktop, inchangées) */
          const midY = (gapTop + gapBot) / 2;
          if (gapBot - gapTop > 16) {
            if (flash > 0.25) { // bande active : bornes en Hz vraies
              U.chip(ctx, `bande ${k + 1}/${n} : ${U.fmt.hz(fLo[k])} – ${U.fmt.hz(fHi[k])}`,
                specX, midY, { color: colK, size: 10 });
            }
            if (scale === 'erb' && n === 32) {
              U.text(ctx, '32 bandes ERB = le choix de DeepFilterNet',
                specX + specW - 2, midY + 3,
                { size: 11, color: palette.green, align: 'right', bold: true });
            }
          }
        }
      });
    },
  });
})();
