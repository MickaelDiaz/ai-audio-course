/* ============================================================
   Audio AI Atlas — « Mel & ERB — l'oreille comme prior »
   Banque de filtres perceptuelle appliquée à un VRAI spectre FFT :
   512 bins linéaires compressés en N bandes (Mel / ERB / linéaire),
   énergies = vraie multiplication banque × spectre.
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
      <p>L'oreille n'analyse pas le son sur une grille linéaire : la <strong>cochlée</strong> déroule
      les fréquences le long de la membrane basilaire avec une résolution très fine dans les graves
      et de plus en plus grossière vers les aigus. Les échelles <strong>Mel</strong>
      (<code>mel = 2595·log₁₀(1 + f/700)</code>) et <strong>ERB</strong>
      (<em>Equivalent Rectangular Bandwidth</em>, <code>≈ 21,4·log₁₀(1 + 0,00437·f)</code>)
      modélisent cette courbe : des filtres serrés sous 1&nbsp;kHz, de plus en plus larges au-dessus.
      Ici le spectre est calculé à 16&nbsp;kHz (axe 0–8&nbsp;kHz) ; les systèmes <em>full-band</em>
      appliquent exactement la même logique à 48&nbsp;kHz (axe 0–24&nbsp;kHz).</p>
      <p>Pourquoi compresser 512 bins en N bandes ? Parce qu'un réseau qui prédit un
      <strong>gain par bande</strong> n'a plus que N sorties au lieu de 512 : moins de paramètres,
      moins de variance d'estimation, et une cible alignée sur ce que l'oreille perçoit réellement.
      Chaque énergie de bande est une <strong>somme pondérée</strong> de dizaines de bins voisins —
      la banque agit comme un lissage structuré qui <strong>stabilise l'apprentissage</strong> :
      une erreur sur une bande large coûte peu perceptivement, alors qu'un gain erratique bin par
      bin produit des artefacts musicaux audibles.</p>
      <p>C'est le pattern récurrent « <strong>gains grossiers perceptuels + raffinement fin
      ailleurs</strong> » : <strong>DeepFilterNet</strong> prédit 32 gains ERB pour sculpter
      l'enveloppe spectrale sur tout le spectre, puis un étage de <em>deep filtering</em> raffine la
      structure fine des seules basses fréquences (harmoniques de la voix), là où l'oreille exige de
      la précision. On dépense les paramètres là où ils comptent, et on laisse le prior perceptuel
      faire le reste.</p>`,

    init(stage) {
      const ctx = stage.ctx;

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

      /* ---------- Boucle de rendu ---------- */
      stage.onFrame((t, dt) => {
        stage.clear();
        const W = stage.W, H = stage.H;
        const narrow = W < 560;

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

        /* --- Mise en page (recalculée chaque frame depuis W/H) --- */
        const M = 10;
        const specX = M, specW = W - 2 * M;
        const specY = M + 14;
        const specH = Math.round(H * 0.42);
        const axisY = specY + specH;
        const bandsH = Math.round(H * 0.26);
        const bandsY = H - M - bandsH;
        const bandsX = specX, bandsW = specW;
        const gapTop = axisY + 18, gapBot = bandsY - 16;
        const xOfBin = (b) => specX + ((b + 0.5) / BINS) * specW;

        /* ===== HAUT : spectre linéaire vrai + banque superposée ===== */
        U.frame(ctx, specX, specY, specW, specH,
          narrow ? 'Spectre — 512 bins' : 'Spectre linéaire — 512 bins (0–8 kHz, sr 16 kHz)');
        U.text(ctx, 'source : ' + srcName, specX + specW - 2, specY - 6,
          { size: 11, color: srcColor, align: 'right', bold: true });

        U.bars(ctx, dispSpec, specX, specY, specW, specH,
          { color: srcColor, alpha: 0.38, gap: 0 });

        if (!narrow) { // formule d'espacement vraie, affichée
          const formula = scale === 'mel' ? 'mel = 2595·log₁₀(1 + f/700)'
            : scale === 'erb' ? 'ERB ≈ 21,4·log₁₀(1 + 0,00437·f)'
            : 'espacement linéaire uniforme';
          U.text(ctx, formula, specX + specW - 8, specY + 14,
            { size: 10, color: palette.faint, align: 'right', mono: true });
        }

        /* Filtres colorés (viridis), remplis + arête supérieure */
        const hCurve = specH * 0.92;
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

        /* Axe des fréquences (Hz), annoté */
        ctx.strokeStyle = palette.grid;
        ctx.lineWidth = 1;
        const tickStep = W < 700 ? 2000 : 1000;
        for (let f = 0; f <= NYQ; f += tickStep) {
          const x = specX + (f / NYQ) * specW;
          ctx.beginPath(); ctx.moveTo(x, axisY); ctx.lineTo(x, axisY + 4); ctx.stroke();
          U.text(ctx, U.fmt.hz(f), x, axisY + 14, {
            size: 9, color: palette.faint, mono: true,
            align: f === 0 ? 'left' : f === NYQ ? 'right' : 'center',
          });
        }

        /* ===== MILIEU : lignes de compression bins → bande active ===== */
        const lo = loBin[k], hi = hiBin[k];
        const colK = U.viridis(k / Math.max(n - 1, 1));
        const x2 = bandsX + ((k + 0.5) / n) * bandsW;
        const y2 = bandsY + bandsH * (1 - dispBand[k]);
        if (flash > 0.02) {
          const nLines = Math.min(9, hi - lo + 1);
          ctx.strokeStyle = colK;
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.55 * flash;
          for (let j = 0; j < nLines; j++) {
            const b = nLines === 1 ? lo : lo + Math.round((j * (hi - lo)) / (nLines - 1));
            const x1 = xOfBin(b);
            const y1 = specY + specH * (1 - dispSpec[b]);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(U.lerp(x1, x2, grow), U.lerp(y1, y2, grow));
            ctx.stroke();
          }
          ctx.globalAlpha = 1;
          if (grow > 0.95) U.glowDot(ctx, x2, y2, 2.5 + 1.5 * flash, colK);
        }

        /* Annotations de la zone médiane */
        const midY = (gapTop + gapBot) / 2;
        if (gapBot - gapTop > 16) {
          if (!narrow && flash > 0.25) { // bande active : bornes en Hz vraies
            U.chip(ctx, `bande ${k + 1}/${n} : ${U.fmt.hz(fLo[k])} – ${U.fmt.hz(fHi[k])}`,
              specX, midY, { color: colK, size: 10 });
          }
          if (scale === 'erb' && n === 32) {
            U.text(ctx, '32 bandes ERB = le choix de DeepFilterNet',
              specX + specW - 2, midY + 3,
              { size: narrow ? 9 : 11, color: palette.green, align: 'right', bold: true });
          }
        }

        /* ===== BAS : énergies par bande (banque × spectre) ===== */
        U.frame(ctx, bandsX, bandsY, bandsW, bandsH,
          narrow ? 'Énergies par bande' : 'Énergies par bande — banque × spectre');
        const ratio = (BINS / n).toFixed(1).replace(/\.0$/, '');
        U.text(ctx, `512 bins → ${n} bandes (÷${ratio})`,
          bandsX + bandsW - 2, bandsY - 6,
          { size: narrow ? 10 : 11, color: palette.text, align: 'right', mono: true });

        const bw = bandsW / n;
        for (let i = 0; i < n; i++) {
          const v = dispBand[i];
          const bh = Math.max(v * (bandsH - 4), 1);
          const x = bandsX + i * bw;
          const isK = i === k;
          ctx.save();
          if (isK && flash > 0.05) { ctx.shadowColor = colK; ctx.shadowBlur = 10 * flash; }
          ctx.fillStyle = U.viridis(i / Math.max(n - 1, 1));
          ctx.globalAlpha = isK ? 0.7 + 0.3 * flash : 0.8;
          ctx.fillRect(x + 1, bandsY + bandsH - 2 - bh, Math.max(bw - 2, 1), bh);
          ctx.restore();
        }
        ctx.globalAlpha = 1;
      });
    },
  });
})();
