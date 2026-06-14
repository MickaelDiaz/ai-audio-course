/* ============================================================
   Audio AI Atlas — module « Convolution 2D — sur le spectrogramme »
   Un kernel 3×3 balaie un vrai spectrogramme de pseudo-parole ;
   la carte de sortie (convolution exacte) se révèle au fil du scan.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  /* ---------- constantes ---------- */
  const IMG = 56;                       // côté de l'image spectrogramme
  const KS = 3, OUT = IMG - KS + 1;     // convolution « valid » → 54×54
  const TOTAL = OUT * OUT;              // 2916 cellules de sortie
  const SR = 16000, NFFT = 128, HOP = 64;
  const HOLD = Math.round(TOTAL * 0.22); // pause (en cellules) carte complète

  const KERNELS = [
    { id: 'harm',  label: 'Harmoniques (bord horizontal)', w: [-1, -1, -1, 2, 2, 2, -1, -1, -1] },
    { id: 'trans', label: 'Transitoires (bord vertical)',  w: [-1, 2, -1, -1, 2, -1, -1, 2, -1] },
    { id: 'blur',  label: 'Lissage (moyenne 1/9)',         w: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9] },
    { id: 'lap',   label: 'Laplacien (contours)',          w: [0, -1, 0, -1, 4, -1, 0, -1, 0] },
  ];
  const KBY = {};
  for (const k of KERNELS) KBY[k.id] = k;

  /* entiers avec espaces de milliers ; décimaux à virgule française */
  const th = (n) => String(n).replace(/\B(?=(\d{3})+$)/g, ' ');
  const fr2 = (x) => (x < 0 ? '−' : '') + Math.abs(x).toFixed(2).replace('.', ',');
  const fmtW = (v) => Math.abs(v - 1 / 9) < 1e-9 ? '1/9' : Number.isInteger(v) ? String(v) : fr2(v);

  /* Spectrogramme 56×56 : pseudo-parole à 16 kHz, FFT 128 (rfftMag), 56 bins
     graves conservés, 56 trames consécutives (hop 64), log-normalisé [0,1].
     t0 est choisi pour contenir une plosive (départ de syllabe, large bande,
     verticale) précédée d'une syllabe voisée (harmoniques horizontaux). */
  function buildSpectrogram() {
    let S = 2; // reproduit les conditions internes de U.gen.speech (SYL = 0.19 s)
    for (let s = 2; s < 500; s++) {
      if (U.hash(s) < 0.82 && U.hash(s + 0.7) > 0.45 && U.hash(s - 1) < 0.82) { S = s; break; }
    }
    const t0 = S * 0.19 - 0.40 * (IMG * HOP / SR); // plosive à ~40 % de l'image
    const spec = new Float32Array(IMG * IMG);      // rangée 0 = aigu (haut)
    for (let x = 0; x < IMG; x++) {
      const buf = U.gen.buffer(U.gen.speech, NFFT, SR, t0 + x * HOP / SR);
      const mag = U.rfftMag(buf);                  // 64 bins, on garde les 56 du bas
      for (let r = 0; r < IMG; r++) spec[r * IMG + x] = Math.log10(mag[IMG - 1 - r] + 1e-3);
    }
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < spec.length; i++) { if (spec[i] < mn) mn = spec[i]; if (spec[i] > mx) mx = spec[i]; }
    const d = (mx - mn) || 1;
    for (let i = 0; i < spec.length; i++) spec[i] = (spec[i] - mn) / d;
    return spec;
  }

  /* Repères pédagogiques mesurés sur l'image elle-même (rien de décoratif). */
  function analyse(spec) {
    let plosiveCol = 0, best = -1;                 // colonne la plus énergique en HF
    for (let x = 0; x < IMG; x++) {
      let s = 0;
      for (let r = 0; r < IMG >> 1; r++) s += spec[r * IMG + x];
      if (s > best) { best = s; plosiveCol = x; }
    }
    const harmRow = new Uint8Array(IMG);           // rangées graves à forte énergie moyenne
    let mxRow = 0;
    const mean = new Float32Array(IMG);
    for (let r = IMG - 16; r < IMG; r++) {
      let s = 0;
      for (let x = 0; x < IMG; x++) s += spec[r * IMG + x];
      mean[r] = s / IMG;
      if (mean[r] > mxRow) mxRow = mean[r];
    }
    for (let r = IMG - 16; r < IMG; r++) harmRow[r] = mean[r] > 0.8 * mxRow ? 1 : 0;
    return { plosiveCol, harmRow };
  }

  AtlasRegister({
    id: 'conv2d',
    title: 'Convolution 2D — sur le spectrogramme',
    category: 'layers',
    icon: '⊞',
    summary: 'Un kernel 3×3 balaie un vrai spectrogramme : les bords horizontaux allument les harmoniques, les verticaux les transitoires.',
    explain: `
      <p>Un <strong>spectrogramme</strong> est une matrice temps × fréquence : rien n'empêche de le traiter
      comme une image et d'y appliquer une <strong>convolution 2D</strong>. Chaque valeur de la carte de
      sortie ne dépend que d'une petite fenêtre (ici 3×3) de l'entrée, et le même jeu de 9 poids balaie
      toute l'image : c'est le <strong>partage de poids</strong> qui rend les CNN si économes. Mais attention,
      contrairement à une photo, les deux axes ne sont <strong>pas équivalents</strong> : translater un motif
      dans le temps ne change pas sa nature (un « a » prononcé plus tard reste un « a »), alors que le
      translater en fréquence change la hauteur, le timbre, voire le phonème. D'où l'usage d'échelles
      log/mel et, parfois, de convolutions contraintes le long de l'axe fréquentiel.</p>
      <p>Les <strong>premiers kernels</strong> appris par un encoder audio ressemblent précisément aux filtres
      de ce module : des détecteurs de bords orientés. Un bord <strong>horizontal</strong> répond aux
      <strong>partiels tenus</strong> (harmoniques d'une voyelle, note tenue) ; un bord <strong>vertical</strong>
      répond aux <strong>transitoires</strong> (plosives, attaques) — essayez les deux noyaux et observez
      quelles structures « s'allument » dans la carte de sortie. Les couches suivantes recombinent ces
      primitives en motifs de plus en plus abstraits : formants, phonèmes, timbres.</p>
      <p>Le coût d'une couche standard est <code>k·k·C<sub>in</sub>·C<sub>out</sub></code> paramètres : avec un
      noyau 3×3 et 32 canaux en entrée comme en sortie, 9·32·32 = 9 216 poids. La convolution
      <strong>depthwise separable</strong> factorise l'opération : un filtre spatial 3×3 <em>par canal</em>
      (9·32 = 288), puis une convolution <em>pointwise</em> 1×1 qui mélange les canaux (32·32 = 1 024),
      soit 1 312 paramètres — environ <strong>7 fois moins</strong> pour une qualité proche. C'est le standard
      de l'<strong>audio embarqué</strong> (détection de mots-clés, débruitage temps réel sur mobile),
      popularisé par MobileNet et repris par la plupart des modèles « edge ».</p>`,

    init(stage) {
      const ctx = stage.ctx;

      /* ---------- précalculs (rien de coûteux dans onFrame) ---------- */
      const spec = buildSpectrogram();
      const { plosiveCol, harmRow } = analyse(spec);

      // entrée en magma : offscreen 56×56, agrandi en nearest-neighbor au dessin
      const srcCv = document.createElement('canvas');
      srcCv.width = IMG; srcCv.height = IMG;
      const srcCx = srcCv.getContext('2d');
      const srcIm = srcCx.createImageData(IMG, IMG);
      for (let i = 0; i < spec.length; i++) {
        const c = U.magmaRGB(spec[i]), o = i * 4;
        srcIm.data[o] = c[0]; srcIm.data[o + 1] = c[1]; srcIm.data[o + 2] = c[2]; srcIm.data[o + 3] = 255;
      }
      srcCx.putImageData(srcIm, 0, 0);

      // sortie : convolution exacte, recalculée seulement au changement de noyau
      const feat = new Float32Array(TOTAL);
      const outCv = document.createElement('canvas');
      outCv.width = OUT; outCv.height = OUT;
      const outCx = outCv.getContext('2d');
      const outIm = outCx.createImageData(OUT, OUT);
      let maxAbs = 1, builtId = '';

      function rebuild(id) {
        builtId = id;
        const w = KBY[id].w;
        maxAbs = 1e-9;
        for (let r = 0; r < OUT; r++) {
          for (let c = 0; c < OUT; c++) {
            let s = 0;
            for (let kr = 0; kr < KS; kr++)
              for (let kc = 0; kc < KS; kc++) s += w[kr * KS + kc] * spec[(r + kr) * IMG + (c + kc)];
            feat[r * OUT + c] = s;
            const a = Math.abs(s);
            if (a > maxAbs) maxAbs = a;
          }
        }
        for (let i = 0; i < TOTAL; i++) {    // normalisation par |max| ; négatif → sombre
          const c = U.viridisRGB(U.clamp(feat[i] / maxAbs, 0, 1)), o = i * 4;
          outIm.data[o] = c[0]; outIm.data[o + 1] = c[1]; outIm.data[o + 2] = c[2]; outIm.data[o + 3] = 255;
        }
        outCx.putImageData(outIm, 0, 0);
      }

      /* ---------- contrôles ---------- */
      const selKernel = stage.addSelect({
        label: 'Noyau 3×3',
        options: KERNELS.map(k => ({ value: k.id, label: k.label })),
        value: 'harm',
      });
      const slSpeed = stage.addSlider({
        label: 'Vitesse de balayage', min: 60, max: 2400, step: 60, value: 600,
        format: (v) => v + ' cel/s',
      });
      const slChan = stage.addSlider({
        label: 'Canaux (Cin = Cout)', min: 8, max: 128, step: 8, value: 32,
        format: (v) => v + ' ch',
      });
      rebuild(selKernel.value);

      /* ---------- état d'animation ---------- */
      let progress = 0;                          // position du scan (cellules, float)
      let chipA = 0, chipTxt = '', chipCol = palette.voice;

      const cellPos = (i) => {                   // serpentin : trajet continu, sans saut
        const row = (i / OUT) | 0, k = i % OUT;
        return [(row & 1) === 0 ? k : OUT - 1 - k, row, k];
      };

      stage.onFrame((t, dt) => {
        const W = stage.W, H = stage.H;
        stage.clear();

        if (selKernel.value !== builtId) { rebuild(selKernel.value); progress = 0; }

        progress += slSpeed.value * dt;
        const cycle = TOTAL + HOLD;
        if (progress > 1e7) progress %= cycle;   // t peut être très grand
        const p = progress % cycle;
        const scanning = p < TOTAL;
        const idx = scanning ? (p | 0) : TOTAL - 1;
        const frac = scanning ? p - idx : 0;
        const [c1, r1, k1] = cellPos(idx);
        const [c2, r2] = cellPos(Math.min(idx + 1, TOTAL - 1));
        const fc = U.lerp(c1, c2, frac), frow = U.lerp(r1, r2, frac);
        const rawCur = feat[r1 * OUT + c1];
        const normCur = U.clamp(rawCur / maxAbs, 0, 1);

        /* ---------- layout, recalculé chaque frame depuis W/H ---------- */
        const narrow = W <= 560, tight = W < 720;
        const mTop = 44, mBot = 30, gap = narrow ? 14 : 16;
        const innerH = H - mTop - mBot - 14;     // 14 px pour « temps → »
        const cw = narrow ? 0 : U.clamp(W * 0.22, 168, 230);
        const nGaps = narrow ? 1 : 2;
        const sideW = (W - 24 - cw - gap * nGaps - (narrow ? 0 : 14)) / 2;
        const S = Math.max(40, Math.min(sideW, innerH));
        const totalW = S * 2 + gap * nGaps + cw;
        const lx = (W - totalW) / 2 + (narrow ? 0 : 7);
        const ly = mTop + Math.max(0, (innerH - S) / 2);
        const cpx = lx + S + gap;                // colonne centrale (si large)
        const ox = narrow ? cpx : cpx + cw + gap;
        const cs = S / IMG, cs2 = S / OUT;

        U.text(ctx, narrow ? 'Conv2D sur le spectrogramme' : 'Convolution 2D — le spectrogramme vu comme une image',
          12, 18, { size: 13, bold: true });

        /* ---------- gauche : entrée magma + kernel en serpentin ---------- */
        U.frame(ctx, lx - 1, ly - 1, S + 2, S + 2, narrow ? 'entrée 56×56 (magma)' : 'entrée 56×56 — spectrogramme (magma)');
        ctx.save();
        ctx.imageSmoothingEnabled = false;       // cellules carrées, nearest-neighbor
        ctx.drawImage(srcCv, lx, ly, S, S);
        ctx.restore();
        U.text(ctx, 'temps →', lx + S / 2, ly + S + 12, { size: 10, color: palette.faint, align: 'center' });
        if (!narrow) {
          ctx.save();
          ctx.translate(lx - 6, ly + S / 2);
          ctx.rotate(-Math.PI / 2);
          U.text(ctx, 'fréquence →', 0, 0, { size: 10, color: palette.faint, align: 'center' });
          ctx.restore();
        }
        const kx = lx + fc * cs, ky = ly + frow * cs;
        const ka = scanning ? U.ease(Math.min(1, p / 30)) : 0; // apparition easée
        if (ka > 0.01) {
          ctx.save();
          ctx.globalAlpha = ka;
          ctx.strokeStyle = palette.voice;
          ctx.lineWidth = 2;
          ctx.shadowColor = palette.voice;
          ctx.shadowBlur = 8;
          ctx.strokeRect(kx, ky, KS * cs, KS * cs);
          ctx.restore();
        }

        /* ---------- droite : carte de sortie révélée au fil du scan ---------- */
        U.frame(ctx, ox - 1, ly - 1, S + 2, S + 2, narrow ? 'sortie 54×54 (viridis)' : 'sortie 54×54 — feature map (viridis)');
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        if (!scanning) {
          ctx.drawImage(outCv, ox, ly, S, S);
        } else {
          if (r1 > 0) ctx.drawImage(outCv, 0, 0, OUT, r1, ox, ly, S, r1 * cs2);
          const sx = (r1 & 1) === 0 ? 0 : c1;    // rangée courante, sens du serpentin
          ctx.drawImage(outCv, sx, r1, k1 + 1, 1, ox + sx * cs2, ly + r1 * cs2, (k1 + 1) * cs2, cs2);
        }
        ctx.restore();
        if (scanning) {
          U.glowDot(ctx, ox + (fc + 0.5) * cs2, ly + (frow + 0.5) * cs2, 2 + 2.5 * normCur, palette.yellow);
        } else {
          U.text(ctx, 'balayage terminé — relance…', ox + S / 2, ly + S + 12, { size: 10, color: palette.faint, align: 'center' });
        }

        /* ---------- chip contextuelle (détection vraie, mesurée à l'init) ---------- */
        let want = '', wantCol = palette.voice;
        if (scanning) {
          if (c1 <= plosiveCol && plosiveCol <= c1 + 2) {
            want = narrow ? 'ici : transitoire (plosive)' : 'ici : transitoire détecté (plosive)';
            wantCol = palette.rest;
          } else if (harmRow[r1] || harmRow[r1 + 1] || harmRow[r1 + 2]) {
            want = narrow ? 'ici : partiel tenu' : 'ici : partiel tenu (harmonique)';
            wantCol = palette.voice;
          }
        }
        if (want) { chipTxt = want; chipCol = wantCol; }
        chipA += ((want ? 1 : 0) - chipA) * Math.min(1, dt * 7);
        if (chipA > 0.02 && chipTxt) {
          const chx = Math.max(lx + 2, Math.min(kx - 16, lx + S - 170));
          const chy = ky > ly + 26 ? ky - 13 : ky + KS * cs + 13;
          ctx.save();
          ctx.globalAlpha = U.smoothstep(chipA);
          U.chip(ctx, chipTxt, chx, chy, { color: chipCol });
          ctx.restore();
        }

        /* ---------- centre (si W > 560) : le calcul exact, pas à pas ---------- */
        if (!narrow) {
          const w = KBY[builtId].w;
          const g = Math.max(18, Math.min(34, (cw - 36) / 3, (innerH - 130) / 6));
          const gridW = g * 3 + 4;
          const gx = cpx + (cw - gridW) / 2;
          const blockH = 14 + gridW + 30 + gridW + 22 + 20 + 18;
          let yy = ly + Math.max(0, (S - blockH) / 2);

          U.text(ctx, 'fenêtre 3×3 (pixels)', cpx + cw / 2, yy + 6, { size: 10, color: palette.dim, align: 'center' });
          yy += 14;
          for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
              const v = spec[(r1 + r) * IMG + (c1 + c)];
              const x0 = gx + c * (g + 2), y0 = yy + r * (g + 2);
              ctx.fillStyle = U.magma(v);
              ctx.fillRect(x0, y0, g, g);
              U.text(ctx, g >= 26 ? v.toFixed(2) : v.toFixed(1), x0 + g / 2, y0 + g / 2,
                { size: 9, mono: true, align: 'center', baseline: 'middle', color: v > 0.62 ? '#20121f' : '#f3e8e2' });
            }
          }
          yy += gridW + 4;
          U.text(ctx, '×  poids du noyau', cpx + cw / 2, yy + 8, { size: 10, color: palette.dim, align: 'center' });
          yy += 14;
          ctx.save();
          ctx.strokeStyle = palette.voice;
          ctx.globalAlpha = 0.55;
          ctx.strokeRect(gx - 3, yy - 3, gridW + 4, gridW + 4); // rappel du cadre corail
          ctx.restore();
          for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
              const v = w[r * 3 + c];
              const x0 = gx + c * (g + 2), y0 = yy + r * (g + 2);
              ctx.fillStyle = palette.panel2;
              ctx.fillRect(x0, y0, g, g);
              U.text(ctx, fmtW(v), x0 + g / 2, y0 + g / 2,
                { size: 9, mono: true, bold: true, align: 'center', baseline: 'middle',
                  color: v > 0 ? palette.voice : v < 0 ? palette.blue : palette.dim });
            }
          }
          yy += gridW + 20;
          U.text(ctx, 'Σ poids · pixel = ' + fr2(rawCur), cpx + cw / 2, yy,
            { size: 12, bold: true, mono: true, align: 'center' });
          yy += 18;
          const lbl = 'sortie (÷|max|) : ' + fr2(normCur);
          ctx.font = '11px ' + U.MONO;
          const tw = ctx.measureText(lbl).width;
          ctx.fillStyle = U.viridis(normCur);
          ctx.fillRect(cpx + cw / 2 - tw / 2 - 14, yy - 9, 10, 10);
          U.text(ctx, lbl, cpx + cw / 2 + 1, yy, { size: 11, mono: true, align: 'center', color: palette.dim });
        }

        /* ---------- bas : coût en paramètres, réellement calculé ---------- */
        const C = slChan.value;
        const pStd = 9 * C * C;                 // standard : k·k·Cin·Cout
        const pDW = 9 * C + C * C;              // depthwise (9·Cin) + pointwise (Cin·Cout)
        const ratio = (pStd / pDW).toFixed(1).replace('.', ',');
        const by = H - 11;
        ctx.font = '11px ' + U.MONO;
        const sA = tight ? `3×3, C=${C} : std ${th(pStd)}`
          : `noyau 3×3, Cin=Cout=${C} — standard : 9·${C}·${C} = ${th(pStd)} poids`;
        const sB = tight ? ` vs DW+PW ${th(pDW)}`
          : `   depthwise+pointwise : 9·${C} + ${C}·${C} = ${th(pDW)}`;
        const sC = `  (÷${ratio})`;
        const wA = ctx.measureText(sA).width, wB = ctx.measureText(sB).width, wC = ctx.measureText(sC).width;
        let bx = (W - wA - wB - wC) / 2;
        U.text(ctx, sA, bx, by, { size: 11, mono: true, color: palette.dim }); bx += wA;
        U.text(ctx, sB, bx, by, { size: 11, mono: true, color: palette.voice }); bx += wB;
        U.text(ctx, sC, bx, by, { size: 11, mono: true, bold: true, color: palette.rest });
      });
    },
  });
})();
