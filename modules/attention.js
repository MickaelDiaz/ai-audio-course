/* ============================================================
   Audio AI Atlas — module « attention »
   Self-attention sur 12 frames spectrales réelles : scores Q·K,
   softmax exact, sortie pondérée, matrice T×T, masque causal,
   têtes multiples. Tous les nombres affichés sont calculés.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  const T = 12;          // tokens (frames)
  const D = 16;          // dims par token (bins spectraux)
  const CYC = 3.0;       // secondes par query

  AtlasRegister({
    id: 'attention',
    title: 'Attention — chaque frame regarde les autres',
    category: 'layers',
    icon: '◉',
    summary: 'Q·Kᵀ → softmax → moyenne pondérée : scores réels entre 12 frames spectrales, masque causal et coût T².',
    explain: `
      <p><strong>Self-attention</strong> : chaque frame audio est projetée en <strong>query</strong>,
      <strong>key</strong> et <strong>value</strong>. Le score entre les frames <code>i</code> et <code>j</code>
      est un produit scalaire <code>q·k</code> normalisé (ici : similarité réelle entre colonnes spectrales de
      16 bins, divisée par une température τ). Le <strong>softmax</strong> transforme ces scores en distribution
      de probabilités (Σ = 1), et la sortie est la <strong>moyenne pondérée des values</strong> : chaque frame
      recompose sa représentation à partir de toutes les autres. Tout ce qui s'affiche ici est calculé
      exactement — scores, softmax, vecteur de sortie.</p>
      <p>Ce <strong>contexte global en un saut</strong> explique pourquoi l'attention domine en traitement
      offline : une frame voisée peut consulter directement une harmonique située 800 ms plus loin, sans passer
      par une chaîne récurrente. Les <strong>têtes multiples</strong> partitionnent les dimensions (16 = 4×4) :
      chaque tête calcule sa propre matrice d'attention sur son sous-espace et se spécialise (graves,
      fricatives…). C'est la brique des séparateurs état de l'art (<strong>Mel-RoFormer</strong>, BS-RoFormer)
      et des encodeurs type Whisper / AST.</p>
      <p>Le revers : le coût est <strong>quadratique</strong>. T frames → T² paires : 12 frames = 144 paires,
      mais 1 s d'audio à hop 10 ms = 100 frames → 10 000 paires, et 10 s → 1 000 000. En <strong>streaming</strong>,
      le futur n'existe pas : le <strong>masque causal</strong> interdit le triangle supérieur de la matrice, et
      contrairement à un RNN il n'y a <em>pas d'état compact</em> — il faut conserver tout le cache K/V, qui
      grandit avec la durée.</p>
      <p>Sur NPU embarqué, le <code>softmax</code> (exponentielles, normalisations) et les matmuls dynamiques de
      l'attention retombent souvent en <strong>fallback CPU</strong> : c'est le coût caché. D'où les compromis
      temps réel : fenêtres d'attention locales, attention linéaire, ou hybrides conv/RNN qui gardent un état
      borné.</p>`,

    init(stage) {
      const ctx = stage.ctx;

      /* ---------- features précalculées : T frames × D bins (rfftMag sur gen.speech) ---------- */
      const SR = 16000, N = 512, HOP = 0.07, T0 = 0.31;
      const feat = new Float32Array(T * D);  // pour les scores (compression sqrt)
      const disp = new Float32Array(T * D);  // pour l'affichage (normalisé par colonne)
      const per = (N / 2) / D;               // bins FFT regroupés par bin affiché
      for (let i = 0; i < T; i++) {
        const mag = U.rfftMag(U.gen.buffer(U.gen.speech, N, SR, T0 + i * HOP));
        let cmax = 1e-6;
        for (let d = 0; d < D; d++) {
          let s = 0;
          for (let k = 0; k < per; k++) s += mag[d * per + k];
          const v = Math.sqrt(s / per);
          feat[i * D + d] = v;
          if (v > cmax) cmax = v;
        }
        for (let d = 0; d < D; d++) disp[i * D + d] = Math.pow(feat[i * D + d] / cmax, 0.8);
      }

      /* ---------- buffers pré-alloués (rien d'alloué dans onFrame) ---------- */
      const Wfull = new Float32Array(T * T);                       // attention 16 dims
      const Wheads = [0, 1, 2, 3].map(() => new Float32Array(T * T));
      const srow = new Float32Array(T);                            // scratch softmax
      const dispW = new Float32Array(T);                           // distribution lissée (barres)
      const outVec = new Float32Array(D);                          // sortie pondérée
      const order = new Array(T); for (let i = 0; i < T; i++) order[i] = i;

      /* Attention exacte sur les dims [d0, d0+dn) : cos-sim / τ puis softmax (max-shift) */
      function computeAttn(Wout, d0, dn, tau, causal) {
        for (let q = 0; q < T; q++) {
          const lim = causal ? q : T - 1;
          let mx = -1e30;
          for (let j = 0; j <= lim; j++) {
            let dot = 0, nq = 0, nj = 0;
            for (let d = d0; d < d0 + dn; d++) {
              const a = feat[q * D + d], b = feat[j * D + d];
              dot += a * b; nq += a * a; nj += b * b;
            }
            const s = dot / (Math.sqrt(nq * nj) + 1e-9) / tau;
            srow[j] = s; if (s > mx) mx = s;
          }
          let sum = 0;
          for (let j = 0; j <= lim; j++) { srow[j] = Math.exp(srow[j] - mx); sum += srow[j]; }
          for (let j = 0; j < T; j++) Wout[q * T + j] = j <= lim ? srow[j] / sum : 0;
        }
      }

      /* ---------- coût quadratique : valeurs réellement calculées ---------- */
      const fi = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
      const HOPMS = 10;
      const T1 = Math.round(1000 / HOPMS), T10 = Math.round(10000 / HOPMS);
      const costFull = [
        `T = ${T} frames  →  T² = ${fi(T * T)} paires`,
        `1 s (hop ${HOPMS} ms)  →  T = ${fi(T1)}  →  ${fi(T1 * T1)} paires`,
        `10 s  →  T = ${fi(T10)}  →  ${fi(T10 * T10)} paires`,
      ];
      const costNarrow = [
        `T = ${T} → ${fi(T * T)} paires`,
        `1 s → ${fi(T1 * T1)}`,
        `10 s → ${fi(T10 * T10)}`,
      ];

      /* ---------- contrôles ---------- */
      const ctlCausal = stage.addToggle({ label: 'Masque causal (streaming)', value: false });
      const ctlHeads = stage.addSelect({
        label: 'Têtes d\'attention',
        options: [
          { value: '1', label: '1 tête (d = 16)' },
          { value: '2', label: '2 têtes (d = 8)' },
          { value: '4', label: '4 têtes (d = 4)' },
        ],
        value: '1',
      });
      const ctlTau = stage.addSlider({
        label: 'Température τ', min: 0.04, max: 0.4, step: 0.01, value: 0.12,
        format: (v) => 'τ = ' + (+v).toFixed(2),
      });

      let lastKey = '';
      function ensure() {
        const tau = ctlTau.value, causal = ctlCausal.value, h = +ctlHeads.value || 1;
        const key = tau + '|' + causal + '|' + h;
        if (key === lastKey) return;
        lastKey = key;
        computeAttn(Wfull, 0, D, tau, causal);
        if (h > 1) { const dn = D / h; for (let k = 0; k < h; k++) computeAttn(Wheads[k], k * dn, dn, tau, causal); }
      }

      /* Colonne spectrale (bin 0 = grave, en bas) */
      function drawColumn(x, y, w, h, base, gain) {
        const bh = h / D;
        for (let d = 0; d < D; d++) {
          ctx.fillStyle = U.magma(U.clamp((gain != null ? gain[d] : disp[base + d]), 0, 1));
          ctx.fillRect(x, y + (D - 1 - d) * bh, w, bh + 0.5);
        }
      }

      stage.onFrame((t, dt) => {
        stage.clear();
        ensure();
        const Wd = stage.W, Hd = stage.H;
        const m = 12, narrow = Wd < 560;
        const causal = ctlCausal.value, h = +ctlHeads.value || 1;

        /* ---- cycle : ~3 s par query, easé ---- */
        const cyc = Math.floor(t / CYC), p = (t % CYC) / CYC;
        const qi = cyc % T;
        const revealed = Math.min(T, cyc);                       // lignes de matrice déjà remplies
        const lineIn = U.ease(Math.min(1, p / 0.35));            // apparition des liens
        const rowFill = U.smoothstep((p - 0.2) / 0.55);          // remplissage de la ligne courante

        /* distribution affichée : suit la query avec lissage (la rangée de barres "vit") */
        const kS = 1 - Math.exp(-dt * 6);
        let sumW = 0;
        for (let j = 0; j < T; j++) { dispW[j] += (Wfull[qi * T + j] - dispW[j]) * kS; sumW += dispW[j]; }
        for (let d = 0; d < D; d++) {
          let s = 0;
          for (let j = 0; j < T; j++) s += dispW[j] * disp[j * D + d];
          outVec[d] = s;                                          // vraie moyenne pondérée
        }

        /* ---- layout (tout depuis W/H, chaque frame) ---- */
        const outW = U.clamp(Wd * 0.055, 26, 42);
        const tokArea = Wd - m * 2 - outW - 18;
        const tokW = tokArea / T, boxW = Math.max(8, tokW - 4);
        const yTok = 26;
        const tokH = U.clamp(Hd * 0.16, 44, 72);
        const gapH = U.clamp(Hd * 0.11, 30, 60);
        const yBars = yTok + tokH + gapH;
        const barH = U.clamp(Hd * 0.07, 18, 36);
        const yBot = yBars + barH + 24;
        const botH = Hd - yBot - m;
        const bx = (i) => m + i * tokW + (tokW - boxW) / 2;
        const cx = (i) => m + i * tokW + tokW / 2;

        /* ---- titre + chip causal ---- */
        U.text(ctx, narrow ? `T = ${T} frames audio` : `T = ${T} frames audio — colonnes spectrales ${D} bins (rfftMag, parole)`,
          m, 16, { size: 11, color: palette.dim });
        if (causal) U.chip(ctx, 'streaming ⇒ causal obligatoire', Wd - m - (narrow ? 168 : 186), 12, { color: palette.teal });

        /* ---- tokens : colonnes spectrales dans des cases arrondies ---- */
        for (let i = 0; i < T; i++) {
          const x = bx(i), isQ = i === qi;
          ctx.save();
          if (isQ) { ctx.shadowColor = palette.voice; ctx.shadowBlur = 13; }
          U.roundRect(ctx, x, yTok, boxW, tokH, 5);
          ctx.fillStyle = palette.panel; ctx.fill();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = isQ ? palette.voice : palette.grid;
          ctx.lineWidth = isQ ? 1.6 : 1; ctx.stroke();
          ctx.restore();
          drawColumn(x + 2, yTok + 2, boxW - 4, tokH - 4, i * D);
          if (boxW >= 16) U.text(ctx, isQ ? 'query' : 'f' + i, cx(i), yTok - 4,
            { size: 9, align: 'center', color: isQ ? palette.voice : palette.faint, bold: isQ });
        }

        /* ---- liens courbes query → keys (épaisseur + alpha = softmax réel) ---- */
        const limQ = causal ? qi : T - 1;
        ctx.save();
        ctx.strokeStyle = palette.voice;
        for (let j = 0; j <= limQ; j++) {
          if (j === qi) continue;
          const w = Wfull[qi * T + j];
          if (w < 0.004) continue;
          ctx.globalAlpha = lineIn * (0.08 + 0.92 * w);
          ctx.lineWidth = 0.6 + 8 * w;
          ctx.beginPath();
          ctx.moveTo(cx(qi), yTok + tokH);
          ctx.quadraticCurveTo((cx(qi) + cx(j)) / 2, yTok + tokH + gapH * 0.85, cx(j), yTok + tokH);
          ctx.stroke();
        }
        ctx.restore();

        /* ---- barres softmax sous les tokens ---- */
        let maxW = 1e-6, jMax = 0;
        for (let j = 0; j < T; j++) if (dispW[j] > maxW) { maxW = dispW[j]; jMax = j; }
        ctx.strokeStyle = palette.grid; ctx.beginPath();
        ctx.moveTo(m, yBars + barH); ctx.lineTo(m + tokArea, yBars + barH); ctx.stroke();
        for (let j = 0; j < T; j++) {
          const hb = (dispW[j] / maxW) * barH;
          ctx.globalAlpha = j <= limQ || !causal ? 0.92 : 0.25;
          ctx.fillStyle = j === qi ? palette.voice : (causal && j > qi ? palette.faint : palette.blue);
          ctx.fillRect(bx(j), yBars + barH - hb, boxW, hb);
          ctx.globalAlpha = 1;
        }
        U.text(ctx, U.fmt.pct(dispW[jMax]), cx(jMax), yBars - 3, { size: 9, align: 'center', color: palette.voice, mono: true });
        if (!narrow) U.text(ctx, 'poids = softmax( q·k / τ )', m, yBars + barH + 13, { size: 10, color: palette.dim, mono: true });
        U.text(ctx, `Σ wⱼ = ${Math.round(sumW * 100)} %`, m + tokArea, yBars + barH + 13,
          { size: 10, align: 'right', color: palette.dim, mono: true });

        /* ---- vecteur de sortie (moyenne pondérée réelle) + contributions ---- */
        const ox = Wd - m - outW;
        ctx.save();
        ctx.shadowColor = palette.mix; ctx.shadowBlur = 10;
        U.roundRect(ctx, ox, yTok, outW, tokH, 5);
        ctx.fillStyle = palette.panel; ctx.fill(); ctx.shadowBlur = 0;
        ctx.strokeStyle = palette.mix; ctx.lineWidth = 1.4; ctx.stroke();
        ctx.restore();
        drawColumn(ox + 2, yTok + 2, outW - 4, tokH - 4, 0, outVec);
        U.text(ctx, 'sortie', ox + outW / 2, yTok - 4, { size: 9, align: 'center', color: palette.mix, bold: true });
        U.text(ctx, 'Σ wⱼ·vⱼ', ox + outW / 2, yTok + tokH + 12, { size: 9, align: 'center', color: palette.faint, mono: true });
        /* top contributions (triées sur la vraie ligne softmax) */
        order.sort((a, b) => Wfull[qi * T + b] - Wfull[qi * T + a]);
        let cy = yTok + tokH + 26;
        for (let r = 0; r < 3; r++) {
          const j = order[r], w = Wfull[qi * T + j];
          if (w < 0.05) break;
          U.text(ctx, `f${j} · ${Math.round(w * 100)} %`, ox + outW, cy, { size: 9, align: 'right', color: palette.rest, mono: true });
          ctx.strokeStyle = palette.rest; ctx.globalAlpha = 0.8 * lineIn; ctx.lineWidth = 1;
          U.roundRect(ctx, bx(j) - 1.5, yTok - 1.5, boxW + 3, tokH + 3, 6); ctx.stroke();
          ctx.globalAlpha = 1;
          cy += 12;
        }

        /* ---- matrice d'attention T×T (remplie ligne par ligne au fil des cycles) ---- */
        const mLeft = m + (narrow ? 12 : 16);
        const showMinis = h > 1 && !narrow;
        const readW = narrow ? 150 : 230;
        const unitsX = T * (1 + (showMinis ? h * 0.42 : 0));
        const cellX = (Wd - m - mLeft - readW - 16 - (showMinis ? h * 10 + 8 : 0)) / unitsX;
        const cell = U.clamp(Math.min((botH - 22) / T, cellX), 4, 19);
        const matW = cell * T, mx = mLeft, my = yBot + 14;

        U.text(ctx, narrow ? 'attention T×T' : `matrice d'attention ${T}×${T}`, mx, yBot + 6, { size: 10, color: palette.dim });
        U.text(ctx, 'key →', mx + matW, yBot + 6, { size: 9, align: 'right', color: palette.faint });
        ctx.save();
        ctx.translate(mx - 6, my + matW / 2); ctx.rotate(-Math.PI / 2);
        U.text(ctx, 'query ↓', 0, 0, { size: 9, align: 'center', color: palette.faint });
        ctx.restore();

        ctx.fillStyle = palette.panel; ctx.fillRect(mx, my, matW, matW);
        const jReveal = Math.floor(rowFill * T + 1e-4);
        for (let q = 0; q < T; q++) {
          const full = q < revealed || revealed === T;
          const nCells = full ? T : (q === qi ? jReveal : 0);
          for (let j = 0; j < nCells; j++) {
            if (causal && j > q) continue;
            ctx.fillStyle = U.viridis(Math.sqrt(Wfull[q * T + j]));
            ctx.fillRect(mx + j * cell, my + q * cell, cell - 0.5, cell - 0.5);
          }
        }
        if (causal) {                                            /* triangle futur : hachuré sombre */
          ctx.strokeStyle = 'rgba(139,150,165,0.22)'; ctx.lineWidth = 1;
          for (let q = 0; q < T; q++) for (let j = q + 1; j < T; j++) {
            const x = mx + j * cell, y = my + q * cell;
            ctx.fillStyle = 'rgba(8,10,14,0.85)'; ctx.fillRect(x, y, cell - 0.5, cell - 0.5);
            ctx.beginPath(); ctx.moveTo(x, y + cell - 0.5); ctx.lineTo(x + cell - 0.5, y); ctx.stroke();
          }
        }
        ctx.strokeStyle = palette.voice; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.9;
        ctx.strokeRect(mx - 1, my + qi * cell - 1, matW + 2, cell + 1.5);  // ligne de la query courante
        ctx.globalAlpha = 1;
        ctx.strokeStyle = palette.grid; ctx.strokeRect(mx, my, matW, matW);

        /* ---- miniatures des têtes (vrais calculs par sous-vecteurs) ---- */
        let rx = mx + matW + 18;
        if (showMinis) {
          const mc = Math.max(2, Math.floor(cell * 0.42)), mw = mc * T;
          for (let k = 0; k < h; k++) {
            const hx = rx + k * (mw + 10);
            ctx.fillStyle = palette.panel; ctx.fillRect(hx, my, mw, mw);
            for (let q = 0; q < T; q++) {
              if (!(q < revealed || revealed === T || q === qi)) continue;
              const nC = (q < revealed || revealed === T) ? T : jReveal;
              for (let j = 0; j < nC; j++) {
                if (causal && j > q) continue;
                ctx.fillStyle = U.viridis(Math.sqrt(Wheads[k][q * T + j]));
                ctx.fillRect(hx + j * mc, my + q * mc, mc, mc);
              }
            }
            ctx.strokeStyle = palette.grid; ctx.strokeRect(hx, my, mw, mw);
            U.text(ctx, `tête ${k + 1}`, hx + mw / 2, my + mw + 11, { size: 9, align: 'center', color: palette.faint });
          }
          rx += h * (mw + 10) + 8;
        }

        /* ---- readout : coût quadratique (valeurs calculées) ---- */
        const lines = narrow ? costNarrow : costFull;
        const lh = narrow ? 13 : 16, fs = narrow ? 9 : 11;
        U.text(ctx, 'Coût quadratique', rx, my + 4, { size: narrow ? 11 : 12, bold: true, color: palette.text });
        for (let i = 0; i < lines.length; i++)
          U.text(ctx, lines[i], rx, my + 4 + (i + 1) * lh + 2, { size: fs, color: i === 0 ? palette.dim : palette.rest, mono: true });
        const ny = my + 4 + 4 * lh + 8;
        if (ny + lh < Hd - 2) {
          U.text(ctx, narrow ? 'softmax : souvent fallback CPU' : 'softmax/attention : souvent fallback CPU sur', rx, ny, { size: fs, color: palette.orange });
          U.text(ctx, narrow ? 'sur NPU embarqué — coût caché' : 'NPU embarqué — le coût caché', rx, ny + lh - 2, { size: fs, color: palette.orange });
        }
      });
    },
  });
})();
