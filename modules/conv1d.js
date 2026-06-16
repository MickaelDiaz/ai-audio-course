/* ============================================================
   Audio AI Atlas — module « Convolution 1D — kernels sur le temps »
   Un kernel de 9 poids glisse sur une pseudo-parole : produit
   scalaire vrai, sortie révélée, causalité, dilation, cône TCN.
   Mise en page responsive : empilée verticalement et AGRANDIE sur
   mobile (entrée, poids+calcul, sortie, puis le cône TCN en grand),
   layout desktop inchangé.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;
  const TAU = U.TAU;

  const N = 360;             // échantillons du signal d'entrée
  const SR = 16000;          // Hz
  const K = 9;               // taps du kernel
  const MID = (K - 1) / 2;   // 4
  const L = 3;               // couches empilées (inset TCN/WaveNet)
  const PLO = 205;           // index de la plosive (construction du signal)

  const fr = (v, d) => v.toFixed(d).replace('.', ','); // décimale française

  AtlasRegister({
    id: 'conv1d',
    title: 'Convolution 1D — kernels sur le temps',
    category: 'layers',
    icon: '∗',
    summary: 'Un kernel de 9 poids glisse sur la forme d’onde : filtre FIR appris, causalité pour le streaming, dilation pour le contexte.',
    explain: `
      <p>Une <dfn class="term" data-term="convolution">convolution 1D</dfn> fait glisser un petit vecteur de K poids — le
      <dfn class="term" data-term="kernel">kernel</dfn> — le long du signal. À chaque position, elle calcule le
      <dfn class="term" data-term="dot-product">produit scalaire</dfn> entre les poids et les K échantillons couverts :
      c'est exactement un <dfn class="term" data-term="fir">filtre FIR</dfn> (réponse impulsionnelle finie). La seule
      différence avec le traitement du signal classique : les poids ne sont pas conçus à la
      main, ils sont <dfn class="term" data-term="gradient-descent">appris par descente de gradient</dfn>. Moyenne mobile
      (passe-bas), différence (passe-haut) ou dérivée lissée (détecteur de <dfn class="term" data-term="transient">transitoires</dfn>) sont
      des cas particuliers que le réseau peut redécouvrir… ou dépasser.</p>
      <p>Une vraie couche <code>Conv1d</code> apprend des dizaines de kernels en parallèle :
      un <dfn class="term" data-term="filterbank">banc de filtres</dfn>. Chaque kernel produit un <dfn class="term" data-term="channel">canal</dfn> de
      sortie, une <dfn class="term" data-term="feature-map">carte d'activation</dfn> qui répond à <em>son</em> motif (attaque, <dfn class="term" data-term="harmonique">harmonicité</dfn>,
      souffle…). Le même jeu de poids balaie tout le signal — c'est le <dfn class="term" data-term="weight-sharing">partage de poids</dfn>.
      Les couches suivantes combinent ces canaux pour détecter des structures de
      plus en plus abstraites — phonèmes, notes, événements sonores.</p>
      <p><strong>Causalité</strong> : un kernel centré regarde (K−1)/2 · d échantillons dans
      le futur (<dfn class="term" data-term="lookahead">look-ahead</dfn>). Inacceptable en <dfn class="term" data-term="streaming">streaming</dfn> temps réel :
      il faudrait attendre ces échantillons, donc ajouter de la <dfn class="term" data-term="latency">latence</dfn>. Les convolutions
      <dfn class="term" data-term="causal">causales</dfn> ne regardent que le passé — obligatoires pour la génération
      <dfn class="term" data-term="autoregressive">autorégressive</dfn> et le débruitage ou la séparation en direct.</p>
      <p><dfn class="term" data-term="dilation">Dilation</dfn> : en espaçant les taps de d échantillons, le <dfn class="term" data-term="receptive-field">champ réceptif</dfn>
      d'une couche passe à (K−1)·d+1 sans ajouter ni poids ni calcul. En empilant L couches
      dilatées (souvent d doublé à chaque étage : 1, 2, 4, 8…), le champ réceptif croît
      <strong>exponentiellement</strong> : c'est l'idée centrale de <dfn class="term" data-term="wavenet">WaveNet</dfn>
      et des <dfn class="term" data-term="tcn">TCN</dfn> pour couvrir des centaines de millisecondes d'audio avec de
      tout petits kernels.</p>`,

    init(stage) {
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

      /* ---------- Signal d'entrée FIXE : pseudo-parole, généré une fois ---------- */
      const sig = new Float32Array(N);
      let maxIn = 1e-6;
      for (let i = 0; i < N; i++) {
        const tt = i / SR;
        const e1 = Math.exp(-Math.pow((i - 88) / 60, 2));   // voyelle 1
        const e2 = Math.exp(-Math.pow((i - 296) / 46, 2));  // voyelle 2
        let v = e1 * (0.55 * Math.sin(TAU * 430 * tt) + 0.30 * Math.sin(TAU * 860 * tt + 1.2) + 0.18 * Math.sin(TAU * 1290 * tt + 0.5));
        v += e2 * (0.50 * Math.sin(TAU * 360 * tt + 0.7) + 0.27 * Math.sin(TAU * 720 * tt) + 0.12 * Math.sin(TAU * 1440 * tt));
        if (i >= PLO) v += (U.hash(i * 17.23) - 0.5) * 2.6 * Math.exp(-(i - PLO) / 5.5); // plosive : burst bref large bande
        v += (U.hash(i * 3.7 + 11) - 0.5) * 0.05; // souffle léger
        sig[i] = v;
        if (Math.abs(v) > maxIn) maxIn = Math.abs(v);
      }
      const iScale = 0.92 / maxIn;

      /* ---------- Kernels (K = 9), pré-alloués ---------- */
      const KERNELS = (() => {
        const lp = new Float32Array(K).fill(1 / K);                 // moyenne mobile
        const hp = new Float32Array(K).fill(-1 / (K - 1)); hp[MID] = 1; // centre − voisins
        const tr = new Float32Array(K);                             // dérivée de gaussienne
        let m = 1e-6;
        for (let i = 0; i < K; i++) { tr[i] = -(i - MID) * Math.exp(-Math.pow(i - MID, 2) / 8); m = Math.max(m, Math.abs(tr[i])); }
        for (let i = 0; i < K; i++) tr[i] *= 0.55 / m;
        const learn = new Float32Array(K);                          // poids hash « appris »
        for (let i = 0; i < K; i++) learn[i] = (U.hash(i * 12.9898 + 7.7) - 0.5) * 1.1;
        return { lp, hp, tr, learn };
      })();
      const KHINT = {
        lp: 'Passe-bas : chaque sortie = moyenne des 9 voisins → lisse, atténue les hautes fréquences.',
        hp: 'Passe-haut : centre − voisins → ne garde que les variations rapides du signal.',
        tr: 'Dérivée lissée : répond aux attaques — la plosive produit un pic net dans la sortie.',
        learn: 'Poids arbitraires (« appris ») : le SGD façonne chaque kernel en détecteur de motif.',
      };

      /* ---------- Contrôles ---------- */
      const selKernel = stage.addSelect({
        label: 'Kernel (K = 9)',
        options: [
          { value: 'lp', label: 'Passe-bas (moyenne mobile)' },
          { value: 'hp', label: 'Passe-haut (différence)' },
          { value: 'tr', label: 'Détecteur de transitoires' },
          { value: 'learn', label: '« Appris » (poids arbitraires)' },
        ],
        value: 'tr',
      });
      const tglCausal = stage.addToggle({ label: 'Mode causal (streaming)', value: false });
      const sldDil = stage.addSlider({ label: 'Dilation', min: 0, max: 3, step: 1, value: 0, format: (v) => 'd = ' + (1 << v) });
      const sldSpeed = stage.addSlider({ label: 'Vitesse de balayage', min: 0.3, max: 3, step: 0.1, value: 1, format: (v) => '×' + v.toFixed(1) });

      /* ---------- Convolution VRAIE (recalculée seulement au changement) ---------- */
      const out = new Float32Array(N);
      let curKey = '', curW = KERNELS.tr, wMax = 1, peakIdx = 0, maxOut = 1;
      function offAt(i, d, causal) { return causal ? (i - (K - 1)) * d : (i - MID) * d; }
      function recompute(kid, d, causal) {
        curW = KERNELS[kid] || KERNELS.tr;
        wMax = 1e-6;
        for (let i = 0; i < K; i++) wMax = Math.max(wMax, Math.abs(curW[i]));
        maxOut = 1e-6; peakIdx = 0;
        for (let n = 0; n < N; n++) {
          let acc = 0;
          for (let i = 0; i < K; i++) {
            const xi = n + offAt(i, d, causal);
            if (xi >= 0 && xi < N) acc += curW[i] * sig[xi]; // zero-padding hors bornes
          }
          out[n] = acc;
          if (Math.abs(acc) > maxOut) { maxOut = Math.abs(acc); peakIdx = n; }
        }
      }

      /* ============================================================
         HELPERS DE DESSIN — partagés entre compact (mobile) et desktop.
         Chacun reçoit une géométrie explicite (x, y, w, h) ; aucune
         hypothèse de ratio paysage. Toutes les tailles passent par fs().
         ============================================================ */

      /* --- Panneau d'ENTRÉE : waveform fixe + fenêtre glissante + K points --- */
      function drawInput(g, st) {
        const { ctx } = stage;
        const { x0, plotW, waveY, waveH, d, causal, n, fade } = st;
        const px = (i) => x0 + (i / (N - 1)) * plotW;
        const cy = waveY + waveH / 2;
        const yIn = (i) => cy - sig[i] * iScale * (waveH / 2);
        const ptR = g.compact ? 3 : 2.6;

        U.frame(ctx, x0 - 8, waveY - 5, plotW + 16, waveH + 10);
        ctx.save();
        ctx.strokeStyle = palette.grid; ctx.beginPath(); ctx.moveTo(x0, cy); ctx.lineTo(x0 + plotW, cy); ctx.stroke();
        ctx.restore();
        U.wave(ctx, sig, x0, waveY, plotW, waveH, { color: palette.blue, alpha: 0.5, lw: 1.2, scale: iScale });
        U.arrow(ctx, px(PLO) + 24, waveY + 12, px(PLO) + 3, waveY + waveH * 0.26, { color: palette.dim, head: 5, alpha: 0.8 });
        U.text(ctx, 'plosive', px(PLO) + 28, waveY + 14, { size: fs(g.compact ? 10 : 10), color: palette.dim });

        /* Fenêtre glissante : échantillons couverts + futur en jaune (look-ahead) */
        const minOff = offAt(0, d, causal), maxOff = offAt(K - 1, d, causal);
        const i1 = Math.max(0, n + minOff), i2 = Math.min(N - 1, n + maxOff);
        const wx1 = px(i1) - 3, wx2 = px(i2) + 3;
        ctx.save();
        ctx.globalAlpha = 0.07 * fade; ctx.fillStyle = palette.voice;
        ctx.fillRect(wx1, waveY, Math.max(wx2 - wx1, 1), waveH);
        if (!causal && i2 > n) {
          ctx.globalAlpha = 0.12 * fade; ctx.fillStyle = palette.yellow;
          ctx.fillRect(px(n) + 2, waveY, Math.max(wx2 - px(n) - 2, 1), waveH);
        }
        ctx.globalAlpha = 0.3 * fade; ctx.strokeStyle = palette.voice; ctx.lineWidth = 1;
        ctx.strokeRect(wx1, waveY, Math.max(wx2 - wx1, 1), waveH);
        ctx.globalAlpha = fade;
        for (let i = 0; i < K; i++) {
          const xi = n + offAt(i, d, causal);
          if (xi < 0 || xi >= N) continue;
          const future = xi > n;
          ctx.fillStyle = future ? palette.yellow : palette.voice;
          ctx.beginPath(); ctx.arc(px(xi), yIn(xi), ptR, 0, TAU); ctx.fill();
        }
        ctx.strokeStyle = palette.voice; ctx.lineWidth = 1.2;           // anneau = échantillon courant
        ctx.beginPath(); ctx.arc(px(n), yIn(n), g.compact ? 5.5 : 5, 0, TAU); ctx.stroke();
        ctx.restore();

        /* Chip causalité / look-ahead (valeurs vraies), ancrée en haut à droite du panneau */
        const la = MID * d;
        const chSize = fs(g.compact ? 10 : 10);
        const chy = waveY + (g.compact ? 14 : 12);
        if (!causal) {
          const label = `look-ahead = ${la} éch (+${fr(la / SR * 1000, 1)} ms)`;
          drawChipRight(ctx, label, x0 + plotW, chy, palette.yellow, chSize);
        } else {
          drawChipRight(ctx, 'causal : passé uniquement → streaming OK', x0 + plotW, chy, palette.voice, chSize);
        }
      }

      /* Pastille alignée à droite : on mesure sa largeur pour la coller au bord. */
      function drawChipRight(ctx, str, xRight, y, color, size) {
        ctx.font = `600 ${size}px ${U.FONT}`;
        const w = ctx.measureText(str).width + 14;
        U.chip(ctx, str, Math.max(8, xRight - w), y, { color, size });
      }

      /* --- Panneau POIDS + CALCUL : barres wᵢ alignées sous la fenêtre + Σ --- */
      function drawCalc(g, st) {
        const { ctx } = stage;
        const { x0, plotW, barsY, barsH, d, causal, n, fade } = st;
        const px = (i) => x0 + (i / (N - 1)) * plotW;
        const yb = barsY + barsH / 2;
        const minOff = offAt(0, d, causal), maxOff = offAt(K - 1, d, causal);
        const wx1 = px(Math.max(0, n + minOff)) - 3, wx2 = px(Math.min(N - 1, n + maxOff)) + 3;
        const bw = U.clamp((plotW / (N - 1)) * d * 0.45, g.compact ? 2.5 : 1.5, g.compact ? 12 : 9);

        if (g.compact) U.frame(ctx, x0 - 8, barsY - 5, plotW + 16, barsH + 10);
        ctx.save();
        ctx.globalAlpha = 0.5 * fade; ctx.strokeStyle = palette.faint;
        ctx.beginPath(); ctx.moveTo(wx1, yb); ctx.lineTo(wx2, yb); ctx.stroke();
        ctx.globalAlpha = 0.9 * fade;
        for (let i = 0; i < K; i++) {
          const xi = n + offAt(i, d, causal);
          const bx = px(xi);
          if (bx < x0 - 2 || bx > x0 + plotW + 2) continue;
          const hb = (curW[i] / wMax) * (barsH / 2 - 2);
          ctx.fillStyle = curW[i] >= 0 ? palette.voice : palette.blue;   // positifs corail, négatifs bleu
          ctx.fillRect(bx - bw / 2, Math.min(yb, yb - hb), bw, Math.abs(hb));
        }
        ctx.restore();
        U.text(ctx, 'poids wᵢ du kernel', x0, barsY + (g.compact ? 12 : 6), { size: fs(g.compact ? 10 : 9.5), color: palette.faint });

        /* Produit scalaire VRAI (= out[n]) */
        const sVal = (out[n] >= 0 ? '+' : '') + fr(out[n], 3);
        const margin = g.compact ? 95 : 150;
        const sx = U.clamp(px(n), x0 + margin * 0.6, x0 + plotW - margin * 0.6);
        const sw = U.clamp(g.W / 2, margin, g.W - margin);
        ctx.save(); ctx.globalAlpha = fade;
        U.text(ctx, `Σ wᵢ·xᵢ = ${sVal}`, g.compact ? sw : sx, barsY + barsH + (g.compact ? 16 : 11),
          { size: fs(g.compact ? 12.5 : 11.5), mono: true, bold: true, align: 'center', color: palette.mix });
        ctx.restore();
      }

      /* --- Panneau SORTIE : convolution vraie révélée jusqu'à la position --- */
      function drawOutput(g, st) {
        const { ctx } = stage;
        const { x0, plotW, outY, outH, nF, n, fade, kid } = st;
        const px = (i) => x0 + (i / (N - 1)) * plotW;
        const oyc = outY + outH / 2;
        const oScale = 0.92 / maxOut;
        const yOut = (i) => oyc - out[i] * oScale * (outH / 2);

        U.frame(ctx, x0 - 8, outY - 5, plotW + 16, outH + 10);
        U.text(ctx, 'Sortie y = (w ∗ x) — convolution réelle', x0 + 4, outY + (g.compact ? 15 : 13), { size: fs(g.compact ? 11 : 11), color: palette.dim });
        ctx.save();
        ctx.strokeStyle = palette.grid; ctx.beginPath(); ctx.moveTo(x0, oyc); ctx.lineTo(x0 + plotW, oyc); ctx.stroke();
        ctx.restore();
        U.wave(ctx, out, x0, outY, plotW, outH, { color: palette.faint, alpha: 0.16, lw: 1, scale: oScale }); // aperçu fantôme
        if (n >= 2) U.wave(ctx, out.subarray(0, n + 1), x0, outY, Math.max(px(n) - x0, 1), outH, { color: palette.voice, alpha: 0.95 * fade, lw: 1.6, scale: oScale });
        ctx.save(); ctx.globalAlpha = fade;
        U.glowDot(ctx, px(n), U.clamp(yOut(n), outY, outY + outH), 4, palette.voice);
        ctx.restore();

        /* Le détecteur de transitoires fait ressortir la plosive */
        if (kid === 'tr') {
          const a = fade * U.ease((nF - peakIdx - 6) / 18);
          if (a > 0.02) {
            ctx.save(); ctx.globalAlpha = a;
            const axL = U.clamp(px(peakIdx), x0 + 70, x0 + plotW - 90);
            U.text(ctx, 'la plosive ressort !', axL, outY + (g.compact ? 15 : 13), { size: fs(g.compact ? 11 : 11), bold: true, color: palette.rest });
            U.arrow(ctx, axL - 6, outY + (g.compact ? 13 : 11), px(peakIdx), U.clamp(yOut(peakIdx), outY + 6, outY + outH - 6), { color: palette.rest, head: 5, alpha: a });
            ctx.restore();
          }
        }

        /* Axe temps */
        U.text(ctx, '0 ms', x0, outY + outH + fs(13), { size: fs(g.compact ? 10 : 9), color: palette.faint });
        U.text(ctx, `${fr(N / SR * 1000, 1)} ms`, x0 + plotW, outY + outH + fs(13), { size: fs(g.compact ? 10 : 9), align: 'right', color: palette.faint });
      }

      /* --- INSET : cône du champ réceptif (TCN / WaveNet) ---
         Dessiné dans un rectangle (ix, iy, iw, ih) FOURNI par l'appelant.
         Sur mobile : pleine largeur, en bas. Sur desktop : coin du panneau sortie. */
      function drawCone(g, ix, iy, iw, ih, d, causal, t) {
        const { ctx } = stage;
        if (iw < 60 || ih < 70) return;   // garde-fou : place insuffisante
        ctx.save();
        U.roundRect(ctx, ix, iy, iw, ih, 8);
        ctx.fillStyle = palette.panel; ctx.globalAlpha = 0.97; ctx.fill();
        ctx.globalAlpha = 1; ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.stroke();
        U.text(ctx, 'Cône du champ réceptif — TCN / WaveNet', ix + 10, iy + fs(15), { size: fs(g.compact ? 11 : 10), bold: true, color: palette.dim });

        const nd = Math.max(3, 6 * d + 5);          // assez de neurones pour le cône
        const gx0 = ix + 12, sp = (iw - 24) / Math.max(1, nd - 1);
        const yTop = iy + fs(30), yBot = iy + ih - fs(34);
        const rowGap = Math.max(8, (yBot - yTop) / L);
        const rowY = (r) => yBot - r * rowGap;      // r = 0 (entrée) … L (sortie)
        const topIdx = causal ? nd - 3 : (nd - 1) >> 1;
        const xd = (i) => gx0 + i * sp;
        const dotR = g.compact ? 1.7 : 1.3, dotRa = g.compact ? 2.6 : 2;

        ctx.strokeStyle = palette.voice; ctx.lineWidth = 0.8;
        for (let r = L; r >= 1; r--) {              // connexions dilatées
          const span = L - r;
          const m0 = causal ? -2 * span : -span, m1 = causal ? 0 : span;
          ctx.globalAlpha = 0.26 + 0.10 * Math.sin(t * 2 + r);
          ctx.beginPath();
          for (let m = m0; m <= m1; m++) {
            const p = topIdx + m * d;
            for (let j = 0; j < 3; j++) {
              const c = p + (causal ? -j : j - 1) * d;
              if (c >= 0 && c < nd) { ctx.moveTo(xd(p), rowY(r)); ctx.lineTo(xd(c), rowY(r - 1)); }
            }
          }
          ctx.stroke();
        }
        for (let r = 0; r <= L; r++) {              // neurones (points)
          const span = L - r;
          for (let i = 0; i < nd; i++) {
            const k = i - topIdx;
            const m = k / d;
            const active = k % d === 0 && (causal ? (m <= 0 && m >= -2 * span) : Math.abs(m) <= span);
            ctx.globalAlpha = active ? 0.95 : 0.3;
            ctx.fillStyle = active ? palette.voice : palette.faint;
            ctx.beginPath(); ctx.arc(xd(i), rowY(r), active ? dotRa : dotR, 0, TAU); ctx.fill();
          }
        }
        ctx.globalAlpha = 1;
        U.glowDot(ctx, xd(topIdx), rowY(L), 3, palette.voice);
        const rf = (K - 1) * d * L + 1;             // champ réceptif VRAI de la pile
        U.text(ctx, `RF = (K−1)·d·L+1 = ${rf} éch ≈ ${fr(rf / SR * 1000, 1)} ms`, ix + 10, iy + ih - fs(18), { size: fs(g.compact ? 10.5 : 9.5), mono: true, color: palette.text });
        U.text(ctx, `K=${K}, L=${L} couches, d=${d} — 3 taps/couche · entrée en bas`, ix + 10, iy + ih - fs(6), { size: fs(g.compact ? 9.5 : 8.5), color: palette.faint });
        ctx.restore();
      }

      /* ---------- Boucle de rendu ---------- */
      stage.onFrame((t) => {
        const ctx = stage.ctx, W = stage.W, H = stage.H;
        stage.clear();
        const compact = stage.compact;
        const d = 1 << sldDil.value;
        const causal = tglCausal.value;
        const kid = selKernel.value;
        const key = kid + '|' + d + '|' + causal;
        if (key !== curKey) { curKey = key; recompute(kid, d, causal); }

        /* --- Cycle de balayage : glisse easée, pause, fondu, reprise (commun) --- */
        const T = 9 / sldSpeed.value;
        const p = (t / T) % 1;
        let prog, fade;
        if (p < 0.86) { prog = U.smoothstep(p / 0.86); fade = Math.min(1, p / 0.03); }
        else if (p < 0.96) { prog = 1; fade = 1; }
        else { prog = 1; fade = 1 - U.smoothstep((p - 0.96) / 0.04); }
        const nF = prog * (N - 1);
        const n = U.clamp(Math.round(nF), 0, N - 1);

        const x0 = 14, plotW = Math.max(40, W - 28);
        const px = (i) => x0 + (i / (N - 1)) * plotW;

        /* g : contexte partagé (dimensions + flags) ; st : état de frame complet. */
        const g = { compact, W, H };

        if (compact) {
          /* ===== MOBILE : empilement vertical, tout agrandi, rien de masqué ===== */
          const titleY = fs(15);
          U.text(ctx, `Entrée x[n] — pseudo-parole`, x0, titleY, { size: fs(11.5), color: palette.dim });
          U.text(ctx, `${N} éch · ${fr(N / SR * 1000, 1)} ms @ 16 kHz`, x0, titleY + fs(13), { size: fs(10.5), color: palette.faint });
          U.text(ctx, `K=${K} · d=${d} · ${causal ? 'causal' : 'centré'}`, W - 14, titleY, { size: fs(11.5), align: 'right', color: palette.dim, mono: true });

          /* Découpage vertical : entrée (waveform), poids+Σ, sortie, cône.
             On répartit la hauteur utile en bandes ; toutes les hauteurs > 0. */
          const top = titleY + fs(20);
          const bot = H - 6;
          const avail = Math.max(120, bot - top);
          const gapV = fs(20);                       // espace pour titres/Σ entre bandes

          const waveH = U.clamp(avail * 0.20, 70, 150);
          const barsH = U.clamp(avail * 0.11, 40, 90);
          const outH = U.clamp(avail * 0.22, 80, 170);
          const coneH = U.clamp(avail * 0.26, 90, 220);

          const waveY = top + fs(4);
          const barsY = waveY + waveH + gapV;
          const outY = barsY + barsH + gapV;
          const coneY = Math.min(outY + outH + gapV + fs(2), bot - coneH);

          const st = { x0, plotW, waveY, waveH, barsY, barsH, outY, outH, nF, n, fade, d, causal, kid, W, H };

          /* Ligne guide verticale entrée → sortie (avant les panneaux) */
          ctx.save();
          ctx.globalAlpha = 0.16 * fade; ctx.strokeStyle = palette.mix; ctx.setLineDash([3, 4]);
          ctx.beginPath(); ctx.moveTo(px(n), waveY); ctx.lineTo(px(n), outY + outH); ctx.stroke();
          ctx.restore();

          drawInput(g, st);
          drawCalc(g, st);
          drawOutput(g, st);
          drawCone(g, x0 - 8, Math.max(coneY, outY + outH + fs(8)), plotW + 16, coneH, d, causal, t);

          /* Légende du kernel courant, tout en bas */
          U.text(ctx, KHINT[kid] || '', W / 2, H - fs(4), { size: fs(10), align: 'center', color: palette.dim });
        } else {
          /* ===== DESKTOP / tablette : mise en page d'origine, inchangée ===== */
          const topY = 24, topH = H * 0.50 - topY;
          const waveY = topY, waveH = topH * 0.60;
          const barsY = waveY + waveH + 6, barsH = topH * 0.30;
          const outY = H * 0.56, outH = H - outY - 30;

          U.text(ctx, `Entrée x[n] — pseudo-parole · ${N} éch (${fr(N / SR * 1000, 1)} ms @ 16 kHz)`, x0, 15, { size: 11, color: palette.dim });
          U.text(ctx, `K = ${K} · d = ${d} · ${causal ? 'causal' : 'centré'}`, W - 14, 15, { size: 11, align: 'right', color: palette.dim, mono: true });

          const st = { x0, plotW, waveY, waveH, barsY, barsH, outY, outH, nF, n, fade, d, causal, kid, W, H };

          drawInput(g, st);
          drawCalc(g, st);

          /* Ligne guide verticale entrée → sortie */
          ctx.save();
          ctx.globalAlpha = 0.16 * fade; ctx.strokeStyle = palette.mix; ctx.setLineDash([3, 4]);
          ctx.beginPath(); ctx.moveTo(px(n), waveY); ctx.lineTo(px(n), outY + outH); ctx.stroke();
          ctx.restore();

          drawOutput(g, st);

          U.text(ctx, KHINT[kid] || '', W / 2, H - 8, { size: 10, align: 'center', color: palette.dim });

          /* Inset TCN / WaveNet, coin bas-droit du panneau sortie (si assez large) */
          if (W > 560) {
            const iw = Math.min(252, W * 0.30);
            const ih = Math.min(134, outH * 0.92);
            if (ih >= 86) drawCone(g, W - 14 - iw, outY + outH - ih, iw, ih, d, causal, t);
          }
        }
      });
    },
  });
})();
