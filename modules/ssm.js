/* ============================================================
   Audio AI Atlas — module « SSM & Mamba — l'état continu »
   Un vrai système dynamique x' = A·x + B·u(t), y = C·x,
   intégré par Euler à chaque frame, avec sélectivité Mamba.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;
  const TAU = U.TAU;

  AtlasRegister({
    id: 'ssm',
    title: "SSM & Mamba — l'état continu",
    category: 'layers',
    icon: 'Ŝ',
    summary: "Un système dynamique simulé en direct : mémoire multi-échelle et gate sélective de Mamba.",
    explain: `
      <p>Un <strong>State Space Model</strong> est, au fond, un <strong>RNN linéaire à temps continu</strong> :
      <code>x' = A·x + B·u(t)</code>, <code>y = C·x</code>, ensuite discrétisé pour tourner sur des séquences.
      Ici la matrice <code>A</code> est diagonale et stable : chaque composante de l'état est une
      <strong>fuite mémorielle</strong> indépendante de constante de temps <code>τ = 1/|λ|</code>. Avec des λ
      étalés de −20 à −0,6 s⁻¹, l'état combine des mémoires de 50 ms à ~1,7 s : c'est une
      <strong>mémoire multi-échelle</strong>, exactement ce qu'il faut pour de l'audio (phonèmes courts,
      prosodie longue).</p>
      <p>L'intérêt structurel : comme la récurrence est <em>linéaire</em>, on peut entraîner en
      <strong>parallèle</strong> sur toute la séquence (scan parallèle / convolution équivalente), puis
      inférer en <strong>récurrence O(1) par pas</strong> avec un état de taille fixe — là où l'attention
      relit tout le passé (O(T) par pas, cache qui grossit). Sur le papier, c'est le meilleur des deux
      mondes : le débit d'entraînement d'un Transformer, le coût d'inférence d'un RNN.</p>
      <p><strong>Mamba</strong> ajoute la <strong>sélectivité</strong> : <code>B</code> (et le pas Δ) deviennent
      des fonctions de l'entrée. Concrètement, une <strong>gate</strong> décide à chaque instant ce qui a le
      droit de s'écrire dans la mémoire. Activez « Sélectif » avec l'entrée « Parole + bruit » : sans gate,
      l'état intègre le bruit de fond en continu ; avec gate, il ne s'écrit que pendant la parole —
      c'est <em>le</em> point de Mamba : filtrer ce qui entre dans l'état, pas seulement ce qui en sort.</p>
      <p>Honnêteté d'ingénieur : ces opérateurs (scan sélectif, kernels fusionnés) sont récents, et le
      support des <strong>compilateurs NPU embarqués</strong> reste inégal — à vérifier sur votre cible
      matérielle avant d'en dépendre en production.</p>`,

    init(stage) {
      const ctx = stage.ctx;

      /* ---- Système dynamique : constantes (fixes, déterministes) ---- */
      const LAM = [-0.6, -1.2, -2.5, -5, -10, -20];     // valeurs propres (s⁻¹)
      const NS = LAM.length;
      const B = new Float32Array(NS);                    // gain d'entrée (hash, normalisé par |λ| → gain DC comparable)
      const C = new Float32Array(NS);                    // lecture (hash, signes mêlés)
      let cAbs = 0;
      for (let i = 0; i < NS; i++) {
        B[i] = Math.abs(LAM[i]) * (0.75 + 0.5 * U.hash(i + 3.7));
        C[i] = U.hash(i + 13.3) - 0.5;
        cAbs += Math.abs(C[i]);
      }
      for (let i = 0; i < NS; i++) C[i] *= 1.3 / cAbs;   // Σ|C| = 1.3 → y lisible

      /* ---- Buffers circulaires (pré-alloués, jamais en onFrame) ---- */
      const SIM_HZ = 200, H_STEP = 1 / SIM_HZ;
      const NBUF = 800;                                   // fenêtre affichée = 4,0 s
      const WIN_S = NBUF / SIM_HZ;
      const bufU = new Float32Array(NBUF);
      const bufY = new Float32Array(NBUF);
      const bufG = new Float32Array(NBUF);
      const bufX = []; for (let i = 0; i < NS; i++) bufX.push(new Float32Array(NBUF));
      let wIdx = 0;                                       // prochaine écriture = échantillon le plus ancien

      /* ---- État de simulation ---- */
      const x = new Float32Array(NS);
      let tSim = 0, env = 0, selSmooth = 0, acc = 0;

      /* ---- Contrôles ---- */
      const inputSel = stage.addSelect({
        label: 'Entrée u(t)',
        options: [
          { value: 'speech', label: 'Enveloppe de parole' },
          { value: 'pulse', label: 'Impulsions périodiques' },
          { value: 'sine', label: 'Sinus lent' },
          { value: 'mix', label: 'Parole + bruit de fond' },
        ],
        value: 'speech',
      });
      const selTgl = stage.addToggle({ label: 'Sélectif (Mamba)', value: false });
      const thrSld = stage.addSlider({
        label: 'Seuil de la gate', min: 0.05, max: 0.8, step: 0.01, value: 0.35,
        format: (v) => v.toFixed(2),
      });
      const noiseSld = stage.addSlider({
        label: 'Bruit de fond', min: 0, max: 0.5, step: 0.01, value: 0.25,
        format: (v) => U.fmt.pct(v),
      });

      /* ---- Entrée u(t) — continue en temps de simulation ---- */
      function speechEnv() {
        let raw = 0;
        for (let s = 0; s < 4; s++) raw += Math.abs(U.gen.speech(tSim - s * H_STEP * 0.25));
        raw = raw * 0.25 * 2.8;
        env += (raw - env) * (raw > env ? 0.55 : 0.10);   // attaque rapide, relâche lente
        return U.clamp(env, 0, 1.2);
      }
      function inputAt() {
        const mode = inputSel.value;
        if (mode === 'pulse') {
          const ph = tSim % 0.8;
          return ph < 0.06 ? Math.sin(Math.PI * ph / 0.06) : 0;
        }
        if (mode === 'sine') return 0.85 * Math.sin(TAU * 0.45 * tSim);
        const e = speechEnv();
        if (mode === 'speech') return e;
        const nl = noiseSld.value;                        // plancher de bruit + scintillement
        return e * 0.9 + nl * (0.7 + 0.6 * U.noise1(tSim * 37)) + (U.hash(Math.floor(tSim * 200)) - 0.5) * 0.12 * nl;
      }
      const sigmoid = (z) => 1 / (1 + Math.exp(-z));

      /* ---- Tracé d'un buffer circulaire (oscilloscope défilant) ---- */
      function trace(buf, px, py, pw, ph, sc, color, alpha, lw) {
        ctx.save();
        ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = lw;
        ctx.beginPath();
        const stride = Math.max(1, Math.floor(NBUF / Math.max(40, pw)));
        const cy = py + ph / 2, half = ph / 2 - 1.5;
        let first = true;
        for (let k = 0; k < NBUF; k += stride) {
          const v = buf[(wIdx + k) % NBUF];
          const gx = px + (k / (NBUF - 1)) * pw;
          const gy = cy - U.clamp(v * sc, -half, half);
          if (first) { ctx.moveTo(gx, gy); first = false; } else ctx.lineTo(gx, gy);
        }
        ctx.stroke();
        ctx.restore();
      }
      function zeroLine(px, py, pw, ph) {
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, py + ph / 2); ctx.lineTo(px + pw, py + ph / 2); ctx.stroke();
      }

      /* ---- Boucle ---- */
      stage.onFrame((t, dt) => {
        /* 1) Simulation : Euler à pas fixe (stable pour h·|λ|max = 0.1) */
        const selective = selTgl.value;
        selSmooth += ((selective ? 1 : 0) - selSmooth) * 0.12;
        const eSel = U.smoothstep(selSmooth);
        const thr = thrSld.value;
        acc = Math.min(acc + dt, 0.3);                   // accumulateur → cadence exacte, sans dérive
        let nSteps = Math.floor(acc * SIM_HZ);
        acc -= nSteps * H_STEP;
        while (nSteps-- > 0) {
          tSim += H_STEP;
          const u = inputAt();
          const g = sigmoid(8 * (Math.abs(u) - thr));
          const w = U.lerp(1, g, eSel);                   // B ← g·B (fondu doux au toggle)
          let y = 0;
          for (let i = 0; i < NS; i++) {
            x[i] += H_STEP * (LAM[i] * x[i] + B[i] * w * u);
            y += C[i] * x[i];
          }
          bufU[wIdx] = u; bufY[wIdx] = y; bufG[wIdx] = g;
          for (let i = 0; i < NS; i++) bufX[i][wIdx] = x[i];
          wIdx = (wIdx + 1) % NBUF;
        }

        /* 2) Layout — tout depuis W/H, chaque frame */
        stage.clear();
        const W = stage.W, Ht = stage.H;
        const narrow = W < 560;
        const pad = 10, headerH = narrow ? 22 : 26, labelH = 16;
        const panelW = W > 620 ? Math.min(250, W * 0.27) : 0;
        const plotX = pad;
        const plotW = W - pad * 2 - (panelW ? panelW + 12 : 0);
        const gateSpace = (narrow ? 12 : 15) * eSel;
        const availH = Ht - headerH - 3 * labelH - gateSpace - pad - 4;
        const uH = Math.max(24, availH * 0.24);
        const sH = Math.max(40, availH * 0.46);
        const yH = Math.max(24, availH * 0.30);
        const uY = headerH + labelH;
        const gateY = uY + uH + labelH;
        const sY = gateY + gateSpace + (eSel > 0.02 ? 3 : 0);
        const yY = sY + sH + labelH;

        /* 3) En-tête : équation + mode courant */
        U.text(ctx, narrow ? "x′ = A·x + B·u   y = C·x" : "x′ = A·x + B·u(t)    y = C·x    A = diag(λ₁…λ₆),  λ ∈ [−20, −0.6] s⁻¹",
          plotX, headerH - 9, { size: narrow ? 10 : 11, color: palette.dim, mono: true });
        if (!narrow) {
          const lbl = selective ? 'Sélectif (Mamba) · B ← g·B' : 'LTI · B fixe';
          ctx.font = '600 10px ' + U.FONT;
          const cw = ctx.measureText(lbl).width + 14;
          U.chip(ctx, lbl, plotX + plotW - cw, headerH - 12, { color: selective ? palette.voice : palette.rest });
        }

        /* 4) Bandeau 1 — entrée u(t) (bleu) */
        U.frame(ctx, plotX, uY, plotW, uH);
        U.text(ctx, 'Entrée u(t)', plotX + 2, uY - 5, { size: 11, color: palette.dim });
        zeroLine(plotX, uY, plotW, uH);
        const uScale = uH / 2 / 1.3;
        trace(bufU, plotX, uY, plotW, uH, uScale, palette.blue, 0.95, 1.5);
        const uNow = bufU[(wIdx + NBUF - 1) % NBUF];
        U.glowDot(ctx, plotX + plotW, uY + uH / 2 - U.clamp(uNow * uScale, -uH / 2 + 2, uH / 2 - 2), 2.5, palette.blue);
        if (!narrow) U.text(ctx, 'u = ' + uNow.toFixed(2), plotX + plotW - 4, uY - 5, { size: 10, color: palette.blue, align: 'right', mono: true });

        /* 5) Gate sélective — bandeau lumineux corail (s'ouvre/se ferme en direct) */
        U.text(ctx, 'État x(t) ∈ ℝ⁶ — six constantes de temps', plotX + 2, gateY - 5, { size: 11, color: palette.dim });
        if (eSel > 0.02) {
          const gh = gateSpace;
          ctx.fillStyle = palette.panel;
          ctx.fillRect(plotX, gateY, plotW, gh);
          const strideG = Math.max(1, Math.floor(NBUF / 220));
          const cw2 = plotW / Math.ceil(NBUF / strideG);
          let gSum = 0;
          for (let k = 0, col = 0; k < NBUF; k += strideG, col++) {
            const g = bufG[(wIdx + k) % NBUF];
            gSum += g;
            if (g > 0.02) {
              ctx.globalAlpha = g * eSel;
              ctx.fillStyle = palette.voice;
              ctx.fillRect(plotX + col * cw2, gateY, cw2 + 0.5, gh);
            }
          }
          ctx.globalAlpha = 1;
          const gOpen = gSum / Math.ceil(NBUF / strideG);
          if (eSel > 0.5) {
            U.text(ctx, 'gate g(t) — ouverte ' + U.fmt.pct(gOpen) + ' de la fenêtre', plotX + plotW - 4, gateY - 5,
              { size: 10, color: palette.voice, align: 'right' });
          }
        }

        /* 6) Bandeau 2 — les 6 composantes de l'état (viridis) */
        U.frame(ctx, plotX, sY, plotW, sH);
        zeroLine(plotX, sY, plotW, sH);
        const sScale = sH / 2 / 1.7;
        for (let i = 0; i < NS; i++) trace(bufX[i], plotX, sY, plotW, sH, sScale, U.viridis(i / (NS - 1)), 0.8, 1.3);
        let nrm = 0; for (let i = 0; i < NS; i++) nrm += x[i] * x[i];
        nrm = Math.sqrt(nrm);
        U.text(ctx, '‖x‖ = ' + nrm.toFixed(2), plotX + plotW - 6, sY + 13, { size: 10, color: palette.dim, align: 'right', mono: true });
        if (!narrow) {
          /* annotations vraies : τ = 1/|λ| pour la plus rapide et la plus lente */
          const cyS = sY + sH / 2, halfS = sH / 2 - 2;
          const yFast = cyS - U.clamp(x[NS - 1] * sScale, -halfS, halfS);
          const yArr = U.clamp(yFast, sY + 24, sY + sH - 30);
          const tFast = 'λ = ' + LAM[NS - 1] + ' s⁻¹ → oubli en ' + U.fmt.ms(1 / Math.abs(LAM[NS - 1]));
          const tSlow = 'λ = ' + LAM[0] + ' s⁻¹ → mémoire ≈ ' + U.fmt.ms(1 / Math.abs(LAM[0]));
          U.text(ctx, tFast, plotX + plotW - 6, sY + sH - 20, { size: 10, color: U.viridis(1), align: 'right' });
          U.text(ctx, tSlow, plotX + plotW - 6, sY + sH - 7, { size: 10, color: U.viridis(0), align: 'right' });
          U.arrow(ctx, plotX + plotW - 120, sY + sH - 24, plotX + plotW - 36, yArr, { color: U.viridis(1), alpha: 0.35, head: 4 });
          /* la morale, en direct, sur l'entrée bruitée */
          if (inputSel.value === 'mix') {
            const msg = selective ? 'gate fermée hors parole : le bruit n’entre pas dans l’état'
              : 'sans sélectivité : le bruit s’intègre en continu dans l’état';
            const col = selective ? palette.voice : palette.rest;
            ctx.font = '600 10px ' + U.FONT;
            const mw = ctx.measureText(msg).width + 12;
            ctx.fillStyle = palette.panel; ctx.globalAlpha = 0.85;
            U.roundRect(ctx, plotX + 6, sY + 5, mw, 16, 4); ctx.fill(); ctx.globalAlpha = 1;
            U.text(ctx, msg, plotX + 12, sY + 16, { size: 10, color: col, bold: true });
          }
        }

        /* 7) Bandeau 3 — sortie y(t) (corail) */
        U.frame(ctx, plotX, yY, plotW, yH);
        U.text(ctx, 'Sortie y = C·x', plotX + 2, yY - 5, { size: 11, color: palette.dim });
        zeroLine(plotX, yY, plotW, yH);
        const yScale = yH / 2 / 1.2;
        trace(bufY, plotX, yY, plotW, yH, yScale, palette.voice, 0.95, 1.5);
        const yNow = bufY[(wIdx + NBUF - 1) % NBUF];
        U.glowDot(ctx, plotX + plotW, yY + yH / 2 - U.clamp(yNow * yScale, -yH / 2 + 2, yH / 2 - 2), 2.5, palette.voice);
        U.text(ctx, 'fenêtre : ' + WIN_S.toFixed(1) + ' s  ·  Euler à ' + SIM_HZ + ' pas/s', plotX + plotW - 4, yY - 5,
          { size: 10, color: palette.faint, align: 'right' });

        /* 8) Panneau droit (W > 620) : comparatif + constantes de temps */
        if (panelW) {
          const px = W - pad - panelW, py = headerH + labelH;
          const ph = Ht - py - pad;
          U.frame(ctx, px, py, panelW, ph);
          U.text(ctx, 'RNN vs SSM vs Attention', px + 2, py - 5, { size: 11, color: palette.dim });
          const c1 = px + 10, c2 = px + panelW * 0.30, c3 = px + panelW * 0.50, c4 = px + panelW * 0.72;
          let ry = py + 18;
          U.text(ctx, 'coût/pas', c2, ry, { size: 9, color: palette.faint });
          U.text(ctx, 'état', c3, ry, { size: 9, color: palette.faint });
          U.text(ctx, 'training', c4, ry, { size: 9, color: palette.faint });
          const rows = [
            ['RNN', 'O(1)', 'n', 'séquentiel', palette.rest],
            ['SSM', 'O(1)', 'n', 'parallèle (scan)', palette.voice],
            ['Attention', 'O(T)', 'tout le passé', 'parallèle', palette.mix],
          ];
          for (let r = 0; r < 3; r++) {
            ry += 17;
            if (rows[r][0] === 'SSM') {
              ctx.fillStyle = palette.voice; ctx.globalAlpha = 0.10;
              ctx.fillRect(px + 4, ry - 11, panelW - 8, 15);
              ctx.globalAlpha = 1;
            }
            U.text(ctx, rows[r][0], c1, ry, { size: 10, bold: true, color: rows[r][4] });
            U.text(ctx, rows[r][1], c2, ry, { size: 10, color: palette.text, mono: true });
            U.text(ctx, rows[r][2], c3, ry, { size: 9, color: palette.dim });
            U.text(ctx, rows[r][3], c4, ry, { size: 9, color: palette.dim });
          }
          /* constantes de temps réelles : τᵢ = 1/|λᵢ| */
          ry += 24;
          if (ry + 14 < py + ph) U.text(ctx, 'Mémoires de l’état (τ = 1/|λ|)', c1, ry, { size: 10, bold: true, color: palette.text });
          for (let i = 0; i < NS; i++) {
            const ly = ry + 14 + i * 13;
            if (ly > py + ph - 4) break;
            ctx.fillStyle = U.viridis(i / (NS - 1));
            ctx.fillRect(c1, ly - 7, 8, 8);
            U.text(ctx, 'λ = ' + LAM[i] + ' s⁻¹  →  τ = ' + U.fmt.ms(1 / Math.abs(LAM[i])), c1 + 14, ly,
              { size: 10, color: palette.dim, mono: true });
          }
        }
      });
    },
  });
})();
