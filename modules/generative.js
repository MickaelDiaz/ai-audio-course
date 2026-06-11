/* ============================================================
   Audio AI Atlas — module « generative »
   GAN, Diffusion, Flow Matching : trois façons de transporter
   du bruit vers la distribution des sons propres (métaphore 2D).
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;
  const { TAU, hash, lerp, clamp, ease, smoothstep, noise1, text, chip, glowDot, arrow, roundRect } = U;

  AtlasRegister({
    id: 'generative',
    title: 'GAN, Diffusion, Flow Matching',
    category: 'archi',
    icon: '✦',
    summary: 'Transporter du bruit vers la distribution des sons propres — et le coût réel en forwards (NFE).',
    explain: `
      <p><strong>Discriminatif ou génératif ?</strong> Un débruiteur par <strong>masking</strong> ne fait
      qu'<em>atténuer</em> ce qui est déjà dans le signal : zéro hallucination possible — c'est le choix sûr
      du broadcast. Un modèle <strong>génératif</strong>, lui, <em>reconstruit ce qui n'existe plus</em> :
      extension de bande (BWE), déclippage, codec neuronal… Mais il peut <strong>halluciner</strong> des
      phonèmes plausibles-et-faux, et son coût se compte en <strong>NFE</strong> (Number of Function
      Evaluations) : le nombre de forwards du réseau nécessaires pour produire <em>une</em> sortie.</p>
      <p>Les trois familles sont trois façons de <strong>transporter</strong> une distribution de bruit vers
      la variété des sons propres (le croissant teal). Le <strong>GAN</strong> saute en un seul forward
      (NFE = 1), mais son entraînement adversarial générateur-contre-discriminateur est instable
      (mode collapse, équilibre fragile). La <strong>diffusion</strong> débruite pas à pas — une marche
      aléatoire dont le bruit décroît step après step — et reste la référence qualité en offline, au prix
      de NFE = 16 à 200 forwards. Le <strong>flow matching</strong> apprend directement un <em>champ de
      vitesses</em> dont les trajectoires sont quasi droites : une ODE que l'on intègre en très peu de
      steps (NFE = 2 à 8), avec un entraînement aussi stable qu'une simple régression.</p>
      <p>Le verdict temps réel : <code>coût = NFE × latence d'un forward</code>. Si un forward coûte 2 ms
      sur NPU et que la frame audio fait 8 ms, NFE = 16 → 32 ms : intenable. La <strong>distillation
      one-step</strong> (ramener un modèle de diffusion ou de flow à NFE = 1) est le pont vers le temps
      réel embarqué — direction très active du domaine (Stream.FM, flow matching causal, consistency
      models).</p>`,

    init(stage) {
      const ctx = stage.ctx;

      /* ---------- constantes ---------- */
      const N_TGT = 140, N_BLOB = 90, N_ACT = 40, TRAIL = 22, LIFE = 1.3, NB = 36;
      const FWD_MS = 2, BUDGET_MS = 8, SCALE_MS = 64; // échelle de la jauge = 32 forwards max

      /* ---------- croissant cible (FIXE : arc paramétrique + jitter hash) ---------- */
      const tox = new Float32Array(N_TGT), toy = new Float32Array(N_TGT), tal = new Float32Array(N_TGT);
      for (let i = 0; i < N_TGT; i++) {
        const u = (i + 0.5) / N_TGT;
        const th = Math.PI * lerp(0.56, 1.44, u);            // arc « ( » ouvert vers la droite
        const thick = 0.13 + 0.30 * Math.sin(Math.PI * u);   // plus épais au centre
        const r = 1 + (hash(i * 2.317 + 0.71) - 0.5) * thick;
        const tj = (hash(i * 5.913 + 3.31) - 0.5) * 0.06;    // jitter tangentiel
        tox[i] = Math.cos(th + tj) * r;
        toy[i] = Math.sin(th + tj) * r;
        tal[i] = 0.35 + 0.45 * hash(i * 9.173 + 7.7);
      }
      const tpx = new Float32Array(N_TGT), tpy = new Float32Array(N_TGT); // positions px (recalc/frame)

      /* ---------- blob de bruit (gaussien ~ somme de 4 hash) ---------- */
      const gauss = (s) => (hash(s) + hash(s + 17.13) + hash(s + 41.77) + hash(s + 71.31) - 2) * 1.05;
      const nbx = new Float32Array(N_BLOB), nby = new Float32Array(N_BLOB), nba = new Float32Array(N_BLOB);
      for (let i = 0; i < N_BLOB; i++) {
        nbx[i] = gauss(i * 1.733 + 0.37);
        nby[i] = gauss(i * 3.911 + 9.13);
        nba[i] = 0.25 + 0.45 * hash(i * 6.271 + 2.9);
      }

      /* ---------- particules actives (~40, boucle organique) ---------- */
      let seedCtr = 1;
      const parts = [];
      function respawn(P, p0) {
        const s = (seedCtr++) * 13.371;
        P.seed = s;
        P.sx = gauss(s + 0.7); P.sy = gauss(s + 5.9);         // départ, en unités de σ
        P.ti = (hash(s + 9.1) * N_TGT) | 0;                   // cible sur le croissant
        P.dur = 0.8 + 0.55 * hash(s + 12.3);
        P.bend = (hash(s + 15.7) - 0.5) * 2;                  // courbure légère (flow)
        P.p = p0;
        P.tn = 0; P.th = 0;                                   // trail vide
      }
      for (let i = 0; i < N_ACT; i++) {
        const P = { trx: new Float32Array(TRAIL), try_: new Float32Array(TRAIL), tn: 0, th: 0 };
        respawn(P, hash(i * 3.7 + 0.21) * LIFE);              // phases étalées → flux continu
        parts.push(P);
      }

      /* ---------- contrôles ---------- */
      let modeT = 1; // fondu à l'arrivée sur un mode
      const clearTrails = () => { for (const P of parts) { P.tn = 0; P.th = 0; } };
      const selMode = stage.addSelect({
        label: 'mode',
        options: [{ value: 'gan', label: 'GAN' }, { value: 'diff', label: 'Diffusion' }, { value: 'fm', label: 'Flow Matching' }],
        value: 'gan',
        onChange: () => { modeT = 0.15; clearTrails(); },
      });
      const slDiff = stage.addSlider({ label: 'NFE diffusion', min: 2, max: 32, step: 1, value: 16, format: (v) => v + ' steps' });
      const slFm = stage.addSlider({ label: 'NFE flow matching', min: 1, max: 8, step: 1, value: 4, format: (v) => v + (v > 1 ? ' steps' : ' step') });

      /* ---------- champ de vitesses (cache, reconstruit au resize) ---------- */
      let fW = 0, fH = 0, fn = 0, ffx = null, ffy = null, fdx = null, fdy = null;
      stage.onResize(() => { fW = 0; clearTrails(); });
      function rebuildField(W, yA, yB) {
        const gap = clamp(Math.min(W, yB - yA) / 10, 30, 48);
        const cols = Math.max(2, Math.floor(W / gap)), rows = Math.max(2, Math.floor((yB - yA) / gap));
        const n = cols * rows;
        if (!ffx || ffx.length < n) { ffx = new Float32Array(n); ffy = new Float32Array(n); fdx = new Float32Array(n); fdy = new Float32Array(n); }
        fn = 0;
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          const gx = (c + 0.5) * gap, gy = yA + (r + 0.5) * gap;
          let best = 1e12, bi = 0;
          for (let i = 0; i < N_TGT; i++) {
            const dx = tpx[i] - gx, dy = tpy[i] - gy, d2 = dx * dx + dy * dy;
            if (d2 < best) { best = d2; bi = i; }
          }
          const d = Math.sqrt(best);
          if (d < 10) continue; // pas de flèche sur le croissant lui-même
          ffx[fn] = gx; ffy[fn] = gy;
          fdx[fn] = (tpx[bi] - gx) / d; fdy[fn] = (tpy[bi] - gy) / d;
          fn++;
        }
      }

      /* ---------- trajectoires (déterministes : seed + progrès) ---------- */
      let ppx = 0, ppy = 0, cvx = 0, cvy = 0;
      function curveFM(P, u, sx, sy, tx, ty, S) {
        const dx = tx - sx, dy = ty - sy, L = Math.hypot(dx, dy) || 1;
        const b = Math.sin(Math.PI * u) * S * 0.14 * P.bend;  // léger bombé ⊥ à la corde
        cvx = sx + dx * u - dy / L * b;
        cvy = sy + dy * u + dx / L * b;
      }
      function computePos(P, p01, mode, nfe, S, sx, sy, tx, ty) {
        if (mode === 'gan') {                                  // UN forward : saut direct easé
          const e = ease(p01);
          ppx = lerp(sx, tx, e); ppy = lerp(sy, ty, e);
        } else if (mode === 'diff') {                          // marche aléatoire biaisée, σ ↓
          const q = p01 * nfe;
          let k = Math.floor(q), f = q - k;
          if (k >= nfe) { k = nfe; f = 0; }
          let x = sx, y = sy;
          for (let j = 0; j < k; j++) {
            const a = 1 / (nfe - j);
            const ns = S * 0.62 / Math.pow(nfe, 0.55) * Math.pow(1 - (j + 1) / nfe, 1.25);
            x += (tx - x) * a + (hash(P.seed + j * 7.131) - 0.5) * 2 * ns;
            y += (ty - y) * a + (hash(P.seed + 47.7 + j * 9.713) - 0.5) * 2 * ns;
          }
          if (f > 0 && k < nfe) {                              // interpolation dans le step courant
            const a = 1 / (nfe - k);
            const ns = S * 0.62 / Math.pow(nfe, 0.55) * Math.pow(1 - (k + 1) / nfe, 1.25);
            const nx = x + (tx - x) * a + (hash(P.seed + k * 7.131) - 0.5) * 2 * ns;
            const ny = y + (ty - y) * a + (hash(P.seed + 47.7 + k * 9.713) - 0.5) * 2 * ns;
            x = lerp(x, nx, f); y = lerp(y, ny, f);
          }
          ppx = x; ppy = y;
        } else {                                               // flow matching : ODE quasi droite
          const q = p01 * nfe;
          let k = Math.floor(q), f = q - k;
          if (k >= nfe) { k = nfe; f = 0; }
          curveFM(P, k / nfe, sx, sy, tx, ty, S);
          const c0x = cvx, c0y = cvy;
          curveFM(P, Math.min(1, (k + 1) / nfe), sx, sy, tx, ty, S);
          ppx = lerp(c0x, cvx, f); ppy = lerp(c0y, cvy, f);
        }
      }

      /* ==================== boucle de rendu ==================== */
      stage.onFrame((t, dt) => {
        stage.clear();
        const W = stage.W, Hp = stage.H;
        const small = W < 560;
        modeT = Math.min(1, modeT + dt * 1.6);
        const fade = ease(modeT);

        /* --- layout (recalculé chaque frame depuis W/H) --- */
        const roH = small ? 54 : 62;
        const y0 = small ? 26 : 30, y1 = Math.max(y0 + 80, Hp - roH - 10);
        const sH = y1 - y0;
        const R = Math.min(sH * 0.40, W * 0.20);               // échelle du croissant
        const sig = Math.min(sH, W) * 0.08;                    // σ du blob de bruit (px)
        const ccx = W * 0.76, ccy = y0 + sH * 0.50;
        const ncx = W * 0.205, ncy = y0 + sH * 0.52;
        const mode = selMode.value;
        const nfe = mode === 'gan' ? 1 : mode === 'diff' ? slDiff.value : slFm.value;

        for (let i = 0; i < N_TGT; i++) { tpx[i] = ccx + tox[i] * R; tpy[i] = ccy + toy[i] * R; }

        /* --- champ de vitesses en fond (flow matching) --- */
        if (mode === 'fm') {
          if (fW !== W || fH !== Hp) { rebuildField(W, y0, y1); fW = W; fH = Hp; }
          ctx.save();
          ctx.globalAlpha = 0.9 * fade;
          ctx.strokeStyle = 'rgba(139,150,165,0.30)'; ctx.lineWidth = 1;
          ctx.beginPath();
          for (let k = 0; k < fn; k++) {
            ctx.moveTo(ffx[k] - fdx[k] * 6, ffy[k] - fdy[k] * 6);
            ctx.lineTo(ffx[k] + fdx[k] * 6, ffy[k] + fdy[k] * 6);
          }
          ctx.stroke();
          ctx.fillStyle = 'rgba(139,150,165,0.45)';
          ctx.beginPath();
          for (let k = 0; k < fn; k++) {
            const hx = ffx[k] + fdx[k] * 6, hy = ffy[k] + fdy[k] * 6;
            ctx.moveTo(hx + 1.2, hy); ctx.arc(hx, hy, 1.2, 0, TAU);
          }
          ctx.fill();
          ctx.restore();
        }

        /* --- annotation de transport (bas de scène) --- */
        if (!small) {
          const yA = y1 - 4;
          arrow(ctx, ncx + sig * 3.2, yA, ccx - R * 1.3, yA, { color: palette.faint, dash: [4, 5], alpha: 0.55 });
          text(ctx, 'transport : q₀(bruit) → p(données)', (ncx + ccx) / 2, yA - 8, { align: 'center', size: 10, color: palette.faint, mono: true });
        }

        /* --- blob de bruit --- */
        ctx.save();
        ctx.fillStyle = palette.dim;
        for (let i = 0; i < N_BLOB; i++) {
          ctx.globalAlpha = nba[i];
          ctx.beginPath(); ctx.arc(ncx + nbx[i] * sig, ncy + nby[i] * sig, 1.8, 0, TAU); ctx.fill();
        }
        ctx.restore();
        text(ctx, small ? 'bruit' : 'bruit (gaussien)', ncx, ncy + sig * 2.6 + 14, { align: 'center', size: 11, color: palette.dim });

        /* --- croissant des sons propres --- */
        ctx.save();
        ctx.fillStyle = palette.voice;
        const rd = small ? 1.6 : 2.1;
        for (let i = 0; i < N_TGT; i++) {
          ctx.globalAlpha = tal[i];
          ctx.beginPath(); ctx.arc(tpx[i], tpy[i], rd, 0, TAU); ctx.fill();
        }
        ctx.restore();
        if (small) {
          text(ctx, 'parole propre', ccx + R * 0.30, ccy, { align: 'center', size: 10, color: '#2dd4bfbb' });
        } else {
          text(ctx, 'distribution des spectres', ccx + R * 0.32, ccy - 7, { align: 'center', size: 11, color: '#2dd4bfcc' });
          text(ctx, 'de parole propre', ccx + R * 0.32, ccy + 8, { align: 'center', size: 11, color: '#2dd4bfcc' });
        }

        /* --- frontière du discriminateur (GAN) : oscille puis se resserre --- */
        if (mode === 'gan') {
          const tight = smoothstep(0.5 - 0.5 * Math.cos(TAU * ((t * 0.12) % 1))); // 0→1→0, période ≈ 8,3 s
          const midX = (ncx + sig * 2.4 + ccx - R * 1.2) / 2;
          ctx.save();
          ctx.strokeStyle = palette.pink; ctx.lineWidth = 1.4;
          ctx.setLineDash([6, 5]); ctx.globalAlpha = 0.85 * fade;
          ctx.beginPath();
          let topX = midX;
          for (let i = 0; i < NB; i++) {
            const u = i / (NB - 1);
            const wob = (1 - tight) * (Math.sin(u * 7 + t * 1.8) * 13 + (noise1(u * 3 + t * 0.6) - 0.5) * 18);
            const lx = midX + wob, ly = y1 - u * sH;                 // état lâche : quasi verticale
            const a = Math.PI * lerp(0.50, 1.50, u), Rb = R * 1.22;  // état appris : épouse le croissant
            const ax = ccx + Math.cos(a) * Rb + Math.sin(u * 9 + t * 2.2) * 4;
            const ay = ccy + Math.sin(a) * Rb;
            const px = lerp(lx, ax, tight), py = lerp(ly, ay, tight);
            if (u > 0.96) topX = px;
            i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          }
          ctx.stroke(); ctx.setLineDash([]); ctx.restore();
          if (!small) {
            text(ctx, 'frontière du discriminateur', topX, y0 + 10, { align: 'center', size: 10, color: palette.pink });
            text(ctx, '← faux', topX - 46, y0 + 24, { align: 'right', size: 10, color: palette.faint });
            text(ctx, 'vrai →', topX + 46, y0 + 24, { align: 'left', size: 10, color: palette.faint });
          }
        }

        /* --- particules : voyage bruit → croissant, puis respawn --- */
        const durMul = mode === 'gan' ? 0.9 : mode === 'diff' ? 1.7 : 1.15;
        const trailA = mode === 'diff' ? 0.55 : mode === 'gan' ? 0.30 : 0.40;
        ctx.save();
        ctx.lineCap = 'round';
        for (let n = 0; n < N_ACT; n++) {
          const P = parts[n];
          P.p += dt / (P.dur * durMul);
          if (P.p >= LIFE) respawn(P, 0);
          const p01 = clamp(P.p, 0, 1);
          const sx = ncx + P.sx * sig, sy = ncy + P.sy * sig;
          const tx = tpx[P.ti], ty = tpy[P.ti];
          computePos(P, p01, mode, nfe, R, sx, sy, tx, ty);

          if (dt > 0 && P.p < 1.06) {                          // traînée (coords px, vidée au resize)
            P.trx[P.th] = ppx; P.try_[P.th] = ppy;
            P.th = (P.th + 1) % TRAIL;
            if (P.tn < TRAIL) P.tn++;
          }
          if (P.tn > 1) {                                      // traînée qui s'estompe
            ctx.strokeStyle = palette.mix;
            ctx.lineWidth = mode === 'diff' ? 1.4 : 1.1;
            for (let s = 1; s < P.tn; s++) {
              const i0 = (P.th - P.tn + s - 1 + 2 * TRAIL) % TRAIL;
              const i1 = (P.th - P.tn + s + 2 * TRAIL) % TRAIL;
              ctx.globalAlpha = Math.pow(s / P.tn, 1.8) * trailA * fade;
              ctx.beginPath(); ctx.moveTo(P.trx[i0], P.try_[i0]); ctx.lineTo(P.trx[i1], P.try_[i1]); ctx.stroke();
            }
          }
          if (mode === 'fm' && !small) {                       // jalons des steps d'intégration ODE
            const k = Math.min(nfe, Math.floor(p01 * nfe));
            ctx.fillStyle = palette.blue;
            ctx.globalAlpha = 0.5 * fade;
            ctx.beginPath();
            for (let j = 1; j <= k; j++) {
              curveFM(P, j / nfe, sx, sy, tx, ty, R);
              ctx.moveTo(cvx + 1.6, cvy); ctx.arc(cvx, cvy, 1.6, 0, TAU);
            }
            ctx.fill();
          }
          if (P.p < 1) {                                       // tête en vol (violet = transformation)
            ctx.fillStyle = palette.mix;
            ctx.globalAlpha = 0.9 * fade;
            ctx.beginPath(); ctx.arc(ppx, ppy, 2.2, 0, TAU); ctx.fill();
            ctx.globalAlpha = 0.16 * fade;
            ctx.beginPath(); ctx.arc(ppx, ppy, 5.5, 0, TAU); ctx.fill();
          } else {                                             // arrivée : devient « voix » (teal)
            const q = clamp((P.p - 1) / (LIFE - 1), 0, 1);
            ctx.globalAlpha = (1 - q) * fade;
            glowDot(ctx, tx, ty, 2.6 * (1 - 0.5 * q), palette.voice);
            ctx.strokeStyle = palette.voice;
            ctx.globalAlpha = (1 - q) * 0.5 * fade;
            ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.arc(tx, ty, 3 + 11 * ease(q), 0, TAU); ctx.stroke();
          }
        }
        ctx.restore();

        /* --- chips de mode --- */
        let cx0 = 12; const cy0 = 15;
        if (mode === 'gan') {
          cx0 += chip(ctx, 'générateur : 1 forward', cx0, cy0, { color: palette.green }) + 8;
          if (!small) chip(ctx, 'training adversarial : instable', cx0, cy0, { color: palette.pink });
        } else if (mode === 'diff') {
          cx0 += chip(ctx, 'NFE = ' + nfe + ' forwards par sortie', cx0, cy0, { color: palette.mix }) + 8;
          if (!small) chip(ctx, 'le bruit σ décroît à chaque step', cx0, cy0, { color: palette.blue });
        } else {
          cx0 += chip(ctx, 'ODE : trajectoires droites → peu de steps', cx0, cy0, { color: palette.mix }) + 8;
          if (!small) chip(ctx, 'NFE = ' + nfe, cx0, cy0, { color: palette.blue });
        }
        if (W >= 880) {
          const desc = mode === 'gan'
            ? '1 saut direct easé ; le discriminateur (rose) cherche la frontière vrai / faux'
            : mode === 'diff'
              ? 'débruitage itératif : marche aléatoire dont le bruit décroît step après step'
              : 'champ de vitesses appris (flèches grises) ; intégration en quelques steps quasi droits';
          text(ctx, desc, W - 12, cy0 + 4, { align: 'right', size: 10, color: palette.dim });
        }

        /* --- readout coût temps réel (VRAI calcul) --- */
        const cost = nfe * FWD_MS;
        const ok = cost <= BUDGET_MS;
        const ratio = cost / BUDGET_MS;
        const vc = ok ? palette.green : palette.red;
        const roY = Hp - roH;
        ctx.save();
        roundRect(ctx, 8, roY, W - 16, roH - 6, 8);
        ctx.fillStyle = palette.panel; ctx.fill();
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.stroke();
        ctx.restore();
        const s1 = small ? 'coût = NFE × 2 ms (1 forward NPU)'
          : 'coût = NFE × 1 forward — sur NPU : 1 forward = 2 ms, frame audio = 8 ms';
        text(ctx, s1, 16, roY + 17, { size: 10, color: palette.dim, mono: true });
        const s2a = 'NFE = ' + nfe + '  →  ' + cost + ' ms par sortie';
        const s2b = small
          ? (ok ? '✓ temps réel' : '✗ ' + ratio.toFixed(1) + '× le budget')
          : (ok ? '✓ tient dans la frame de 8 ms (' + Math.round(ratio * 100) + ' % du budget)'
                : '✗ trop lent : ' + ratio.toFixed(1) + ' × la frame de 8 ms');
        ctx.font = '600 12px ' + U.MONO;
        const w2 = ctx.measureText(s2a).width;
        text(ctx, s2a, 16, roY + (small ? 36 : 38), { size: 12, bold: true, mono: true, color: palette.text });
        text(ctx, s2b, 16 + w2 + 14, roY + (small ? 36 : 38), { size: 12, bold: true, mono: true, color: vc });

        if (!small) {                                          // jauge 0–64 ms, repère budget à 8 ms
          const gW = Math.min(W * 0.26, 220), gX = W - 16 - gW - 6, gY = roY + 22, gH = 9;
          ctx.save();
          roundRect(ctx, gX, gY, gW, gH, 4); ctx.fillStyle = palette.panel2; ctx.fill();
          ctx.globalAlpha = 0.9;
          roundRect(ctx, gX, gY, Math.max(2, clamp(cost / SCALE_MS, 0, 1) * gW), gH, 4);
          ctx.fillStyle = vc; ctx.fill();
          ctx.globalAlpha = 0.8;
          const bx = gX + gW * (BUDGET_MS / SCALE_MS);
          ctx.strokeStyle = palette.text; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(bx, gY - 3); ctx.lineTo(bx, gY + gH + 3); ctx.stroke();
          ctx.restore();
          text(ctx, '8 ms', bx, gY - 6, { align: 'center', size: 9, color: palette.dim });
          text(ctx, 'coût / sortie', gX, gY + gH + 12, { size: 9, color: palette.faint });
          text(ctx, '64 ms', gX + gW, gY + gH + 12, { align: 'right', size: 9, color: palette.faint });
        }
      });
    },
  });
})();
