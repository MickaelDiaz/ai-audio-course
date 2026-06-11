/* ============================================================
   Audio AI Atlas — module « RNN, GRU, LSTM — la mémoire récurrente »
   Une cellule qui se relit elle-même : état caché, portes, déroulé
   temporel et propagation (ou mort) d'un pulse selon la persistance.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;
  const sigm = (v) => 1 / (1 + Math.exp(-v));

  AtlasRegister({
    id: 'rnn',
    title: 'RNN, GRU, LSTM — la mémoire récurrente',
    category: 'layers',
    icon: '↻',
    summary: 'Une cellule, un état caché et des portes : comment un réseau se souvient du passé en O(1) par pas.',
    explain: `
      <p>Une <strong>cellule récurrente</strong> ne voit jamais la séquence entière : à chaque pas, elle lit
      une frame audio x<sub>t</sub> et son propre état caché h<sub>t−1</sub>, puis produit h<sub>t</sub>.
      Toute la mémoire tient dans un seul vecteur de n floats — un coût <strong>O(1) par pas</strong>,
      là où l'attention d'un Transformer doit relire les T positions passées (O(T)). Pour un flux audio
      potentiellement infini, cette différence est décisive.</p>
      <p>La faiblesse du RNN simple est le <strong>vanishing gradient</strong> : à l'entraînement, le gradient
      remonte le temps en étant multiplié à chaque pas par un facteur souvent &lt; 1. À ×0.5 par pas, il ne
      reste après 8 pas que 0.5<sup>8</sup> ≈ 0,4 % du signal — impossible d'apprendre une dépendance longue.
      Les <strong>portes</strong> corrigent cela : la <strong>GRU</strong> dose avec z (<em>update</em>) ce qui est
      réécrit dans l'état et avec r (<em>reset</em>) ce qui est relu ; la <strong>LSTM</strong> ajoute un état C
      séparé, un « tapis roulant » traversé par de simples additions et un produit par la porte f — un chemin
      quasi sans atténuation (0.95<sup>8</sup> ≈ 66 %) que le gradient emprunte comme une autoroute.</p>
      <p>Le prix des portes se lit dans les paramètres : pour n=256 unités et m=128 entrées, une GRU pèse
      <code>3(n²+nm+n) ≈ 296 k</code> paramètres et une LSTM <code>4(n²+nm+n) ≈ 394 k</code>.
      C'est pourquoi le <strong>streaming audio embarqué</strong> (débruitage type RNNoise, VAD,
      <em>keyword spotting</em>) adore la GRU : un état à transporter minuscule (256 floats = 0,5 Ko en FP16),
      25 % de calcul en moins que la LSTM, uniquement des sigmoid/tanh et des produits matrice-vecteur très
      efficaces sur NPU, et une <strong>latence de zéro frame</strong> — la sortie tombe dès que la frame
      arrive, sans aucun lookahead.</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const N_H = 10, N_BANDS = 8, N_STEPS = 9, PERIOD = 0.8;
      const FFT_N = 256, SR = 16000;

      /* ---------- buffers pré-alloués (jamais dans onFrame) ---------- */
      const sig = new Float32Array(FFT_N);        // fenêtre audio à analyser
      const x = new Float32Array(N_BANDS);        // frame spectrale courante (8 bandes)
      const h = new Float32Array(N_H);            // état caché affiché
      const hTmp = new Float32Array(N_H);
      const xProj = new Float32Array(N_H);        // projection 8 → 10
      const volHist = new Float32Array(N_STEPS);  // activité récente (vue déroulée)
      const Wp = new Float32Array(N_H * N_BANDS); // poids de projection déterministes
      for (let i = 0; i < Wp.length; i++) Wp[i] = (U.hash(i * 1.618 + 3.7) - 0.5) * 1.7;

      /* ---------- nombres RÉELS affichés ---------- */
      const n = 256, m = 128;
      const base = n * n + n * m + n;                       // n² + n·m + n = 98 560
      const PAR = { rnn: base, gru: 3 * base, lstm: 4 * base };
      const stateKo = n * 2 / 1024;                         // n floats FP16 → 0.5 Ko

      /* ---------- état dynamique ---------- */
      const gates = { z: 0.5, r: 0.5, f: 0.5, i: 0.5, o: 0.5 }; // affichées (lissées)
      const gateT = { z: 0.5, r: 0.5, f: 0.5, i: 0.5, o: 0.5 }; // cibles
      let vol = 0, lastStep = -1, flashT = -9, lastT = 0, pulseT0 = -1;

      /* Nouvelle frame d'entrée : vrai spectre de gen.speech (FFT 256 @ 16 kHz) */
      function genFrame(t) {
        const t0 = t % 64; // temps audio borné (précision sin sur de longues sessions)
        for (let k = 0; k < FFT_N; k++) sig[k] = U.gen.speech(t0 + k / SR);
        const mag = U.rfftMag(sig);
        const per = mag.length / N_BANDS;
        let acc = 0;
        for (let b = 0; b < N_BANDS; b++) {
          let s = 0;
          for (let k = 0; k < per; k++) s += mag[b * per + k];
          x[b] = U.clamp(s / per * 3.5, 0, 1);
          acc += x[b];
        }
        vol = acc / N_BANDS; // « volume » : pilote les portes
      }
      /* La frame atteint la cellule : récurrence réelle h = tanh(0.9·h_rot + 0.5·x_proj) */
      function applyFrame() {
        for (let j = 0; j < N_H; j++) {
          let s = 0;
          for (let b = 0; b < N_BANDS; b++) s += Wp[j * N_BANDS + b] * x[b];
          xProj[j] = Math.tanh(s);
        }
        for (let j = 0; j < N_H; j++) hTmp[j] = Math.tanh(0.9 * h[(j + 1) % N_H] + 0.5 * xProj[j]);
        h.set(hTmp);
        gateT.z = sigm(6.0 * vol - 1.6);  // update : s'ouvre quand l'entrée est forte
        gateT.r = sigm(2.6 - 5.0 * vol);  // reset : relâche le passé quand l'entrée domine
        gateT.f = sigm(2.4 - 3.5 * vol);  // forget : conserve C quand l'entrée est faible
        gateT.i = sigm(7.0 * vol - 2.2);
        gateT.o = sigm(5.0 * vol - 1.0);
        for (let k = 0; k < N_STEPS - 1; k++) volHist[k] = volHist[k + 1];
        volHist[N_STEPS - 1] = vol;
      }
      for (let k = 0; k < 3; k++) { genFrame(k * PERIOD); applyFrame(); } // pré-chauffe
      Object.assign(gates, gateT);

      /* ---------- contrôles ---------- */
      const PRESET = { 'RNN simple': 0.5, 'GRU': 0.93, 'LSTM': 0.95 };
      const KEY = { 'RNN simple': 'rnn', 'GRU': 'gru', 'LSTM': 'lstm' };
      const selArch = stage.addSelect({
        label: 'Architecture',
        options: ['RNN simple', 'GRU', 'LSTM'],
        value: 'GRU',
        onChange: (v) => { slDecay.value = PRESET[v]; },
      });
      const slDecay = stage.addSlider({
        label: 'Persistance de la mémoire', min: 0.3, max: 0.98, step: 0.01,
        value: PRESET['GRU'], format: (v) => '×' + (+v).toFixed(2),
      });
      stage.addButton({ label: 'Injecter un pulse', onClick: () => { pulseT0 = lastT; } });

      /* ---------- jauge circulaire 0–1 (portes) ---------- */
      function gauge(gx, gy, r, val, color, lbl) {
        ctx.save();
        ctx.strokeStyle = palette.faint; ctx.lineWidth = 3; ctx.globalAlpha = 0.4;
        ctx.beginPath(); ctx.arc(gx, gy, r, 0, U.TAU); ctx.stroke();
        ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = 3.2; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.arc(gx, gy, r, -Math.PI / 2, -Math.PI / 2 + U.clamp(val, 0, 1) * U.TAU);
        ctx.stroke();
        ctx.restore();
        U.text(ctx, val.toFixed(2), gx, gy + 1, { align: 'center', baseline: 'middle', size: r > 13 ? 10 : 9, mono: true, bold: true, color });
        U.text(ctx, lbl, gx, gy + r + 11, { size: 9, align: 'center', color: palette.dim });
      }

      stage.onFrame((t, dt) => {
        lastT = t;
        const step = Math.floor(t / PERIOD);
        if (step !== lastStep) {
          if (lastStep >= 0) { applyFrame(); flashT = t; } // la frame en vol arrive
          genFrame(t);                                      // la suivante part de la gauche
          lastStep = step;
        }
        const sm = Math.min(1, dt * 5);
        for (const g in gates) gates[g] += (gateT[g] - gates[g]) * sm;

        stage.clear();
        const W = stage.W, H = stage.H, small = W < 560;
        const M = small ? 8 : 16;
        const arch = KEY[selArch.value] || 'gru';
        const d = slDecay.value;
        const phase = (t % PERIOD) / PERIOD;
        const flashA = U.clamp(1 - (t - flashT) / 0.35, 0, 1);

        U.text(ctx, 'Cellule récurrente — ' + selArch.value, M, 16, { size: small ? 11 : 13, bold: true });
        if (!small) U.text(ctx, '1 frame / ' + U.fmt.ms(PERIOD) + ' · état n=' + n, W - M, 16, { size: 10, color: palette.dim, align: 'right' });

        /* ============ ZONE HAUTE (~55 %) : la cellule en grand ============ */
        const ty = 24, th = H * 0.56 - ty;
        const cw = U.clamp(W * 0.26, 116, 220);
        const ch = U.clamp(th * 0.8, 78, 170);
        const ccx = W * (small ? 0.5 : 0.42);
        const cx0 = ccx - cw / 2, cy0 = ty + (th - ch) / 2 + 6;
        const midY = cy0 + ch * 0.46;

        /* corps de la cellule (node arrondi, halo au moment de la mise à jour) */
        ctx.save();
        if (flashA > 0) { ctx.shadowColor = palette.mix; ctx.shadowBlur = 16 * flashA; }
        U.roundRect(ctx, cx0, cy0, cw, ch, 14);
        ctx.fillStyle = palette.panel; ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = palette.mix; ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.6 + 0.4 * flashA; ctx.stroke();
        ctx.restore();
        U.text(ctx, selArch.value + ' ↻', ccx, cy0 + 12, { size: 10, bold: true, align: 'center', color: palette.mix });

        /* schéma interne selon l'architecture */
        if (arch === 'rnn') {
          const nw = Math.min(58, cw * 0.4), nh = Math.min(26, ch * 0.26);
          U.arrow(ctx, cx0 + 6, midY, ccx - nw / 2 - 3, midY, { color: palette.voice, lw: 1.4 });
          U.arrow(ctx, ccx, cy0 + ch - 8, ccx, midY + nh / 2 + 3, { color: palette.green, lw: 1.4 });
          U.arrow(ctx, ccx + nw / 2 + 3, midY, cx0 + cw - 6, midY, { color: palette.green, lw: 1.4 });
          U.node(ctx, ccx - nw / 2, midY - nh / 2, nw, nh, { title: 'tanh', color: palette.mix, active: flashA > 0.3, size: 11 });
          U.text(ctx, 'xₜ', cx0 + 9, midY - 6, { size: 10, color: palette.voice, mono: true });
          U.text(ctx, 'hₜ₋₁', ccx + 5, cy0 + ch - 11, { size: 9, color: palette.green, mono: true });
          U.text(ctx, 'hₜ', cx0 + cw - 22, midY - 6, { size: 10, color: palette.green, mono: true });
        } else if (arch === 'gru') {
          const gr = U.clamp(ch * 0.17, 11, 17);
          gauge(ccx - cw * 0.22, cy0 + ch * 0.38, gr, gates.z, palette.blue, 'z · update');
          gauge(ccx + cw * 0.22, cy0 + ch * 0.38, gr, gates.r, palette.orange, 'r · reset');
          const nw = Math.min(54, cw * 0.38), nh = Math.min(22, ch * 0.2);
          U.node(ctx, ccx - nw / 2, cy0 + ch * 0.8 - nh / 2, nw, nh, { title: 'tanh h̃', color: palette.mix, active: flashA > 0.3, size: 10 });
        } else { /* lstm : 3 portes + tapis roulant C */
          const yC = cy0 + ch * 0.26;
          ctx.save();
          ctx.strokeStyle = palette.rest; ctx.lineWidth = 3; ctx.globalAlpha = 0.85;
          ctx.beginPath(); ctx.moveTo(cx0 - 12, yC); ctx.lineTo(cx0 + cw + 4, yC); ctx.stroke();
          ctx.restore();
          U.arrow(ctx, cx0 + cw + 4, yC, cx0 + cw + 14, yC, { color: palette.rest, lw: 3, head: 7 });
          for (let i = 0; i < 2; i++) {
            const p = (t * 0.35 + i * 0.5) % 1;
            U.glowDot(ctx, cx0 - 12 + p * (cw + 26), yC, 2.5, palette.rest);
          }
          if (!small) U.text(ctx, 'C — « tapis roulant »', ccx, yC - 7, { size: 9, color: palette.rest, align: 'center' });
          const gr = U.clamp(ch * 0.14, 9, 14), gy = cy0 + ch * 0.62;
          gauge(ccx - cw * 0.3, gy, gr, gates.f, palette.red, 'f · forget');
          gauge(ccx, gy, gr, gates.i, palette.green, 'i · input');
          gauge(ccx + cw * 0.3, gy, gr, gates.o, palette.blue, 'o · output');
        }

        /* flux d'entrée : colonne spectrale 8 bandes (magma) qui arrive de la gauche */
        const colW = small ? 9 : 12;
        const cellHpx = U.clamp(ch / 14, 5, 8), colH = cellHpx * N_BANDS;
        const startX = M + 4, endX = cx0 - colW - 8;
        ctx.save();
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
        ctx.beginPath(); ctx.moveTo(startX, midY); ctx.lineTo(cx0 - 4, midY); ctx.stroke();
        ctx.restore();
        const px = U.lerp(startX, Math.max(endX, startX), U.ease(phase));
        ctx.save();
        ctx.globalAlpha = 0.4 + 0.6 * U.smoothstep(Math.min(phase * 4, 1));
        for (let b = 0; b < N_BANDS; b++) {
          ctx.fillStyle = U.magma(x[b]);
          ctx.fillRect(px, midY - colH / 2 + (N_BANDS - 1 - b) * cellHpx, colW, cellHpx - 1);
        }
        ctx.restore();
        if (!small) {
          U.text(ctx, 'frames audio (spectre 8 bandes, FFT réelle)', startX, midY + colH / 2 + 14, { size: 9, color: palette.dim });
          U.text(ctx, 'volume = ' + vol.toFixed(2) + ' → portes', startX, midY + colH / 2 + 26, { size: 9, color: palette.voice });
        }

        /* état caché h : colonne de 10 cellules viridis, flash à chaque mise à jour */
        const hw = small ? 12 : 16;
        const hColH = U.clamp(ch * 0.95, 56, 160), hCellH = hColH / N_H;
        const hx = cx0 + cw + U.clamp(W * 0.045, 16, 40);
        const hy = cy0 + ch / 2 - hColH / 2;
        for (let j = 0; j < N_H; j++) {
          ctx.fillStyle = U.viridis((h[j] + 1) / 2);
          ctx.fillRect(hx, hy + j * hCellH, hw, hCellH - 1);
        }
        if (flashA > 0) {
          ctx.save();
          ctx.strokeStyle = palette.green; ctx.globalAlpha = flashA; ctx.lineWidth = 2;
          ctx.strokeRect(hx - 2.5, hy - 2.5, hw + 5, hColH + 5);
          ctx.restore();
        }
        U.arrow(ctx, cx0 + cw + 2, midY, hx - 3, midY, { color: palette.green, lw: 1.4 });
        U.text(ctx, 'hₜ', hx + hw / 2, hy - 6, { size: 10, align: 'center', color: palette.green, mono: true });

        /* boucle de récurrence : h repart sous la cellule (↻) */
        const loopY = cy0 + ch + 11;
        ctx.save();
        ctx.strokeStyle = palette.green; ctx.globalAlpha = 0.55; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(hx + hw / 2, hy + hColH + 2);
        ctx.lineTo(hx + hw / 2, loopY);
        ctx.lineTo(cx0 + cw * 0.15, loopY);
        ctx.stroke();
        ctx.restore();
        U.arrow(ctx, cx0 + cw * 0.15, loopY, cx0 + cw * 0.15, cy0 + ch + 2, { color: palette.green, lw: 1.3, alpha: 0.55, head: 5 });
        U.text(ctx, 'hₜ₋₁ ↻ récurrence', ccx + cw * 0.1, loopY - 4, { size: 9, align: 'center', color: palette.green });

        /* readouts VRAIS : paramètres et taille de l'état */
        if (!small) {
          const rx = Math.min(hx + hw + U.clamp(W * 0.05, 24, 80), W - 172);
          let ry = cy0 + 4;
          const line = (s, c, b) => { U.text(ctx, s, rx, ry, { size: 10, mono: true, color: c, bold: !!b }); ry += 14; };
          line('n=256 (état) · m=128 (entrée)', palette.dim);
          line('RNN  : 1·(n²+nm+n) = ' + U.fmt.k(PAR.rnn), arch === 'rnn' ? palette.text : palette.faint, arch === 'rnn');
          line('GRU  : 3·(n²+nm+n) = ' + U.fmt.k(PAR.gru), arch === 'gru' ? palette.text : palette.faint, arch === 'gru');
          line('LSTM : 4·(n²+nm+n) = ' + U.fmt.k(PAR.lstm), arch === 'lstm' ? palette.text : palette.faint, arch === 'lstm');
          ry += 3;
          line('état à transporter : ' + n + ' floats', palette.voice);
          line('= ' + n + ' × 2 o = ' + stateKo.toFixed(1) + ' Ko (FP16)', palette.voice);
          line('affiché : ' + N_H + ' dims / ' + n, palette.faint);
        } else {
          U.text(ctx, 'GRU ' + U.fmt.k(PAR.gru) + ' · LSTM ' + U.fmt.k(PAR.lstm) + ' params · état ' + stateKo.toFixed(1) + ' Ko',
            W / 2, ty + th + 10, { size: 9, align: 'center', color: palette.dim });
        }

        /* ============ ZONE BASSE (~40 %) : vue déroulée sur 9 pas ============ */
        const by = H * 0.62, bh = H - by - (small ? 6 : 10);
        U.frame(ctx, M, by, W - 2 * M, bh, small ? undefined : 'Déroulé sur 9 pas de temps — la même cellule, copiée');
        const innerW = W - 2 * M - 16, dx = innerW / N_STEPS;
        const s = U.clamp(Math.min(dx * 0.55, bh * 0.36), 14, 30);
        const my = by + bh * 0.42;
        for (let k = 0; k < N_STEPS; k++) {
          const xk = M + 8 + dx * (k + 0.5);
          U.roundRect(ctx, xk - s / 2, my - s / 2, s, s, 5);
          ctx.fillStyle = palette.panel2; ctx.fill();
          ctx.strokeStyle = U.viridis(0.25 + volHist[k] * 0.7); ctx.lineWidth = 1.2; ctx.stroke();
          if (s >= 18) U.text(ctx, '↻', xk, my + 1, { size: 11, align: 'center', baseline: 'middle', color: palette.dim });
          if (k < N_STEPS - 1) U.arrow(ctx, xk + s / 2 + 2, my, xk + dx - s / 2 - 3, my, { color: palette.faint, lw: 1.2, head: 5 });
          if (!small) U.arrow(ctx, xk, my + s / 2 + 13, xk, my + s / 2 + 3, { color: palette.voice, lw: 1, head: 4, alpha: 0.5 });
          U.text(ctx, k === N_STEPS - 1 ? 't' : 't−' + (N_STEPS - 1 - k), xk, my + s / 2 + (small ? 12 : 24), { size: 9, align: 'center', color: palette.faint, mono: true });
        }
        /* la mise à jour « live » illumine le pas courant (droite) */
        if (flashA > 0) {
          const xr = M + 8 + dx * (N_STEPS - 0.5);
          ctx.save();
          ctx.strokeStyle = palette.green; ctx.globalAlpha = flashA; ctx.lineWidth = 1.6;
          U.roundRect(ctx, xr - s / 2 - 2, my - s / 2 - 2, s + 4, s + 4, 6); ctx.stroke();
          ctx.restore();
        }

        /* pulse rose : part du pas le plus ancien, atténué d'un facteur RÉEL ×d par pas */
        if (pulseT0 >= 0) {
          const prog = (t - pulseT0) / 0.5;
          if (prog > N_STEPS + 0.3) {
            pulseT0 = -1;
          } else {
            const kInt = Math.min(Math.floor(prog), N_STEPS - 1);
            const pos = Math.min(kInt + U.ease(prog - kInt), N_STEPS - 1);
            const amp = Math.pow(d, pos);
            const xp = M + 8 + dx * (pos + 0.5);
            const fade = 1 - U.clamp((prog - (N_STEPS - 1)) / 1.2, 0, 1);
            for (let k = 0; k <= kInt; k++) { // trace résiduelle atténuée
              ctx.save();
              ctx.globalAlpha = Math.pow(d, k) * 0.3 * fade;
              ctx.fillStyle = palette.pink;
              ctx.beginPath(); ctx.arc(M + 8 + dx * (k + 0.5), my, 2.5, 0, U.TAU); ctx.fill();
              ctx.restore();
            }
            ctx.save();
            ctx.globalAlpha = (0.25 + 0.75 * amp) * fade;
            U.glowDot(ctx, xp, my, 3 + 9 * amp, palette.pink);
            ctx.restore();
            if (fade > 0.05) {
              U.text(ctx, '×' + d.toFixed(2) + ' par pas → amplitude ' + amp.toFixed(3),
                xp, my - s / 2 - 8, { size: 10, align: pos < 4.5 ? 'left' : 'right', color: palette.pink, bold: true, mono: true });
            }
          }
        }

        /* verdict chiffré : que reste-t-il du pulse après 8 pas ? */
        const p8 = Math.pow(d, N_STEPS - 1);
        const verdict = d < 0.7 ? '≈ RNN simple : le signal meurt' : d < 0.9 ? 'zone intermédiaire' : '≈ portes GRU/LSTM : autoroute du gradient';
        const vc = d < 0.7 ? palette.red : d < 0.9 ? palette.rest : palette.green;
        U.text(ctx, 'persistance ×' + d.toFixed(2) + ' → après 8 pas : ' + (p8 >= 0.01 ? p8.toFixed(2) : p8.toExponential(1)) + '  ·  ' + verdict,
          W / 2, by + bh - 8, { size: small ? 9 : 10.5, align: 'center', color: vc });
      });
    },
  });
})();
