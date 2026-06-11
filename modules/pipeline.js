/* ============================================================
   Audio AI Atlas — « Pipeline complet — enhancement temps réel »
   Schéma-bloc animé d'un enhancer type DeepFilterNet QUI FONCTIONNE :
   stems oracle (parole/musique), FFT 256, masque ratio réellement
   appliqué, sorties duales VOICE/REST, re-mix « boost dialogue »,
   budget MACs/frame et latences calculés.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  const SR = 16000, NFFT = 256, BINS = NFFT / 2;
  const FOCUS_T = 3.0;        // s par étape du cycle descriptif
  const PULSE_T = 1.1;        // s par hop visuel (= 1 frame réelle / 10 ms)
  const NPU_BUDGET = 10;      // G MACs/frame dispo sur le NPU
  const LAT_BUDGET = 50;      // ms tolérés (synchro labiale)

  AtlasRegister({
    id: 'pipeline',
    title: 'Pipeline complet — enhancement temps réel',
    category: 'archi',
    icon: '◬',
    summary: 'Le schéma-bloc de bout en bout d’un enhancer temps réel : deux branches, sorties duales VOICE/REST, boost dialogue et budget NPU.',
    explain: `
      <p>Voici la chaîne complète d'un <strong>enhancer temps réel embarqué</strong> type
      <strong>DeepFilterNet</strong>. Après la <strong>STFT</strong> (fenêtre de 20&nbsp;ms, une frame
      toutes les 10&nbsp;ms), le traitement se sépare en <strong>deux branches</strong> : la branche
      « gains » compresse le spectre en <strong>32 bandes ERB</strong>, puis un encodeur convolutif et
      deux <strong>GRU</strong> (mémoire récurrente, strictement causale) prédisent un gain par bande —
      grossier mais appliqué <em>partout</em>, aligné sur la résolution de l'oreille. La branche « fin »
      fait du <em>deep filtering</em> : un petit filtre FIR d'<strong>ordre 5</strong>, dont les
      coefficients complexes sont prédits par bin, raffine la structure fine des seules basses
      fréquences (&lt;&nbsp;5&nbsp;kHz) — là où vivent les harmoniques de la voix et où l'oreille exige
      de la précision. On dépense le calcul là où il compte.</p>
      <p>La sortie est <strong>duale et cohérente</strong> : le réseau estime <code>VOICE</code>, et
      <code>REST = MIX − VOICE</code> est obtenu par <strong>soustraction spectrale</strong> — chaque
      composante du mix finit donc dans l'une ou l'autre sortie, sans perte ni double comptage. C'est
      ce qui rend le <strong>boost dialogue</strong> possible sans artefact : le téléviseur re-mixe
      simplement <code>voice·g + rest</code> (slider 0–12&nbsp;dB). Ici le masque ratio
      <em>oracle</em> est réellement appliqué à un mix parole+musique (FFT 256) : les quatre
      mini-spectrogrammes MIX, VOICE, REST et TV&nbsp;out montrent de vrais spectres, et le re-mix
      réagit réellement au slider.</p>
      <p>Tout tient dans un <strong>budget embarqué</strong> : la somme affichée (≈&nbsp;2,4&nbsp;G
      MACs/frame, dominée par l'encodeur, les GRU et les décodeurs) reste loin du budget NPU
      (~10&nbsp;G par frame de 10&nbsp;ms, soit ~1&nbsp;TOPS). Côté latence, le toggle décompose les
      ~42&nbsp;ms : 20&nbsp;ms de fenêtre d'analyse, 20&nbsp;ms de look-ahead du deep filtering, et
      ~2&nbsp;ms de calcul NPU — sous les 50&nbsp;ms tolérés pour la synchro labiale. C'est l'intérêt
      des GRU plutôt qu'une attention bidirectionnelle : aucune frame future n'est requise.</p>`,

    init(stage) {
      const ctx = stage.ctx;

      /* ---------- Contrôles ---------- */
      const slBoost = stage.addSlider({
        label: 'Boost dialogue', min: 0, max: 12, step: 0.5, value: 6,
        format: (v) => '+' + v.toFixed(1) + ' dB',
      });
      const slMusic = stage.addSlider({
        label: 'Niveau musique', min: 0, max: 100, step: 1, value: 80,
        format: (v) => v + ' %',
      });
      const tgLat = stage.addToggle({ label: 'Latences', value: false });

      /* ---------- Graphe : nodes, arêtes, cycle descriptif ---------- */
      const ND = {
        in:     { c: palette.mix,   t: 'Audio in',         s: 'mix 48 kHz',  st: 'In' },
        stft:   { c: palette.blue,  t: 'STFT',             s: 'fen. 20 ms',  st: 'STFT' },
        erb:    { c: palette.blue,  t: 'ERB 32',           s: 'bandes',      st: 'ERB' },
        enc:    { c: palette.blue,  t: 'Encoder conv',     s: 'features',    st: 'Enc' },
        gru:    { c: palette.blue,  t: 'GRU ×2',           s: 'mémoire',     st: 'GRU' },
        dgains: { c: palette.blue,  t: 'Décodeur gains',   s: '32 gains',    st: 'Gains' },
        cplx:   { c: palette.pink,  t: 'Spectre complexe', s: '0–5 kHz',     st: 'Cplx' },
        ddf:    { c: palette.pink,  t: 'Décodeur DF',      s: 'coeffs',      st: 'DF' },
        filt:   { c: palette.pink,  t: 'Filtre ordre 5',   s: 'FIR temps',   st: 'FIR 5' },
        apply:  { c: palette.violet, t: 'Masque + DF',     s: 'application', st: 'Mask' },
        istft:  { c: palette.violet, t: 'iSTFT',           s: 'overlap-add', st: 'iSTFT' },
        voice:  { c: palette.voice, t: 'VOICE',            s: 'voix isolée', st: 'VOICE' },
        rest:   { c: palette.rest,  t: 'REST',             s: 'mix − voice', st: 'REST' },
      };
      const EDGES = [ // [de, vers, couleur, pointillé?]
        ['in', 'stft', palette.mix],
        ['stft', 'erb', palette.blue], ['erb', 'enc', palette.blue],
        ['enc', 'gru', palette.blue], ['gru', 'dgains', palette.blue],
        ['dgains', 'apply', palette.blue],
        ['stft', 'cplx', palette.pink], ['cplx', 'ddf', palette.pink],
        ['ddf', 'filt', palette.pink], ['filt', 'apply', palette.pink],
        ['apply', 'istft', palette.violet],
        ['istft', 'voice', palette.voice], ['istft', 'rest', palette.rest, true],
      ];
      /* Cycle : rôle + coût plausible en G MACs/frame (somme = Σ affichée) */
      const INFO = [
        { n: 'in',     g: 0,     d: 'le mix TV entre par frames de 10 ms — traitement au fil de l’eau' },
        { n: 'stft',   g: 0.01,  d: 'fenêtre de 20 ms → spectre complexe (recouvrement 50 %)' },
        { n: 'erb',    g: 0.001, d: '32 bandes perceptuelles : l’enveloppe, pas le détail fin' },
        { n: 'enc',    g: 0.8,   d: 'convolutions : extraire des motifs temps-fréquence compacts' },
        { n: 'gru',    g: 0.6,   d: 'mémoire récurrente : contexte passé, zéro frame future requise' },
        { n: 'dgains', g: 0.5,   d: '32 gains ∈ [0,1] par frame : sculpter l’enveloppe partout' },
        { n: 'cplx',   g: 0.002, d: 'bins complexes < 5 kHz : harmoniques de la voix, phase incluse' },
        { n: 'ddf',    g: 0.4,   d: '5 coefficients complexes prédits par bin (filtre profond)' },
        { n: 'filt',   g: 0.08,  d: 'FIR ordre 5 le long du temps : reconstruit la structure fine' },
        { n: 'apply',  g: 0.005, d: 'gains ERB sur tout le spectre + deep filtering sous 5 kHz' },
        { n: 'istft',  g: 0.01,  d: 'recouvrement-addition → signal temporel, frame suivante dans 10 ms' },
        { n: 'voice',  g: 0,     d: 'sorties duales : VOICE + REST = MIX → boost réglable sans artefact' },
      ];
      const TOTAL = INFO.reduce((s, e) => s + e.g, 0); // ≈ 2.41 G MACs/frame
      const LAT = [
        { k: 'stft',  label: 'fenêtre',    ms: 20, c: palette.blue },
        { k: 'apply', label: 'look-ahead', ms: 20, c: palette.pink },
        { k: 'gru',   label: 'NPU',        ms: 2,  c: palette.green },
      ];
      const LAT_TOTAL = LAT.reduce((s, l) => s + l.ms, 0); // 42 ms

      /* ---------- Buffers & scrollers pré-alloués ---------- */
      const sigV = new Float32Array(NFFT), sigM = new Float32Array(NFFT), sigX = new Float32Array(NFFT);
      const dMix = new Float32Array(BINS), dV = new Float32Array(BINS);
      const dR = new Float32Array(BINS), dTV = new Float32Array(BINS);
      const SCRS = [
        { scr: new U.Scroller(140, 40), c: palette.mix },
        { scr: new U.Scroller(140, 40), c: palette.voice },
        { scr: new U.Scroller(140, 40), c: palette.rest },
        { scr: new U.Scroller(140, 40), c: palette.green },
      ];

      const posmod = (x, m) => ((x % m) + m) % m;
      const tr = (m) => Math.pow(Math.min(1, m * 3), 0.55); // magnitude → 0..1 affichable
      function set(k, x, y, w, h) { const n = ND[k]; n.x = x; n.y = y; n.w = w; n.h = h; }
      /* Ancrage d'une arête sur les bords des deux boîtes */
      function anch(a, b) {
        const dx = b.x - a.x, dy = b.y - a.y;
        if (Math.abs(dx) >= Math.abs(dy)) {
          const s = dx >= 0 ? 1 : -1;
          return [a.x + s * a.w / 2, a.y, b.x - s * b.w / 2, b.y];
        }
        const s = dy >= 0 ? 1 : -1;
        return [a.x, a.y + s * a.h / 2, b.x, b.y - s * b.h / 2];
      }

      /* ---------- Boucle de rendu ---------- */
      stage.onFrame((t, dt) => {
        stage.clear();
        const W = stage.W, H = stage.H, narrow = W < 560, M = 10;
        const db = slBoost.value, g = Math.pow(10, db / 20); // gain linéaire VRAI
        const mg = slMusic.value / 100, latOn = tgLat.value;

        /* === Signal réel simplifié : stems oracle + FFT 256 === */
        for (let i = 0; i < NFFT; i++) {
          const ts = t + i / SR;
          sigV[i] = U.gen.speech(ts);
          sigM[i] = U.gen.music(ts) * mg;
          sigX[i] = sigV[i] + sigM[i];
        }
        const magV = U.rfftMag(sigV), magM = U.rfftMag(sigM), magX = U.rfftMag(sigX);
        for (let b = 0; b < BINS; b++) {
          const mask = magV[b] / (magV[b] + magM[b] + 1e-9); // masque ratio oracle
          const vo = magX[b] * mask, ro = magX[b] - vo;       // VOICE + REST = MIX exactement
          dMix[b] = tr(magX[b]); dV[b] = tr(vo); dR[b] = tr(ro);
          dTV[b] = tr(vo * g + ro);                           // re-mix boost dialogue RÉEL
        }
        if (dt > 0) for (let i = 0; i < 4; i++) SCRS[i].scr.push([dMix, dV, dR, dTV][i]);

        /* === Mise en page (recalculée chaque frame) === */
        const scrH = U.clamp(H * 0.13, 26, 42), labY = M + 8, scrY = M + 12;
        let sw, xs;
        if (narrow) {
          sw = (W - 2 * M - 18) / 4;
          xs = [M, M + sw + 6, M + 2 * (sw + 6), M + 3 * (sw + 6)];
        } else {
          sw = U.clamp((W - 2 * M) / 4 - 12, 70, 170);
          const tvX = W - M - sw, reX = tvX - sw - 8, voX = reX - sw - 8;
          xs = [M, voX, reX, tvX]; // MIX au-dessus du node Audio in (à gauche)
        }
        const bannerH = narrow ? 30 : 34, latH = latOn ? 18 : 0;
        const dy0 = scrY + scrH + 14, dy1 = H - M - bannerH - latH - 6;
        const dh = Math.max(40, dy1 - dy0);

        if (!narrow) { /* 1 rangée + 2 branches parallèles */
          const cw = (W - 2 * M) / 9, nw = Math.min(cw - 10, 104);
          const nh = U.clamp(dh * 0.26, 24, 40);
          const cx = (i) => M + cw * (i + 0.5);
          const midY = dy0 + dh * 0.5, topY = dy0 + dh * 0.16, botY = dy0 + dh * 0.84;
          const nwB = Math.min(cw * 1.3 - 10, 122);
          set('in', cx(0), midY, nw, nh); set('stft', cx(1), midY, nw, nh);
          set('erb', cx(2), topY, nw, nh); set('enc', cx(3), topY, nw, nh);
          set('gru', cx(4), topY, nw, nh); set('dgains', cx(5), topY, nw, nh);
          set('cplx', cx(2.3), botY, nwB, nh); set('ddf', cx(3.65), botY, nwB, nh);
          set('filt', cx(5), botY, nw, nh);
          set('apply', cx(6), midY, nw, nh); set('istft', cx(7), midY, nw, nh);
          set('voice', cx(8), topY, nw, nh); set('rest', cx(8), botY, nw, nh);
        } else {       /* 2 rangées si étroit */
          const cw = (W - 2 * M) / 6, nw = cw - 6;
          const nh = U.clamp(dh * 0.24, 22, 34);
          const cx = (i) => M + cw * (i + 0.5);
          const r1 = dy0 + dh * 0.22, r2 = dy0 + dh * 0.72;
          set('in', cx(0), r1, nw, nh); set('stft', cx(1), r1, nw, nh);
          set('erb', cx(2), r1, nw, nh); set('enc', cx(3), r1, nw, nh);
          set('gru', cx(4), r1, nw, nh); set('dgains', cx(5), r1, nw, nh);
          set('cplx', cx(0), r2, nw, nh); set('ddf', cx(1), r2, nw, nh);
          set('filt', cx(2), r2, nw, nh); set('apply', cx(3), r2, nw, nh);
          set('istft', cx(4), r2, nw, nh);
          set('voice', cx(5), r2 - nh * 0.55, nw, nh * 0.95);
          set('rest', cx(5), r2 + nh * 0.55, nw, nh * 0.95);
        }

        /* === Mini-Scrollers vivants (MIX, VOICE, REST, TV out) === */
        const labels = narrow
          ? ['MIX', 'VOICE', 'REST', 'TV +' + db.toFixed(0) + ' dB']
          : ['MIX (entrée)', 'VOICE = mix × masque', 'REST = mix − voice',
             'TV out — voix +' + db.toFixed(1) + ' dB (×' + g.toFixed(1) + ')'];
        for (let i = 0; i < 4; i++) {
          SCRS[i].scr.draw(ctx, xs[i], scrY, sw, scrH);
          ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
          ctx.strokeRect(xs[i] + 0.5, scrY + 0.5, sw - 1, scrH - 1);
          U.text(ctx, labels[i], xs[i], labY, { size: narrow ? 8 : 9.5, color: SCRS[i].c, bold: true });
        }

        /* === Pulses synchrones : 1 hop = 1 frame de 10 ms === */
        const ph = posmod(t / PULSE_T, 1);
        const flash = Math.max(ph > 0.85 ? (ph - 0.85) / 0.15 : 0,
                               ph < 0.12 ? (1 - ph / 0.12) * 0.8 : 0) * 0.55;
        for (const e of EDGES) {
          const [x1, y1, x2, y2] = anch(ND[e[0]], ND[e[1]]);
          U.arrow(ctx, x1, y1, x2, y2,
            { color: e[2], alpha: 0.55, lw: 1.2, head: 5, dash: e[3] ? [4, 3] : null });
          U.glowDot(ctx, U.lerp(x1, x2, ph), U.lerp(y1, y2, ph), 2.2, e[2]);
        }

        /* === Cycle descriptif : focus easé toutes les ~3 s === */
        const fi = Math.floor(posmod(t, FOCUS_T * INFO.length) / FOCUS_T);
        const fp = posmod(t, FOCUS_T) / FOCUS_T;
        const env = U.smoothstep(fp / 0.15) * (1 - U.smoothstep((fp - 0.82) / 0.18));
        const fk = INFO[fi].n;

        /* === Nodes (halo continu = focus easé ∨ pulse traversant) === */
        for (const k in ND) {
          const n = ND[k];
          const glow = Math.max(k === fk ? env : 0, flash);
          if (glow > 0.04) {
            ctx.save();
            ctx.shadowColor = n.c; ctx.shadowBlur = 20 * glow;
            ctx.globalAlpha = 0.5 * glow;
            U.roundRect(ctx, n.x - n.w / 2, n.y - n.h / 2, n.w, n.h, 8);
            ctx.strokeStyle = n.c; ctx.lineWidth = 2; ctx.stroke();
            ctx.restore();
          }
          U.node(ctx, n.x - n.w / 2, n.y - n.h / 2, n.w, n.h, {
            title: narrow ? n.st : n.t, sub: narrow ? null : n.s,
            color: n.c, active: glow > 0.45, size: narrow ? 9 : 11,
          });
        }

        /* === Annotations : branches + cadence === */
        if (!narrow) {
          U.text(ctx, 'branche « gains » — grossier, sur tout le spectre',
            (ND.erb.x + ND.dgains.x) / 2, ND.erb.y - ND.erb.h / 2 - 8,
            { size: 10, color: palette.blue, align: 'center' });
          U.text(ctx, 'branche « fin » — précis, < 5 kHz seulement',
            (ND.cplx.x + ND.filt.x) / 2, ND.cplx.y + ND.cplx.h / 2 + 14,
            { size: 10, color: palette.pink, align: 'center' });
        }
        U.text(ctx, '1 frame / 10 ms', (ND.in.x + ND.stft.x) / 2,
          ND.in.y + ND.in.h / 2 + (latOn && !narrow ? 24 : 14),
          { size: narrow ? 8 : 9, color: palette.faint, align: 'center', mono: true });

        /* === Latences : contributions sous les étages + jauge empilée === */
        if (latOn) {
          if (!narrow) {
            for (const l of LAT) {
              const n = ND[l.k];
              U.text(ctx, l.label + ' ' + l.ms + ' ms', n.x, n.y + n.h / 2 + 12,
                { size: 9, color: l.c, align: 'center' });
            }
          }
          const ly = dy1 + 6, lh = 10;
          const bw = Math.min(W * 0.4, 240), bx = narrow ? M : M + 150;
          if (!narrow) U.text(ctx, 'Latence algorithmique', M, ly + lh - 1, { size: 10, color: palette.dim });
          ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
          U.roundRect(ctx, bx, ly, bw, lh, 3); ctx.stroke();
          let acc = 0;
          for (const l of LAT) {
            const wseg = (l.ms / LAT_BUDGET) * bw;
            ctx.fillStyle = l.c; ctx.globalAlpha = 0.8;
            ctx.fillRect(bx + acc + 1, ly + 1, Math.max(wseg - 2, 1), lh - 2);
            acc += wseg;
          }
          ctx.globalAlpha = 1;
          U.text(ctx, '≈ ' + LAT_TOTAL + ' ms / budget ' + LAT_BUDGET + ' ms', bx + bw + 8, ly + lh - 1,
            { size: 10, color: LAT_TOTAL <= LAT_BUDGET ? palette.green : palette.red, mono: true });
        }

        /* === Bandeau bas : rôle du bloc + Σ MACs + jauge budget NPU === */
        const by = H - M - bannerH, bw2 = W - 2 * M;
        U.roundRect(ctx, M, by, bw2, bannerH, 6);
        ctx.fillStyle = palette.panel; ctx.fill();
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.stroke();

        const rw = narrow ? 112 : 230; // zone Σ à droite
        const e = INFO[fi];
        const macs = e.g >= 0.01 ? ' · ~' + e.g.toFixed(2) + ' G MACs/frame'
          : e.g > 0 ? ' · < 0,01 G MACs/frame' : '';
        ctx.save();
        ctx.beginPath(); ctx.rect(M + 6, by + 2, bw2 - rw - 12, bannerH - 4); ctx.clip();
        ctx.globalAlpha = Math.min(1, env * 1.5);
        U.text(ctx, ND[e.n].t, M + 10, by + (narrow ? 12 : 14),
          { size: narrow ? 10 : 12, bold: true, color: ND[e.n].c });
        U.text(ctx, e.d + macs, M + 10, by + (narrow ? 23 : 28),
          { size: narrow ? 8.5 : 10, color: palette.dim });
        ctx.restore();
        ctx.globalAlpha = 1;

        U.text(ctx, narrow ? 'Σ ' + TOTAL.toFixed(2) + ' G MACs' : 'Σ ≈ ' + TOTAL.toFixed(2) + ' G MACs/frame',
          W - M - 10, by + 13, { size: narrow ? 9 : 11, bold: true, color: palette.text, align: 'right', mono: true });
        const gw = narrow ? 70 : 110, gx = W - M - 10 - gw, gy = by + bannerH - 12;
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        U.roundRect(ctx, gx, gy, gw, 7, 3); ctx.stroke();
        ctx.fillStyle = palette.green; ctx.globalAlpha = 0.85;
        ctx.fillRect(gx + 1, gy + 1, Math.max((TOTAL / NPU_BUDGET) * gw - 2, 1), 5);
        ctx.globalAlpha = 1;
        if (!narrow) U.text(ctx, 'budget NPU ~' + NPU_BUDGET + ' G', gx - 8, gy + 7,
          { size: 8.5, color: palette.faint, align: 'right' });
      });
    },
  });
})();
