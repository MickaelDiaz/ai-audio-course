/* ============================================================
   Audio AI Atlas — « Pipeline complet — enhancement temps réel »
   Schéma-bloc animé d'un enhancer type DeepFilterNet QUI FONCTIONNE :
   stems oracle (parole/musique), FFT 256, masque ratio réellement
   appliqué, sorties duales VOICE/REST, re-mix « boost dialogue »,
   budget MACs/frame et latences calculés.

   Mise en page responsive : sur desktop (≥ 560), une rangée + deux
   branches parallèles, inchangée. Sur mobile (stage.compact), tout est
   EMPILÉ verticalement et le texte agrandi via stage.fs(...) : les quatre
   mini-spectres en haut, le graphe en deux colonnes lisibles au milieu,
   puis la fiche descriptive et le budget NPU en GRAND en bas.
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
      <p>Voici la chaîne complète d'un <strong>enhancer <dfn class="term" data-term="streaming">temps réel</dfn> embarqué</strong> type
      <strong>DeepFilterNet</strong>. Après la <dfn class="term" data-term="stft">STFT</dfn> (<dfn class="term" data-term="window">fenêtre</dfn> de 20&nbsp;ms, une
      <dfn class="term" data-term="frame">trame</dfn> toutes les 10&nbsp;ms), le traitement se sépare en <strong>deux branches</strong> : la branche
      « gains » compresse le <dfn class="term" data-term="spectre">spectre</dfn> en <strong>32 <dfn class="term" data-term="erb">bandes ERB</dfn></strong>, puis un
      <dfn class="term" data-term="encoder-decoder">encodeur</dfn> <dfn class="term" data-term="convolution">convolutif</dfn> et
      deux <dfn class="term" data-term="gru">GRU</dfn> (mémoire récurrente, strictement <dfn class="term" data-term="causal">causale</dfn>) prédisent un gain par bande —
      grossier mais appliqué <em>partout</em>, aligné sur la résolution de l'oreille. La branche « fin »
      fait du <dfn class="term" data-term="deep-filtering">deep filtering</dfn> : un petit <dfn class="term" data-term="fir">filtre FIR</dfn> d'<strong>ordre 5</strong>, dont les
      coefficients complexes sont prédits par <dfn class="term" data-term="fft-bin">bin</dfn>, raffine la structure fine des seules basses
      fréquences (&lt;&nbsp;5&nbsp;kHz) — là où vivent les <dfn class="term" data-term="harmonique">harmoniques</dfn> de la voix et où l'oreille exige
      de la précision. On dépense le calcul là où il compte.</p>
      <p>La sortie est <strong>duale et cohérente</strong> : le réseau estime <code>VOICE</code>, et
      <code>REST = MIX − VOICE</code> est obtenu par <strong>soustraction spectrale</strong> — chaque
      composante du mix finit donc dans l'une ou l'autre sortie, sans perte ni double comptage. C'est
      ce qui rend le <strong>boost dialogue</strong> possible sans artefact : le téléviseur re-mixe
      simplement <code>voice·g + rest</code> (slider 0–12&nbsp;dB). Ici le <dfn class="term" data-term="masking">masque ratio</dfn>
      <em><dfn class="term" data-term="irm">oracle</dfn></em> est réellement appliqué à un mix parole+musique (<dfn class="term" data-term="fft">FFT</dfn> 256) : les quatre
      mini-spectrogrammes MIX, VOICE, REST et TV&nbsp;out montrent de vrais spectres, et le re-mix
      réagit réellement au slider.</p>
      <p>Tout tient dans un <strong>budget embarqué</strong> : la somme affichée (≈&nbsp;2,4&nbsp;G
      <dfn class="term" data-term="parameters">MACs</dfn>/frame, dominée par l'encodeur, les GRU et les décodeurs) reste loin du budget
      <dfn class="term" data-term="npu">NPU</dfn> (~10&nbsp;G par frame de 10&nbsp;ms, soit ~1&nbsp;TOPS). Côté <dfn class="term" data-term="latency">latence</dfn>, le toggle décompose les
      ~42&nbsp;ms : 20&nbsp;ms de fenêtre d'analyse, 20&nbsp;ms de <dfn class="term" data-term="lookahead">look-ahead</dfn> du deep filtering, et
      ~2&nbsp;ms de calcul NPU — sous les 50&nbsp;ms tolérés pour la synchro labiale. C'est l'intérêt
      des GRU plutôt qu'une <dfn class="term" data-term="attention">attention</dfn> bidirectionnelle : aucune trame future n'est requise.</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

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

      /* ============================================================
         HELPERS DE DESSIN — factorisés, réutilisés par compact & desktop.
         Toutes les tailles passent par fs(...) pour rester lisibles
         sur mobile ; sur desktop fs() est l'identité.
         ============================================================ */

      /* Les quatre mini-spectrogrammes vivants (MIX, VOICE, REST, TV out).
         xs[i] : abscisses, sy : ordonnée du haut, sw/scrH : taille d'une vignette,
         labY : ordonnée de la légende. db utilisé pour le texte TV. */
      function drawScrollers(xs, sy, sw, scrH, labY, db, g, compact) {
        const labels = compact
          ? ['MIX', 'VOICE', 'REST', 'TV +' + db.toFixed(0) + ' dB']
          : ['MIX (entrée)', 'VOICE = mix × masque', 'REST = mix − voice',
             'TV out — voix +' + db.toFixed(1) + ' dB (×' + g.toFixed(1) + ')'];
        for (let i = 0; i < 4; i++) {
          SCRS[i].scr.draw(ctx, xs[i], sy, sw, scrH);
          ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
          ctx.strokeRect(xs[i] + 0.5, sy + 0.5, sw - 1, scrH - 1);
          U.text(ctx, labels[i], xs[i], labY, { size: compact ? fs(9.5) : 9.5, color: SCRS[i].c, bold: true });
        }
      }

      /* Arêtes + pulse traversant (1 hop = 1 frame de 10 ms). */
      function drawEdges(ph) {
        for (const e of EDGES) {
          const [x1, y1, x2, y2] = anch(ND[e[0]], ND[e[1]]);
          U.arrow(ctx, x1, y1, x2, y2,
            { color: e[2], alpha: 0.55, lw: 1.2, head: 5, dash: e[3] ? [4, 3] : null });
          U.glowDot(ctx, U.lerp(x1, x2, ph), U.lerp(y1, y2, ph), 2.2, e[2]);
        }
      }

      /* Nodes (halo continu = focus easé ∨ pulse traversant). */
      function drawNodes(fk, env, flash, compact, titleSize) {
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
            title: compact ? n.st : n.t, sub: compact ? null : n.s,
            color: n.c, active: glow > 0.45, size: titleSize,
          });
        }
      }

      /* Bandeau bas : rôle du bloc courant + Σ MACs + jauge budget NPU.
         Disposé dans le rectangle (bx, by, bw, bh). */
      function drawBanner(bx, by, bw, bh, fi, env, compact) {
        U.roundRect(ctx, bx, by, bw, bh, 6);
        ctx.fillStyle = palette.panel; ctx.fill();
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.stroke();

        const e = INFO[fi];
        const macs = e.g >= 0.01 ? ' · ~' + e.g.toFixed(2) + ' G MACs/frame'
          : e.g > 0 ? ' · < 0,01 G MACs/frame' : '';

        if (compact) {
          /* Mobile : on EMPILE sans recouvrement.
             [a] titre du bloc courant
             [b] description en plusieurs lignes (wrap mesuré)
             [c] PUIS, sur leur propre ligne, Σ MACs + jauge NPU.
             La ligne Σ est réservée d'abord (hauteur fixe en bas du
             bandeau) ; la description ne dispose donc que de l'espace
             situé AU-DESSUS. On mesure le wrap, et si le nombre de lignes
             dépasse la place, on rogne d'abord la police puis le nombre de
             lignes — jamais de chevauchement. */
          const padX = bx + fs(10);
          const descW = Math.max(10, bw - fs(20));

          /* [c] Bande réservée en bas pour Σ + jauge (hauteur fixe). */
          const sumBandH = fs(22);
          const gy = by + bh - sumBandH + fs(13);   // ligne de base de Σ / jauge

          /* Géométrie verticale disponible pour titre + description. */
          const titleY = by + fs(15);
          const descTop = titleY + fs(14);           // 1re ligne de description
          const descBottom = by + bh - sumBandH - fs(2);

          /* Auto-ajustement : on réduit la police jusqu'à ce que les lignes
             tiennent dans la zone, sinon on tronque le nombre de lignes. */
          let descFs = fs(11.5);
          let lineH = descFs * 1.18;
          let lines = wrapLines(e.d + macs, descW, descFs);
          const avail = Math.max(0, descBottom - descTop);
          let maxLines = Math.max(1, Math.floor(avail / lineH));
          if (lines.length > maxLines) {
            /* tente de réduire la police pour caser une ligne de plus */
            const minFs = fs(9);
            while (lines.length > maxLines && descFs > minFs) {
              descFs = Math.max(minFs, descFs - fs(0.75));
              lineH = descFs * 1.18;
              lines = wrapLines(e.d + macs, descW, descFs);
              maxLines = Math.max(1, Math.floor(avail / lineH));
            }
            if (lines.length > maxLines) lines = lines.slice(0, maxLines);
          }

          ctx.save();
          ctx.globalAlpha = Math.min(1, env * 1.5);
          U.text(ctx, ND[e.n].t, padX, titleY, { size: fs(13), bold: true, color: ND[e.n].c });
          for (let li = 0; li < lines.length; li++) {
            U.text(ctx, lines[li], padX, descTop + li * lineH,
              { size: descFs, color: palette.dim });
          }
          ctx.restore();
          ctx.globalAlpha = 1;

          /* [c] Σ MACs + jauge budget NPU, sur leur propre ligne en bas. */
          ctx.font = fs(11.5) + 'px ' + U.FONT;
          const sumTxt = 'Σ ' + TOTAL.toFixed(2) + ' G MACs/frame';
          U.text(ctx, sumTxt, padX, gy + fs(2),
            { size: fs(11.5), bold: true, color: palette.text, mono: true });
          const sumW = ctx.measureText(sumTxt).width;
          const gx = padX + Math.min(descW * 0.62, sumW + fs(10), descW - fs(40));
          const gw = bx + bw - fs(10) - gx;
          if (gw > 6) {
            ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
            U.roundRect(ctx, gx, gy - fs(5), gw, fs(8), 3); ctx.stroke();
            ctx.fillStyle = palette.green; ctx.globalAlpha = 0.85;
            ctx.fillRect(gx + 1, gy - fs(5) + 1, Math.max((TOTAL / NPU_BUDGET) * gw - 2, 1), fs(8) - 2);
            ctx.globalAlpha = 1;
            U.text(ctx, '/ NPU ~' + NPU_BUDGET + ' G', bx + bw - fs(10), gy + fs(2),
              { size: fs(9.5), color: palette.faint, align: 'right' });
          }
          return;
        }

        /* Desktop : fiche à gauche (clippée), Σ + jauge à droite. */
        const rw = 230;
        ctx.save();
        ctx.beginPath(); ctx.rect(bx + 6, by + 2, bw - rw - 12, bh - 4); ctx.clip();
        ctx.globalAlpha = Math.min(1, env * 1.5);
        U.text(ctx, ND[e.n].t, bx + 10, by + 14, { size: 12, bold: true, color: ND[e.n].c });
        U.text(ctx, e.d + macs, bx + 10, by + 28, { size: 10, color: palette.dim });
        ctx.restore();
        ctx.globalAlpha = 1;

        U.text(ctx, 'Σ ≈ ' + TOTAL.toFixed(2) + ' G MACs/frame',
          bx + bw - 10, by + 13, { size: 11, bold: true, color: palette.text, align: 'right', mono: true });
        const gw = 110, gx = bx + bw - 10 - gw, gy = by + bh - 12;
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
        U.roundRect(ctx, gx, gy, gw, 7, 3); ctx.stroke();
        ctx.fillStyle = palette.green; ctx.globalAlpha = 0.85;
        ctx.fillRect(gx + 1, gy + 1, Math.max((TOTAL / NPU_BUDGET) * gw - 2, 1), 5);
        ctx.globalAlpha = 1;
        U.text(ctx, 'budget NPU ~' + NPU_BUDGET + ' G', gx - 8, gy + 7,
          { size: 8.5, color: palette.faint, align: 'right' });
      }

      /* Découpe une chaîne en lignes tenant dans maxW px à la taille size. */
      function wrapLines(str, maxW, size) {
        ctx.font = size + 'px ' + U.FONT;
        const words = str.split(' ');
        const lines = [];
        let cur = '';
        for (const w of words) {
          const tryStr = cur ? cur + ' ' + w : w;
          if (ctx.measureText(tryStr).width > maxW && cur) { lines.push(cur); cur = w; }
          else cur = tryStr;
        }
        if (cur) lines.push(cur);
        return lines.length ? lines : [str];
      }

      /* ---------- Boucle de rendu ---------- */
      stage.onFrame((t, dt) => {
        stage.clear();
        const W = stage.W, H = stage.H, compact = stage.compact, M = 10;
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

        /* === Cycle descriptif : focus easé toutes les ~3 s (commun) === */
        const fi = Math.floor(posmod(t, FOCUS_T * INFO.length) / FOCUS_T);
        const fp = posmod(t, FOCUS_T) / FOCUS_T;
        const env = U.smoothstep(fp / 0.15) * (1 - U.smoothstep((fp - 0.82) / 0.18));
        const fk = INFO[fi].n;
        const ph = posmod(t / PULSE_T, 1);
        const flash = Math.max(ph > 0.85 ? (ph - 0.85) / 0.15 : 0,
                               ph < 0.12 ? (1 - ph / 0.12) * 0.8 : 0) * 0.55;

        if (compact) {
          /* ============================================================
             MOBILE — tout empilé verticalement, texte agrandi (fs).
             [1] titre   [2] 4 mini-spectres en grand   [3] graphe en
             2 colonnes lisibles   [4] (latences)   [5] fiche + budget NPU.
             ============================================================ */
          let y = fs(16);
          U.text(ctx, 'Pipeline enhancement temps réel', W / 2, y,
            { size: fs(13), bold: true, align: 'center' });
          y += fs(8);

          /* [2] Mini-spectres : 4 vignettes côte à côte, hautes et lisibles. */
          const sgap = fs(6);
          const sw = Math.max(24, (W - 2 * M - 3 * sgap) / 4);
          const scrH = U.clamp(H * 0.13, 34, 56);
          const labY = y + fs(11);
          const sy = labY + fs(5);
          const xs = [M, M + sw + sgap, M + 2 * (sw + sgap), M + 3 * (sw + sgap)];
          drawScrollers(xs, sy, sw, scrH, labY, db, g, true);
          y = sy + scrH + fs(8);

          /* [5] réservé en bas : fiche descriptive + budget NPU.
             Hauteur suffisante pour empiler titre + description (wrap) PUIS,
             sur leur propre ligne, Σ MACs + jauge NPU sans recouvrement. */
          const bannerH = fs(86);
          const latH = latOn ? fs(26) : 0;
          const bottomY = H - M - bannerH;
          const latY = bottomY - latH - fs(4);

          /* [3] Zone graphe (entre les spectres et le bas). */
          const gTop = y + fs(2);
          const gBot = (latOn ? latY : bottomY) - fs(6);
          const gH = Math.max(120, gBot - gTop);

          /* Deux colonnes : à gauche la chaîne principale (in→stft→erb→enc→
             gru→dgains→apply→istft→voice), à droite la branche fine
             (cplx→ddf→filt) + rest. 9 lignes à gauche, on place les boîtes. */
          const colGap = fs(10);
          const leftW = Math.max(60, (W - 2 * M - colGap) * 0.56);
          const rightW = Math.max(50, W - 2 * M - colGap - leftW);
          const leftX = M + leftW / 2;
          const rightX = M + leftW + colGap + rightW / 2;

          /* Colonne gauche : 8 étages empilés (in,stft,erb,enc,gru,dgains,apply,istft). */
          const leftKeys = ['in', 'stft', 'erb', 'enc', 'gru', 'dgains', 'apply', 'istft'];
          const nL = leftKeys.length;
          const nhL = U.clamp((gH - (nL - 1) * fs(6)) / nL, 20, 40);
          const stepL = nhL + Math.max(fs(4), (gH - nL * nhL) / Math.max(1, nL - 1));
          const nwL = Math.min(leftW, fs(150));
          for (let i = 0; i < nL; i++) {
            set(leftKeys[i], leftX, gTop + nhL / 2 + i * stepL, nwL, nhL);
          }

          /* Colonne droite : branche fine alignée en regard de stft→apply,
             puis VOICE / REST en bas en regard de istft. */
          const cplxY = ND.stft.y;            // part du même point que la branche gauche
          const filtY = ND.apply.y;           // rejoint l'application
          const ddfY = (cplxY + filtY) / 2;
          const nwR = Math.min(rightW, fs(140));
          set('cplx', rightX, cplxY, nwR, nhL);
          set('ddf', rightX, ddfY, nwR, nhL);
          set('filt', rightX, filtY, nwR, nhL);
          /* VOICE et REST en regard de iSTFT, empilés serrés. */
          const voH = Math.max(18, nhL * 0.92);
          set('voice', rightX, ND.istft.y - voH * 0.6, nwR, voH);
          set('rest', rightX, ND.istft.y + voH * 0.6, nwR, voH);

          drawEdges(ph);
          drawNodes(fk, env, flash, true, fs(10));

          /* Cadence : petite étiquette sous le node d'entrée. */
          U.text(ctx, '1 frame / 10 ms', ND.in.x + ND.in.w / 2 + fs(8), ND.in.y,
            { size: fs(9), color: palette.faint, baseline: 'middle', mono: true });

          /* [4] Latences : jauge empilée pleine largeur + valeurs. */
          if (latOn) {
            U.text(ctx, 'Latence algorithmique', M, latY + fs(2),
              { size: fs(10.5), color: palette.dim });
            const ly = latY + fs(8), lh = fs(10);
            const bx = M, bw = W - 2 * M;
            ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
            U.roundRect(ctx, bx, ly, bw, lh, 3); ctx.stroke();
            let acc = 0;
            for (const l of LAT) {
              const wseg = (l.ms / LAT_BUDGET) * bw;
              ctx.fillStyle = l.c; ctx.globalAlpha = 0.85;
              ctx.fillRect(bx + acc + 1, ly + 1, Math.max(wseg - 2, 1), lh - 2);
              /* libellé centré dans son segment si la place le permet */
              if (wseg > fs(40)) {
                ctx.globalAlpha = 1;
                U.text(ctx, l.label + ' ' + l.ms, bx + acc + wseg / 2, ly + lh / 2 + fs(0.5),
                  { size: fs(8.5), color: '#0c0c0e', align: 'center', baseline: 'middle', bold: true });
              }
              acc += wseg;
            }
            ctx.globalAlpha = 1;
            U.text(ctx, '≈ ' + LAT_TOTAL + ' ms / budget ' + LAT_BUDGET + ' ms',
              W - M, ly + lh + fs(11),
              { size: fs(10), color: LAT_TOTAL <= LAT_BUDGET ? palette.green : palette.red,
                align: 'right', mono: true });
          }

          /* [5] Bandeau bas : fiche descriptive + Σ + budget NPU, en grand. */
          drawBanner(M, bottomY, W - 2 * M, bannerH, fi, env, true);
          return;
        }

        /* ============================================================
           DESKTOP / TABLETTE (≥ 560) — disposition inchangée.
           ============================================================ */
        const scrH = U.clamp(H * 0.13, 26, 42), labY = M + 8, scrY = M + 12;
        const sw = U.clamp((W - 2 * M) / 4 - 12, 70, 170);
        const tvX = W - M - sw, reX = tvX - sw - 8, voX = reX - sw - 8;
        const xs = [M, voX, reX, tvX]; // MIX au-dessus du node Audio in (à gauche)

        const bannerH = 34, latH = latOn ? 18 : 0;
        const dy0 = scrY + scrH + 14, dy1 = H - M - bannerH - latH - 6;
        const dh = Math.max(40, dy1 - dy0);

        { /* 1 rangée + 2 branches parallèles */
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
        }

        drawScrollers(xs, scrY, sw, scrH, labY, db, g, false);
        drawEdges(ph);
        drawNodes(fk, env, flash, false, 11);

        /* === Annotations : branches + cadence === */
        U.text(ctx, 'branche « gains » — grossier, sur tout le spectre',
          (ND.erb.x + ND.dgains.x) / 2, ND.erb.y - ND.erb.h / 2 - 8,
          { size: 10, color: palette.blue, align: 'center' });
        U.text(ctx, 'branche « fin » — précis, < 5 kHz seulement',
          (ND.cplx.x + ND.filt.x) / 2, ND.cplx.y + ND.cplx.h / 2 + 14,
          { size: 10, color: palette.pink, align: 'center' });
        U.text(ctx, '1 frame / 10 ms', (ND.in.x + ND.stft.x) / 2,
          ND.in.y + ND.in.h / 2 + (latOn ? 24 : 14),
          { size: 9, color: palette.faint, align: 'center', mono: true });

        /* === Latences : contributions sous les étages + jauge empilée === */
        if (latOn) {
          for (const l of LAT) {
            const n = ND[l.k];
            U.text(ctx, l.label + ' ' + l.ms + ' ms', n.x, n.y + n.h / 2 + 12,
              { size: 9, color: l.c, align: 'center' });
          }
          const ly = dy1 + 6, lh = 10;
          const bw = Math.min(W * 0.4, 240), bx = M + 150;
          U.text(ctx, 'Latence algorithmique', M, ly + lh - 1, { size: 10, color: palette.dim });
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
        drawBanner(M, H - M - bannerH, W - 2 * M, bannerH, fi, env, false);
      });
    },
  });
})();
