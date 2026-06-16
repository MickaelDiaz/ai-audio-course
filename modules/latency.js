/* ============================================================
   Audio AI Atlas — module « Latence & temps réel »
   Une plosive entre dans le système (haut), les contributions à
   la latence s'empilent sur l'axe temporel (blocs colorés), puis
   l'événement ressort, décalé du total VRAI, sur la sortie (bas).
   Jauge de budget à droite, tableau dynamique en pied.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  /* Couleurs sémantiques des contributions (cf. spec) */
  const C = {
    win: palette.blue,    // fenêtre d'analyse
    la: palette.mix,      // look-ahead (violet)
    fifo: palette.dim,    // attente FIFO ≈ ½ hop (gris)
    calc: palette.rest,   // temps de calcul (ambre)
    sig: palette.voice,   // le signal vocal (corail)
    ok: palette.green, warn: palette.rest, over: palette.red,
  };

  const AXIS_MS = 150;          // axe gradué 0–150 ms
  const SPAN_MS = 185;          // fenêtre de signal visible (= espacement des événements)
  const CYCLE = 3.0;            // un événement toutes les ~3 s (temps mur)
  const EV = SPAN_MS / 1000;    // période des événements en temps signal
  const SLOWMO = Math.round(CYCLE / EV); // facteur de ralenti réellement appliqué (×16)

  const fmtMs = (m) => (m < 99.95 ? (Math.round(m * 10) / 10).toFixed(1) : String(Math.round(m))) + ' ms';
  const num = (m) => String(Math.round(m * 10) / 10);

  /* Signal d'entrée : pseudo-parole + plosive synthétique périodique (l'événement) */
  function inputSample(ts) {
    let v = U.gen.speech(ts) * 0.8;
    const tc = ((ts % EV) + EV) % EV;
    if (tc < 0.007) v += (U.hash(ts * 99991) - 0.5) * 2.6 * (1 - tc / 0.007);
    return v;
  }

  AtlasRegister({
    id: 'latency',
    title: 'Latence & temps réel',
    category: 'system',
    icon: '⏱',
    summary: "D'où vient la latence d'un système audio temps réel, et combien coûte chaque milliseconde.",
    explain: `
      <p>La <dfn class="term" data-term="latency">latence</dfn> de bout en bout d'un système audio
      <dfn class="term" data-term="streaming">temps réel</dfn> est une <strong>somme de contributions</strong>.
      La <dfn class="term" data-term="algorithmic-latency">latence algorithmique</dfn> — <dfn class="term" data-term="window">fenêtre d'analyse</dfn>,
      <dfn class="term" data-term="lookahead">look-ahead</dfn> et attente de
      remplissage du tampon (en moyenne <code>½ <dfn class="term" data-term="hop">hop</dfn></code>) — est <strong>incompressible</strong> : elle découle de la
      définition même du traitement, et aucun processeur, si rapide soit-il, ne peut la réduire. Le
      <strong>temps de calcul</strong>, lui, est la seule part réductible : un <dfn class="term" data-term="npu">NPU</dfn> plus rapide le fait fondre,
      sans toucher au reste.</p>
      <p>Pourquoi payer du look-ahead ? Parce que le <strong>futur proche désambiguïse le présent</strong> : une
      <dfn class="term" data-term="plosive">plosive</dfn> (/p/, /t/, /k/) et un claquement parasite se ressemblent dans leurs premières millisecondes ;
      ce qui suit (voyelle <dfn class="term" data-term="voiced">voisée</dfn> ou silence) permet de trancher. Chaque <dfn class="term" data-term="frame">trame</dfn> de contexte futur améliore la
      qualité du débruitage ou de la séparation… au prix d'une trame entière de délai supplémentaire
      (<code>look-ahead × hop</code>).</p>
      <p>Les budgets dépendent de l'usage : en <strong>broadcast TV</strong>, la <dfn class="term" data-term="lip-sync">désynchronisation labiale</dfn> devient
      perceptible vers <strong>100 ms</strong> (on vise &lt; 50 ms pour garder de la marge dans la chaîne) ; en
      <strong>visioconférence</strong>, on tolère ~150 ms aller-retour ; une <strong>aide auditive</strong> exige
      moins de <strong>10 ms</strong>, car le son traité se superpose au son direct qui contourne l'appareil —
      au-delà, l'interférence des deux chemins crée un <dfn class="term" data-term="comb-filter">effet de peigne</dfn>, puis un écho.</p>`,

    init(stage) {
      const ctx = stage.ctx;
      const fs = (n) => stage.fs(n);   // taille de police lisible (agrandie sur mobile)

      /* Buffers pré-alloués (jamais de new Float32Array dans onFrame) */
      const BUF = 2048;
      const inBuf = new Float32Array(BUF), outBuf = new Float32Array(BUF);
      let inView = inBuf, outView = outBuf, viewN = BUF;

      /* Préréglages : règlent réellement les sliders (animation easée) */
      const PRESETS = {
        tv:     { win: 20, la: 4, hop: 5,  calc: 2 }, // 20 + 20 + 2.5 + 2 = 44.5 ms
        visio:  { win: 10, la: 0, hop: 5,  calc: 2 }, // 14.5 ms
        aide:   { win: 4,  la: 0, hop: 2,  calc: 1 }, // 6 ms
        studio: { win: 40, la: 4, hop: 20, calc: 8 }, // 138 ms — hors budget
      };
      let anim = null;
      const custom = () => { sel.value = 'custom'; anim = null; };

      const sWin = stage.addSlider({ label: "Fenêtre d'analyse", min: 4, max: 40, step: 1, value: 20, format: (v) => v + ' ms', onChange: custom });
      const sLA = stage.addSlider({ label: 'Look-ahead', min: 0, max: 4, step: 1, value: 4, format: (v) => (v === 0 ? 'aucun' : v + ' × hop'), onChange: custom });
      const sHop = stage.addSlider({ label: 'Hop (cadence trames)', min: 2, max: 20, step: 0.5, value: 5, format: (v) => v + ' ms', onChange: custom });
      const sCalc = stage.addSlider({ label: 'Temps de calcul', min: 1, max: 8, step: 0.5, value: 2, format: (v) => v + ' ms', onChange: custom });
      const sel = stage.addSelect({
        label: 'Préréglage',
        options: [
          { value: 'custom', label: 'Personnalisé' },
          { value: 'tv', label: 'Broadcast TV' },
          { value: 'visio', label: 'Visio' },
          { value: 'aide', label: 'Aide auditive' },
          { value: 'studio', label: 'Studio offline' },
        ],
        value: 'tv',
        onChange: (v) => {
          if (!PRESETS[v]) return;
          anim = { p: 0, to: PRESETS[v], from: { win: sWin.value, la: sLA.value, hop: sHop.value, calc: sCalc.value } };
        },
      });

      /* Valeurs affichées, lissées (jamais de saut sec) */
      const d = { win: sWin.value, la: sLA.value, hop: sHop.value, calc: sCalc.value };

      /* ============================================================
         HELPERS DE DESSIN — partagés par les deux mises en page.
         Tous reçoivent un objet `L` (layout) recalculé à chaque frame
         et un objet `M` (modèle : segs, total, zone, valeurs lissées).
         Aucun ne suppose un ratio paysage ni un écran large.
         ============================================================ */

      /* Remplit les buffers et trace les deux formes d'onde (entrée / sortie). */
      function drawWaves(L, M, tSig) {
        const { mainX, mainW, yIn, inH, yOut, outH } = L;
        const dly = M.total / 1000;
        const nCols = Math.min(BUF, Math.max(16, Math.floor(mainW)));
        if (nCols !== viewN) { viewN = nCols; inView = inBuf.subarray(0, nCols); outView = outBuf.subarray(0, nCols); }
        for (let i = 0; i < nCols; i++) {
          const ts = tSig - EV * (1 - i / (nCols - 1));
          inBuf[i] = inputSample(ts);
          outBuf[i] = inputSample(ts - dly);
        }
        U.frame(ctx, mainX, yIn, mainW, inH, 'Entrée (micro)');
        U.frame(ctx, mainX, yOut, mainW, outH, 'Sortie (haut-parleur)');
        U.text(ctx, `ralenti ×${SLOWMO}`, mainX + mainW - 4, yIn - fs(6), { size: fs(9), color: palette.faint, align: 'right' });
        U.wave(ctx, inView, mainX + 1, yIn, mainW - 2, inH, { color: C.sig, lw: 1.4, scale: 0.9 });
        U.wave(ctx, outView, mainX + 1, yOut, mainW - 2, outH, { color: C.sig, lw: 1.4, scale: 0.9, alpha: 0.85 });
      }

      /* Trace l'axe temporel central + les événements (courant + précédent).
         `big` (mobile) active les libellés DANS les blocs et les repères. */
      function drawAxisAndEvents(L, M, t, tSig, big) {
        const { mainX, mainW, right, yIn, yOut, outH, yAx, axH, pxPerMs, blockH, axisY, topY, H, pad } = L;
        const { segs, total, zone } = M;

        /* — ligne d'axe + bord droit « maintenant » — */
        ctx.save();
        ctx.strokeStyle = 'rgba(155,152,143,0.55)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mainX, axisY + 0.5); ctx.lineTo(right, axisY + 0.5); ctx.stroke();
        ctx.strokeStyle = 'rgba(244,242,236,0.16)';
        ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(right - 0.5, yIn); ctx.lineTo(right - 0.5, yOut + outH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        const drawEvent = (xE, ageMs, isCur) => {
          const a = isCur ? 1 : 0.8;
          /* graduations en ms, ancrées sur l'événement */
          const step = pxPerMs > 2.2 ? 25 : 50;
          ctx.strokeStyle = palette.grid;
          ctx.lineWidth = 1;
          for (let m = 0; m <= AXIS_MS; m += 5) {
            const gx = xE + m * pxPerMs;
            if (gx < mainX - 2 || gx > right + 2) continue;
            const major = m % step === 0;
            if (!major && pxPerMs < 3.5) continue;
            ctx.globalAlpha = a * (major ? 0.9 : 0.4);
            ctx.beginPath(); ctx.moveTo(gx, axisY); ctx.lineTo(gx, axisY + (major ? 6 : 3)); ctx.stroke();
            if (major) U.text(ctx, String(m), gx, axisY + fs(15), { size: fs(9), color: palette.faint, align: 'center', mono: true });
          }
          ctx.globalAlpha = 1;
          /* blocs de contribution empilés séquentiellement depuis t=0 */
          let cum = 0;
          for (const s of segs) {
            const x0 = xE + cum * pxPerMs, x1 = xE + (cum + s.ms) * pxPerMs;
            cum += s.ms;
            if (s.ms <= 0.01 || x1 <= mainX || x0 >= right) continue;
            const bx = Math.max(x0, mainX), bw = Math.min(x1, right) - bx;
            ctx.globalAlpha = a;
            ctx.fillStyle = s.c + '30';
            ctx.fillRect(bx, axisY - blockH, bw, blockH);
            ctx.strokeStyle = s.c; ctx.globalAlpha = a * 0.8;
            ctx.strokeRect(bx + 0.5, axisY - blockH + 0.5, Math.max(bw - 1, 0.5), blockH - 1);
            /* libellé dans le bloc dès que la place le permet (≥ ~24 px de haut) */
            if (bw > fs(48) && blockH >= 24) {
              U.text(ctx, s.name, bx + bw / 2, axisY - blockH / 2 - fs(3), { size: fs(9), color: s.c, align: 'center' });
              U.text(ctx, num(s.ms) + ' ms', bx + bw / 2, axisY - blockH / 2 + fs(9), { size: fs(9.5), color: palette.text, align: 'center', mono: true });
            }
            ctx.globalAlpha = 1;
          }
          /* marque d'entrée (corail) sur la waveform du haut, à t=0 */
          ctx.globalAlpha = a;
          ctx.strokeStyle = C.sig; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(xE, yIn); ctx.lineTo(xE, axisY + 5); ctx.stroke();
          U.glowDot(ctx, xE, axisY, 2.6, C.sig);
          U.chip(ctx, '💥 plosive', Math.min(xE + 5, right - fs(76)), yIn + fs(11), { color: C.sig, size: fs(9) });
          U.text(ctx, 'entrée', xE - 4, yAx + fs(8), { size: fs(9), color: C.sig, align: 'right' });
          ctx.globalAlpha = 1;

          if (ageMs >= total) {
            /* l'événement ressort : marque de sortie + arc + étiquette du total */
            const fade = U.ease(U.clamp((ageMs - total) / 25, 0, 1)) * a;
            const xX = xE + total * pxPerMs;
            ctx.globalAlpha = fade;
            ctx.strokeStyle = C.sig; ctx.lineWidth = 1.6;
            ctx.beginPath(); ctx.moveTo(xX, axisY - 5); ctx.lineTo(xX, yOut + outH); ctx.stroke();
            U.glowDot(ctx, xX, axisY, 2.6, C.sig);
            U.text(ctx, 'sortie', xX + 4, yAx + fs(8), { size: fs(9), color: C.sig });
            const depth = U.clamp(axH - blockH - 26, 12, 42);
            ctx.lineWidth = 1.1; ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.moveTo(xE, axisY + 6);
            ctx.quadraticCurveTo((xE + xX) / 2, axisY + 20 + depth, xX, axisY + 6);
            ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = C.sig;
            ctx.beginPath(); ctx.moveTo(xX, axisY + 5); ctx.lineTo(xX - 4, axisY + 12); ctx.lineTo(xX + 3, axisY + 12); ctx.closePath(); ctx.fill();
            const lbl = 'latence totale = ' + fmtMs(total);
            const ls = fs(big ? 11 : 11);
            ctx.font = `600 ${ls}px ${U.FONT}`;
            const lw = ctx.measureText(lbl).width;
            let mx = (xE + xX) / 2;
            mx = U.clamp(mx, mainX + lw / 2 + 8, right - lw / 2 - 8); // jamais hors panneau
            const my = axisY + 13 + depth / 2;
            U.roundRect(ctx, mx - lw / 2 - 7, my - ls / 2 - 5, lw + 14, ls + 10, 7);
            ctx.fillStyle = palette.panel; ctx.fill();
            ctx.strokeStyle = C.sig + '66'; ctx.lineWidth = 1; ctx.stroke();
            U.text(ctx, lbl, mx, my, { size: ls, color: zone, bold: true, align: 'center', baseline: 'middle' });
            ctx.globalAlpha = 1;
          } else if (isCur) {
            /* front de traitement : l'événement est encore « dans » le système */
            let acc = 0, fc = C.calc;
            for (const s of segs) { if (ageMs < acc + s.ms) { fc = s.c; break; } acc += s.ms; }
            U.glowDot(ctx, right - 2, axisY - blockH / 2, 3, fc);
          }
        };

        ctx.save();
        ctx.beginPath(); ctx.rect(mainX, topY, mainW, Math.max(1, H - topY - pad)); ctx.clip();
        const kNow = Math.floor(tSig / EV);
        for (let kk = kNow; kk >= Math.max(0, kNow - 1); kk--) {
          const ageMs = (tSig - kk * EV) * 1000;
          const xE = right - ageMs * pxPerMs;
          if (xE + (total + 40) * pxPerMs < mainX) continue;
          drawEvent(xE, ageMs, kk === kNow);
        }
        ctx.restore();
        /* légende d'axe : sur mobile (big), sur sa propre ligne AU-DESSUS des
           marqueurs « entrée »/« sortie » (qui sont sur la ligne yAx+fs(8)),
           pour éviter tout recouvrement ; on raccourcit/réduit si besoin pour
           tenir dans la largeur du panneau. */
        if (big) {
          const axLbl = 'axe : ms écoulées depuis la plosive';
          let axFs = fs(9);
          ctx.font = `${axFs}px ${U.FONT}`;
          const avail = mainW - 4;
          const wMeasured = ctx.measureText(axLbl).width;
          if (wMeasured > avail) axFs = Math.max(fs(7), axFs * avail / wMeasured);
          U.text(ctx, axLbl, mainX + 2, yAx - fs(6), { size: axFs, color: palette.faint });
        } else {
          U.text(ctx, 'axe : ms écoulées depuis la plosive', mainX + 2, yAx + fs(8), { size: fs(9), color: palette.faint });
        }
        U.text(ctx, 'maintenant ▸', right - 4, axisY - blockH - fs(6), { size: fs(9), color: palette.faint, align: 'right' });
      }

      /* Jauge de budget verticale. `G` porte la géométrie de la jauge. */
      function drawGauge(G, M, t, big) {
        const { gX, gaugeW, gTop, gBot, labelTop, vertical } = G;
        const { total, zone } = M;
        const gH = Math.max(1, gBot - gTop);
        const barW = G.barW;
        const barX = gX + (vertical ? (gaugeW - barW) / 2 : 4);
        const yOf = (v) => gBot - (U.clamp(v, 0, AXIS_MS) / AXIS_MS) * gH;
        U.text(ctx, big ? 'Budget de latence' : 'Budget', gX + gaugeW / 2, labelTop, { size: fs(11), color: palette.dim, align: 'center', bold: true });
        const pulse = 0.5 + 0.5 * Math.sin(t * 6);
        for (const [v0, v1, c] of [[0, 50, C.ok], [50, 100, C.warn], [100, 150, C.over]]) {
          const alpha = (c === C.over && total > 100)
            ? Math.round((0.18 + 0.22 * pulse) * 255).toString(16).padStart(2, '0') : '26';
          ctx.fillStyle = c + alpha;
          ctx.fillRect(barX, yOf(v1), barW, Math.max(0, yOf(v0) - yOf(v1)));
          ctx.strokeStyle = c + '55'; ctx.lineWidth = 1;
          ctx.strokeRect(barX + 0.5, yOf(v1) + 0.5, Math.max(0.5, barW - 1), Math.max(0.5, yOf(v0) - yOf(v1) - 1));
        }
        /* graduations chiffrées. En mode vertical (mobile), on les place À GAUCHE
           de la barre (alignées à droite) pour libérer tout l'espace à DROITE aux
           libellés de segments et éviter qu'ils se chevauchent. On vérifie par
           mesure que les chiffres tiennent dans la marge gauche ; sinon on réduit. */
        if (vertical) {
          let gradFs = fs(9);
          ctx.font = `${gradFs}px ${U.MONO}`;
          const leftRoom = Math.max(0, barX - gX) - 4;
          let maxNumW = 0;
          for (const v of [0, 50, 100, 150]) maxNumW = Math.max(maxNumW, ctx.measureText(String(v)).width);
          if (maxNumW > leftRoom && maxNumW > 0) gradFs = Math.max(fs(7), gradFs * leftRoom / maxNumW);
          for (const v of [0, 50, 100, 150]) {
            U.text(ctx, String(v), barX - 5, yOf(v) + 3, { size: gradFs, color: palette.faint, mono: true, align: 'right' });
          }
        } else {
          for (const v of [0, 50, 100, 150]) {
            U.text(ctx, String(v), barX + barW + 5, yOf(v) + 3, { size: fs(9), color: palette.faint, mono: true });
          }
        }
        /* légendes des zones — toujours visibles si la place existe (mobile = oui) */
        if (big || gaugeW >= 80) {
          U.text(ctx, '✓ budget TV', barX + barW + 5, yOf(25), { size: fs(8.5), color: C.ok });
          U.text(ctx, 'tolérable', barX + barW + 5, yOf(75), { size: fs(8.5), color: C.warn });
          U.text(ctx, 'désynchro', barX + barW + 5, yOf(125) - fs(5), { size: fs(8.5), color: C.over });
          U.text(ctx, 'labiale', barX + barW + 5, yOf(125) + fs(6), { size: fs(8.5), color: C.over });
        }
        /* aiguille = total réellement calculé */
        const ny = yOf(total);
        ctx.save();
        if (total > 100) { ctx.shadowColor = C.over; ctx.shadowBlur = 7 + 6 * pulse; }
        ctx.strokeStyle = zone; ctx.fillStyle = zone; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(barX - 2, ny); ctx.lineTo(barX + barW + 2, ny); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(barX + barW + 3, ny); ctx.lineTo(barX + barW + 9, ny - 4); ctx.lineTo(barX + barW + 9, ny + 4); ctx.closePath(); ctx.fill();
        ctx.restore();
        U.text(ctx, fmtMs(total), gX + gaugeW / 2, gBot + fs(14), { size: fs(13), color: zone, bold: true, align: 'center', mono: true });
        if (total > 100) {
          ctx.globalAlpha = 0.6 + 0.4 * pulse;
          U.chip(ctx, '⚠ hors budget', gX + gaugeW / 2 - fs(42), gBot + fs(30), { color: C.over, size: fs(9) });
          ctx.globalAlpha = 1;
        }
      }

      /* Tableau des contributions (vraies valeurs sommées). `T` porte la géométrie. */
      function drawTable(T, M) {
        const { x, w, yTab, big } = T;
        const { segs, total, zone, laMs, fifoMs } = M;
        if (big) {
          /* mobile : grande grille à 5 colonnes, texte lisible */
          const colW = w / 5;
          for (let i = 0; i < 4; i++) {
            const cx = x + colW * (i + 0.5);
            U.text(ctx, segs[i].name, cx, yTab + fs(13), { size: fs(10.5), color: segs[i].c, align: 'center' });
            U.text(ctx, num(segs[i].ms), cx, yTab + fs(32), { size: fs(15), color: palette.text, align: 'center', mono: true });
            U.text(ctx, i < 3 ? '+' : '=', x + colW * (i + 1), yTab + fs(32), { size: fs(14), color: palette.faint, align: 'center' });
          }
          U.text(ctx, 'TOTAL', x + colW * 4.5, yTab + fs(13), { size: fs(10.5), color: palette.dim, align: 'center', bold: true });
          U.text(ctx, fmtMs(total), x + colW * 4.5, yTab + fs(32), { size: fs(15), color: zone, align: 'center', bold: true, mono: true });
          return;
        }
        if (T.full) {
          const colW = w / 5;
          for (let i = 0; i < 4; i++) {
            const cx = x + colW * (i + 0.5);
            U.text(ctx, segs[i].name, cx, yTab + 12, { size: 10, color: segs[i].c, align: 'center' });
            U.text(ctx, num(segs[i].ms), cx, yTab + 30, { size: 12, color: palette.text, align: 'center', mono: true });
            U.text(ctx, i < 3 ? '+' : '=', x + colW * (i + 1), yTab + 30, { size: 12, color: palette.faint, align: 'center' });
          }
          U.text(ctx, 'TOTAL', x + colW * 4.5, yTab + 12, { size: 10, color: palette.dim, align: 'center', bold: true });
          U.text(ctx, fmtMs(total), x + colW * 4.5, yTab + 30, { size: 13, color: zone, align: 'center', bold: true, mono: true });
        } else {
          /* desktop bas : une ligne, mêmes valeurs vraies */
          const parts = [
            [num(d.win), C.win], [' + ', palette.faint], [num(laMs), C.la], [' + ', palette.faint],
            [num(fifoMs), C.fifo], [' + ', palette.faint], [num(d.calc), C.calc], [' = ', palette.faint],
            [fmtMs(total), zone],
          ];
          ctx.font = `600 11px ${U.MONO}`;
          let tw = 0;
          for (const p of parts) tw += ctx.measureText(p[0]).width;
          let px = x + Math.max(0, (w - tw) / 2);
          for (const p of parts) {
            U.text(ctx, p[0], px, yTab + 15, { size: 11, color: p[1], bold: true, mono: true });
            px += ctx.measureText(p[0]).width;
          }
        }
      }

      stage.onFrame((t, dt) => {
        const dte = dt || 1 / 60; // en pause, les contrôles restent réactifs

        /* — préréglage easé : pilote réellement les sliders — */
        if (anim) {
          anim.p = Math.min(1, anim.p + dte / 0.9);
          const e = U.ease(anim.p);
          sWin.value = U.lerp(anim.from.win, anim.to.win, e);
          sLA.value = U.lerp(anim.from.la, anim.to.la, e);
          sHop.value = U.lerp(anim.from.hop, anim.to.hop, e);
          sCalc.value = U.lerp(anim.from.calc, anim.to.calc, e);
          if (anim.p >= 1) anim = null;
        }

        /* — lissage + somme VRAIE des contributions — */
        const k = 1 - Math.exp(-dte * 8);
        d.win += (sWin.value - d.win) * k;
        d.la += (sLA.value - d.la) * k;
        d.hop += (sHop.value - d.hop) * k;
        d.calc += (sCalc.value - d.calc) * k;
        const laMs = d.la * d.hop, fifoMs = d.hop / 2;
        const total = d.win + laMs + fifoMs + d.calc;
        const segs = [
          { name: 'fenêtre', ms: d.win, c: C.win },
          { name: 'look-ahead', ms: laMs, c: C.la },
          { name: '½ hop (FIFO)', ms: fifoMs, c: C.fifo },
          { name: 'calcul', ms: d.calc, c: C.calc },
        ];
        const zone = total <= 50 ? C.ok : total <= 100 ? C.warn : C.over;
        const M = { segs, total, zone, laMs, fifoMs };

        const W = stage.W, H = stage.H;
        const tSig = t * (EV / CYCLE); // temps signal (ralenti ×SLOWMO)
        stage.clear();

        if (stage.compact) {
          /* ===== MOBILE : tout empilé verticalement, texte agrandi ===== */
          const pad = 12;
          const mainX = pad, mainW = Math.max(60, W - pad * 2);
          const right = mainX + mainW;
          const pxPerMs = mainW / SPAN_MS;

          U.text(ctx, 'Latence — la somme des délais', W / 2, fs(16), { size: fs(13), bold: true, align: 'center' });

          /* — budget vertical explicite : on répartit la hauteur disponible
               entre les panneaux empilés. Aucun bloc ne dépasse la scène ;
               si la place manque, chaque clamp garantit une taille positive. — */
          const topY = fs(24);
          const gapS = fs(16), gapL = fs(22);
          const labelH = fs(14);               // place du label « Entrée / Sortie »
          const tableH = U.clamp(fs(40), 36, 64);
          const gaugeLabelH = fs(14), gaugeValH = fs(22);

          // espace total à répartir entre : 2 waves + axe + jauge (+ leurs labels/gaps)
          const fixed = topY + labelH + gapS + labelH + gapL          // titre→in, in label, gap, out label…
            + gaugeLabelH + gaugeValH + tableH + gapL * 2 + gapS * 2;
          const avail = Math.max(120, H - fixed);
          const waveH = U.clamp(avail * 0.16, 40, 90);
          const gaugeH = U.clamp(avail * 0.22, 70, 150);
          const axH = Math.max(96, avail - waveH * 2 - gaugeH);

          const yIn = topY + labelH;
          const yAx = yIn + waveH + gapL;
          const blockH = U.clamp(axH * 0.30, 28, 60);
          const axisY = yAx + blockH + fs(8);
          const yOut = yAx + axH + gapS + labelH;
          const gTop = yOut + waveH + gapL + gaugeLabelH;
          const gBot = gTop + gaugeH;
          const yTab = Math.min(gBot + gaugeValH + gapS, H - tableH); // jamais sous le bord

          const L = { mainX, mainW, right, yIn, inH: waveH, yOut, outH: waveH, yAx, axH, pxPerMs, blockH, axisY, topY, H, pad };

          drawWaves(L, M, tSig);
          drawAxisAndEvents(L, M, t, tSig, true);

          /* jauge horizontale-au-centre : barre verticale centrée, légendes à droite */
          const gaugeW = Math.min(mainW, 200);
          const gX = mainX + (mainW - gaugeW) / 2;
          const G = { gX, gaugeW, gTop, gBot, labelTop: gTop - fs(10), barW: fs(16), vertical: true };
          drawGauge(G, M, t, true);

          drawTable({ x: mainX, w: mainW, yTab, big: true }, M);
        } else {
          /* ===== DESKTOP / TABLETTE : disposition d'origine, inchangée ===== */
          const narrow = W < 560; // (ne se déclenche plus : compact prend le relais < 560)
          const pad = 10, topY = pad + 2;
          const gaugeW = narrow ? 52 : 92;
          const mainX = pad, mainW = Math.max(60, W - pad * 2 - gaugeW - 12);
          const right = mainX + mainW;
          const innerH = H - topY - pad;
          const tableH = H < 380 ? 24 : 46;
          const rest = Math.max(90, innerH - 56 - tableH);
          const axH = U.clamp(rest * 0.36, 72, 150);
          const inH = (rest - axH) / 2, outH = inH;
          const yIn = topY + 16;
          const yAx = yIn + inH + 10;
          const yOut = yAx + axH + 18;
          const yTab = yOut + outH + 12;
          const pxPerMs = mainW / SPAN_MS;
          const blockH = U.clamp(axH * 0.42, 16, 36);
          const axisY = yAx + blockH + 6;

          const L = { mainX, mainW, right, yIn, inH, yOut, outH, yAx, axH, pxPerMs, blockH, axisY, topY, H, pad };
          drawWaves(L, M, tSig);
          drawAxisAndEvents(L, M, t, tSig, false);

          const gX = W - pad - gaugeW;
          const gTop = yIn + 4, gBot = yOut + outH;
          const G = { gX, gaugeW, gTop, gBot, labelTop: yIn - 6, barW: narrow ? 10 : 14, vertical: false };
          drawGauge(G, M, t, false);

          drawTable({ x: mainX, w: mainW, yTab, big: false, full: tableH >= 40 }, M);
        }
      });
    },
  });
})();
