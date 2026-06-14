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
      <p>La latence de bout en bout d'un système audio temps réel est une <strong>somme de contributions</strong>.
      La <strong>latence algorithmique</strong> — fenêtre d'analyse, <strong>look-ahead</strong> et attente de
      remplissage du tampon (en moyenne <code>½ hop</code>) — est <strong>incompressible</strong> : elle découle de la
      définition même du traitement, et aucun processeur, si rapide soit-il, ne peut la réduire. Le
      <strong>temps de calcul</strong>, lui, est la seule part réductible : un NPU plus rapide le fait fondre,
      sans toucher au reste.</p>
      <p>Pourquoi payer du look-ahead ? Parce que le <strong>futur proche désambiguïse le présent</strong> : une
      plosive (/p/, /t/, /k/) et un claquement parasite se ressemblent dans leurs premières millisecondes ;
      ce qui suit (voyelle voisée ou silence) permet de trancher. Chaque trame de contexte futur améliore la
      qualité du débruitage ou de la séparation… au prix d'une trame entière de délai supplémentaire
      (<code>look-ahead × hop</code>).</p>
      <p>Les budgets dépendent de l'usage : en <strong>broadcast TV</strong>, la désynchronisation labiale devient
      perceptible vers <strong>100 ms</strong> (on vise &lt; 50 ms pour garder de la marge dans la chaîne) ; en
      <strong>visioconférence</strong>, on tolère ~150 ms aller-retour ; une <strong>aide auditive</strong> exige
      moins de <strong>10 ms</strong>, car le son traité se superpose au son direct qui contourne l'appareil —
      au-delà, l'interférence des deux chemins crée un <strong>effet de peigne</strong>, puis un écho.</p>`,

    init(stage) {
      const ctx = stage.ctx;

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

        /* — layout, recalculé depuis W/H à chaque frame — */
        const W = stage.W, H = stage.H;
        const narrow = W < 560;
        const pad = 10, topY = pad + 2;
        const gaugeW = narrow ? 52 : 92;
        const mainX = pad, mainW = Math.max(60, W - pad * 2 - gaugeW - 12);
        const right = mainX + mainW;
        const innerH = H - topY - pad;
        const tableH = (narrow || H < 380) ? 24 : 46;
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

        stage.clear();

        /* — signaux : entrée live, sortie = même signal décalé du total réel — */
        const tSig = t * (EV / CYCLE); // temps signal (ralenti ×SLOWMO)
        const dly = total / 1000;
        const nCols = Math.min(BUF, Math.max(16, Math.floor(mainW)));
        if (nCols !== viewN) { viewN = nCols; inView = inBuf.subarray(0, nCols); outView = outBuf.subarray(0, nCols); }
        for (let i = 0; i < nCols; i++) {
          const ts = tSig - EV * (1 - i / (nCols - 1));
          inBuf[i] = inputSample(ts);
          outBuf[i] = inputSample(ts - dly);
        }
        U.frame(ctx, mainX, yIn, mainW, inH, 'Entrée (micro)');
        U.frame(ctx, mainX, yOut, mainW, outH, 'Sortie (haut-parleur)');
        if (!narrow) U.text(ctx, `ralenti ×${SLOWMO}`, right - 4, yIn - 6, { size: 9, color: palette.faint, align: 'right' });
        U.wave(ctx, inView, mainX + 1, yIn, mainW - 2, inH, { color: C.sig, lw: 1.2, scale: 0.9 });
        U.wave(ctx, outView, mainX + 1, yOut, mainW - 2, outH, { color: C.sig, lw: 1.2, scale: 0.9, alpha: 0.85 });

        /* — axe temporel central — */
        ctx.save();
        ctx.strokeStyle = 'rgba(155,152,143,0.55)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mainX, axisY + 0.5); ctx.lineTo(right, axisY + 0.5); ctx.stroke();
        ctx.strokeStyle = 'rgba(244,242,236,0.16)'; // bord droit = maintenant
        ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(right - 0.5, yIn); ctx.lineTo(right - 0.5, yOut + outH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        /* — événements (courant + précédent), tout est clippé à la scène — */
        const drawEvent = (xE, ageMs, isCur) => {
          const a = isCur ? 1 : 0.8;
          /* graduations en ms, ancrées sur l'événement (défilent avec lui) */
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
            if (major) U.text(ctx, String(m), gx, axisY + 16, { size: 9, color: palette.faint, align: 'center', mono: true });
          }
          ctx.globalAlpha = 1;
          /* blocs de contribution, empilés séquentiellement à partir de t=0 ;
             le bord droit du panneau = maintenant → remplissage naturel dans le temps */
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
            if (!narrow && bw > 54 && blockH >= 24) {
              U.text(ctx, s.name, bx + bw / 2, axisY - blockH / 2 - 3, { size: 9, color: s.c, align: 'center' });
              U.text(ctx, num(s.ms) + ' ms', bx + bw / 2, axisY - blockH / 2 + 9, { size: 9.5, color: palette.text, align: 'center', mono: true });
            }
            ctx.globalAlpha = 1;
          }
          /* marque d'entrée (corail) sur la waveform du haut, à t=0 */
          ctx.globalAlpha = a;
          ctx.strokeStyle = C.sig; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(xE, yIn); ctx.lineTo(xE, axisY + 5); ctx.stroke();
          U.glowDot(ctx, xE, axisY, 2.6, C.sig);
          U.chip(ctx, '💥 plosive', Math.min(xE + 5, right - 76), yIn + 11, { color: C.sig, size: 9 });
          if (!narrow) U.text(ctx, 'entrée', xE - 4, yAx + 8, { size: 9, color: C.sig, align: 'right' });
          ctx.globalAlpha = 1;

          if (ageMs >= total) {
            /* l'événement ressort : marque de sortie + arc + étiquette du total */
            const fade = U.ease(U.clamp((ageMs - total) / 25, 0, 1)) * a;
            const xX = xE + total * pxPerMs;
            ctx.globalAlpha = fade;
            ctx.strokeStyle = C.sig; ctx.lineWidth = 1.6;
            ctx.beginPath(); ctx.moveTo(xX, axisY - 5); ctx.lineTo(xX, yOut + outH); ctx.stroke();
            U.glowDot(ctx, xX, axisY, 2.6, C.sig);
            if (!narrow) U.text(ctx, 'sortie', xX + 4, yAx + 8, { size: 9, color: C.sig });
            const depth = U.clamp(axH - blockH - 26, 12, 42);
            ctx.lineWidth = 1.1; ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.moveTo(xE, axisY + 6);
            ctx.quadraticCurveTo((xE + xX) / 2, axisY + 20 + depth, xX, axisY + 6);
            ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = C.sig;
            ctx.beginPath(); ctx.moveTo(xX, axisY + 5); ctx.lineTo(xX - 4, axisY + 12); ctx.lineTo(xX + 3, axisY + 12); ctx.closePath(); ctx.fill();
            const lbl = 'latence totale = ' + fmtMs(total);
            const ls = narrow ? 9.5 : 11;
            ctx.font = `600 ${ls}px ${U.FONT}`;
            const lw = ctx.measureText(lbl).width;
            const mx = (xE + xX) / 2, my = axisY + 13 + depth / 2;
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
        ctx.beginPath(); ctx.rect(mainX, topY, mainW, H - topY - pad); ctx.clip();
        const kNow = Math.floor(tSig / EV);
        for (let kk = kNow; kk >= Math.max(0, kNow - 1); kk--) {
          const ageMs = (tSig - kk * EV) * 1000;
          const xE = right - ageMs * pxPerMs;
          if (xE + (total + 40) * pxPerMs < mainX) continue;
          drawEvent(xE, ageMs, kk === kNow);
        }
        ctx.restore();
        U.text(ctx, 'axe : ms écoulées depuis la plosive', mainX + 2, yAx + 8, { size: 9, color: palette.faint });
        if (!narrow) U.text(ctx, 'maintenant ▸', right - 4, axisY - blockH - 6, { size: 9, color: palette.faint, align: 'right' });

        /* — jauge de budget (droite) : aiguille = total réellement calculé — */
        const gX = W - pad - gaugeW;
        const gTop = yIn + 4, gBot = yOut + outH, gH = gBot - gTop;
        const barX = gX + 4, barW = narrow ? 10 : 14;
        const yOf = (v) => gBot - (U.clamp(v, 0, AXIS_MS) / AXIS_MS) * gH;
        U.text(ctx, 'Budget', gX + gaugeW / 2, yIn - 6, { size: 11, color: palette.dim, align: 'center', bold: true });
        const pulse = 0.5 + 0.5 * Math.sin(t * 6);
        for (const [v0, v1, c] of [[0, 50, C.ok], [50, 100, C.warn], [100, 150, C.over]]) {
          const alpha = (c === C.over && total > 100)
            ? Math.round((0.18 + 0.22 * pulse) * 255).toString(16).padStart(2, '0') : '26';
          ctx.fillStyle = c + alpha;
          ctx.fillRect(barX, yOf(v1), barW, yOf(v0) - yOf(v1));
          ctx.strokeStyle = c + '55'; ctx.lineWidth = 1;
          ctx.strokeRect(barX + 0.5, yOf(v1) + 0.5, barW - 1, yOf(v0) - yOf(v1) - 1);
        }
        for (const v of [0, 50, 100, 150]) {
          U.text(ctx, String(v), barX + barW + 5, yOf(v) + 3, { size: 9, color: palette.faint, mono: true });
        }
        if (!narrow) {
          U.text(ctx, 'budget TV', barX + barW + 5, yOf(50) + 14, { size: 8.5, color: C.ok });
          U.text(ctx, 'désynchro', barX + barW + 5, yOf(100) + 14, { size: 8.5, color: C.over });
          U.text(ctx, 'labiale', barX + barW + 5, yOf(100) + 23, { size: 8.5, color: C.over });
        }
        const ny = yOf(total);
        ctx.save();
        if (total > 100) { ctx.shadowColor = C.over; ctx.shadowBlur = 7 + 6 * pulse; }
        ctx.strokeStyle = zone; ctx.fillStyle = zone; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(barX - 2, ny); ctx.lineTo(barX + barW + 2, ny); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(barX + barW + 3, ny); ctx.lineTo(barX + barW + 9, ny - 4); ctx.lineTo(barX + barW + 9, ny + 4); ctx.closePath(); ctx.fill();
        ctx.restore();
        U.text(ctx, fmtMs(total), gX + gaugeW / 2, gBot + 13, { size: narrow ? 10 : 12, color: zone, bold: true, align: 'center', mono: true });
        if (total > 100 && !narrow) {
          ctx.globalAlpha = 0.6 + 0.4 * pulse;
          U.chip(ctx, '⚠ hors budget', gX + 2, gBot + 28, { color: C.over, size: 9 });
          ctx.globalAlpha = 1;
        }

        /* — tableau dynamique des contributions (vraies valeurs, sommées) — */
        if (tableH >= 40) {
          const colW = mainW / 5;
          for (let i = 0; i < 4; i++) {
            const cx = mainX + colW * (i + 0.5);
            U.text(ctx, segs[i].name, cx, yTab + 12, { size: 10, color: segs[i].c, align: 'center' });
            U.text(ctx, num(segs[i].ms), cx, yTab + 30, { size: 12, color: palette.text, align: 'center', mono: true });
            U.text(ctx, i < 3 ? '+' : '=', mainX + colW * (i + 1), yTab + 30, { size: 12, color: palette.faint, align: 'center' });
          }
          U.text(ctx, 'TOTAL', mainX + colW * 4.5, yTab + 12, { size: 10, color: palette.dim, align: 'center', bold: true });
          U.text(ctx, fmtMs(total), mainX + colW * 4.5, yTab + 30, { size: 13, color: zone, align: 'center', bold: true, mono: true });
        } else {
          /* mode compact : une ligne, mêmes valeurs vraies */
          const parts = [
            [num(d.win), C.win], [' + ', palette.faint], [num(laMs), C.la], [' + ', palette.faint],
            [num(fifoMs), C.fifo], [' + ', palette.faint], [num(d.calc), C.calc], [' = ', palette.faint],
            [fmtMs(total), zone],
          ];
          ctx.font = `600 11px ${U.MONO}`;
          let tw = 0;
          for (const p of parts) tw += ctx.measureText(p[0]).width;
          let px = mainX + Math.max(0, (mainW - tw) / 2);
          for (const p of parts) {
            U.text(ctx, p[0], px, yTab + 15, { size: 11, color: p[1], bold: true, mono: true });
            px += ctx.measureText(p[0]).width;
          }
        }
      });
    },
  });
})();
