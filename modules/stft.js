/* ============================================================
   Audio AI Atlas — module « STFT & spectrogramme »
   Fenêtre glissante + FFT par trame : compromis temps-fréquence,
   recouvrement (overlap) et latence algorithmique.
   Tous les nombres affichés sont réellement calculés.
   ============================================================ */
(function () {
  'use strict';
  const { U, palette } = Atlas;

  const SR = 16000;            // fréquence d'échantillonnage (Hz)
  const VIS = 24000;           // 1,5 s visibles dans le bandeau du haut
  const LOG41 = Math.log(41);  // mapping v = log1p(mag·40)/log(41) → 0..1
  const SIZES = [128, 256, 512, 1024];

  AtlasRegister({
    id: 'stft',
    title: 'STFT & spectrogramme — découper le temps pour voir les fréquences',
    category: 'signal',
    icon: '▦',
    summary: 'Fenêtre glissante + FFT par trame : le compromis temps-fréquence rendu visible.',
    explain: `
      <p>La <strong>transformée de Fourier à court terme</strong> (STFT, <em>Short-Time Fourier
      Transform</em>) répond à une limite de la FFT : un spectre global dit <em>quelles</em> fréquences
      existent, mais pas <em>quand</em>. La parole étant non stationnaire, on découpe le signal en trames
      de <code>N</code> échantillons prélevées tous les <code>hop</code> échantillons, on multiplie chaque
      trame par une <strong>fenêtre</strong> (Hann) pour adoucir ses bords, puis on calcule sa FFT.
      Chaque trame devient une colonne du <strong>spectrogramme</strong> : temps en abscisse, fréquence
      en ordonnée, énergie en couleur.</p>
      <p>Le choix de <code>N</code> impose le <strong>compromis temps-fréquence</strong> (incertitude de
      Heisenberg-Gabor) : la résolution temporelle vaut <code>Δt = N/sr</code> et la résolution
      fréquentielle <code>Δf = sr/N</code>, donc <strong>Δt × Δf = 1</strong>, quoi qu'on fasse.
      Une grande fenêtre (1024 éch. = 64 ms à 16 kHz) sépare finement les harmoniques mais étale les
      <strong>plosives</strong> en stries verticales floues ; une petite fenêtre (128 éch. = 8 ms) capte
      ces transitoires avec netteté, au prix de bandes fréquentielles grossières (125 Hz par case).</p>
      <p>Le recouvrement (<em>overlap</em>) n'est pas un luxe : avec une fenêtre de Hann et un hop de
      50 % ou 25 % de N, la somme des fenêtres successives est constante (condition <strong>COLA</strong>,
      <em>Constant OverLap-Add</em>). La méthode <strong>overlap-add</strong> garantit alors une
      <strong>reconstruction parfaite</strong> du signal par STFT inverse — c'est le socle de presque tous
      les traitements fréquentiels : débruitage, séparation de sources, vocodeur de phase. Une fenêtre
      rectangulaire, elle, crée des discontinuités aux bords : les <strong>fuites spectrales</strong>
      (lobes latéraux) brouillent le spectre.</p>
      <p>Enfin, la fenêtre fixe la <strong>latence algorithmique</strong> : impossible de calculer une
      trame avant d'avoir reçu ses N échantillons, soit <code>N/sr</code> secondes incompressibles.
      C'est pourquoi les systèmes temps réel (téléphonie, débruitage de visioconférence, aides auditives)
      travaillent avec des fenêtres de <strong>10 à 20 ms</strong> : assez longues pour résoudre les
      harmoniques de la voix, assez courtes pour rester imperceptibles dans une conversation.</p>`,

    init(stage) {
      /* ---------- buffers pré-alloués (aucune grosse allocation dans onFrame) ---------- */
      const ring = new Float32Array(VIS);                  // buffer circulaire 1,5 s
      for (let i = 0; i < VIS; i++) ring[i] = U.gen.speech(i / SR);
      let absIdx = VIS;        // prochain échantillon absolu à générer
      let nextHop = VIS;       // fin de la prochaine trame STFT (compteur de hop)

      const seg = {}, vmap = {};
      for (const n of SIZES) { seg[n] = new Float32Array(n); vmap[n] = new Float32Array(n / 2); }
      const zoomBuf = new Float32Array(4096);              // N max + 4 hops max
      const winShape = U.hann(48);                         // forme de Hann pour le dessin

      let scroller = null;
      function buildScroller() {
        const w = U.clamp(Math.round((stage.W || 800) - 24), 64, 1010);
        const h = U.clamp(Math.round((stage.H || 450) * 0.34), 32, 280);
        scroller = new U.Scroller(w, h, U.magmaRGB);
      }
      buildScroller();
      stage.onResize(buildScroller);

      /* ---------- contrôles ---------- */
      const selN = stage.addSelect({
        label: 'Taille de fenêtre N',
        options: SIZES.map((n) => ({ value: String(n), label: `${n} éch. (${U.fmt.ms(n / SR)})` })),
        value: '512',
      });
      const selHop = stage.addSelect({
        label: 'Hop (avancement)',
        options: [
          { value: '0.25', label: '25 % de N' },
          { value: '0.5', label: '50 % de N' },
          { value: '0.75', label: '75 % de N' },
        ],
        value: '0.5',
      });
      const selWin = stage.addSelect({
        label: 'Type de fenêtre',
        options: [{ value: 'hann', label: 'Hann' }, { value: 'rect', label: 'Rectangulaire' }],
        value: 'hann',
      });

      /* ---------- valeurs animées (easées à chaque frame rendue, même en pause) ---------- */
      const sm = { n: 512, hop: 256, rect: 0, long: 2 / 3, lat: 32 };
      let pulse = 0;

      /* Extrait N éch. finissant à endAbs, fenêtre + FFT réelle, mappe en 0..1 (log) */
      function spectrumTo(endAbs, N, useHann, out) {
        const s = seg[N], base = endAbs - N + 1;
        for (let i = 0; i < N; i++) s[i] = ring[(base + i) % VIS];
        const mags = U.rfftMag(s, useHann);
        const g = useHann ? 40 : 20; // rfftMag compense Hann (×2) → on égalise les deux fenêtres
        for (let i = 0; i < out.length; i++) out[i] = Math.log1p(mags[i] * g) / LOG41;
        return out;
      }

      /* Trace le contour de la fenêtre d'analyse (Hann ou rectangulaire) */
      function winPath(ctx, x, w, yBot, h, useHann) {
        ctx.beginPath();
        ctx.moveTo(x, yBot);
        if (useHann) {
          for (let i = 0; i <= 48; i++) ctx.lineTo(x + (i / 48) * w, yBot - (i < 48 ? winShape[i] : 0) * h);
        } else {
          ctx.lineTo(x, yBot - h); ctx.lineTo(x + w, yBot - h); ctx.lineTo(x + w, yBot);
        }
      }

      /* Texte avec fond sombre translucide (lisible sur le spectrogramme) */
      function tag(ctx, s, x, y, o = {}) {
        ctx.save();
        ctx.font = `600 ${o.size || 10}px ${U.FONT}`;
        const tw = ctx.measureText(s).width;
        const ax = o.align === 'right' ? x - tw : o.align === 'center' ? x - tw / 2 : x;
        ctx.globalAlpha = (o.alpha ?? 1) * 0.72;
        ctx.fillStyle = palette.stage;
        ctx.fillRect(ax - 4, y - (o.size || 10) - 2, tw + 8, (o.size || 10) + 7);
        ctx.globalAlpha = o.alpha ?? 1;
        U.text(ctx, s, ax, y, { size: o.size || 10, color: o.color || palette.dim, bold: true });
        ctx.restore();
      }

      /* Waveform min/max par colonne de pixel (1,5 s du buffer circulaire) */
      function drawWaveBand(ctx, x, y, w, h, now) {
        ctx.save();
        ctx.strokeStyle = palette.grid;
        ctx.beginPath(); ctx.moveTo(x + 1, y + h / 2); ctx.lineTo(x + w - 1, y + h / 2); ctx.stroke();
        ctx.fillStyle = palette.voice;
        ctx.globalAlpha = 0.85;
        const cy = y + h / 2, sc = h * 0.46;
        const first = now - VIS + 1;
        const cols = Math.max(2, Math.floor(w) - 2);
        for (let px = 0; px < cols; px++) {
          const s0 = first + Math.floor((px / cols) * VIS);
          const s1 = first + Math.floor(((px + 1) / cols) * VIS);
          let mn = 1e9, mx = -1e9;
          for (let s = s0; s < s1; s++) { const v = ring[s % VIS]; if (v < mn) mn = v; if (v > mx) mx = v; }
          if (mn > mx) { mn = 0; mx = 0; }
          const yT = cy - U.clamp(mx, -1, 1) * sc;
          const yB = cy - U.clamp(mn, -1, 1) * sc;
          ctx.fillRect(x + 1 + px, yT, 1, Math.max(1, yB - yT));
        }
        ctx.restore();
      }

      /* Fenêtre courante surlignée + fantômes espacés du hop, à droite de la waveform */
      function drawWindowsMain(ctx, x, y, w, h, useHann) {
        const pps = (w - 2) / VIS;
        const wn = Math.max(2, sm.n * pps);
        const hp = Math.max(0.5, sm.hop * pps);
        const xr = x + w - 1;
        const yBot = y + h - 3, hc = Math.max(6, h - 12);
        ctx.save();
        ctx.strokeStyle = palette.voice;
        for (let k = 4; k >= 1; k--) {              // fantômes des 4 trames précédentes
          ctx.globalAlpha = 0.34 * (1 - k / 5);
          ctx.lineWidth = 1;
          winPath(ctx, xr - wn - k * hp, wn, yBot, hc, useHann);
          ctx.stroke();
        }
        ctx.globalAlpha = 0.13; ctx.fillStyle = palette.voice;
        ctx.fillRect(xr - wn, y + 2, wn, h - 4);    // rectangle corail de la fenêtre courante
        ctx.globalAlpha = 0.18;
        winPath(ctx, xr - wn, wn, yBot, hc, useHann); ctx.fill();
        ctx.globalAlpha = 0.95; ctx.lineWidth = 1.5;
        winPath(ctx, xr - wn, wn, yBot, hc, useHann); ctx.stroke();
        ctx.restore();
        return xr - wn;
      }

      /* Loupe : la fenêtre + ses 4 prédécesseures, recouvrement et hop annotés */
      function drawZoom(ctx, zx, zy, zw, zh, now, N, hop, hopFrac, useHann) {
        U.frame(ctx, zx, zy, zw, zh, 'Loupe — fenêtre, Hann & recouvrement');
        const span = N + 4 * hop, base = now - span + 1;
        for (let i = 0; i < span; i++) zoomBuf[i] = ring[(base + i) % VIS];
        const ix = zx + 3, iw = zw - 6;
        U.wave(ctx, zoomBuf.subarray(0, span), ix, zy + 8, iw, Math.max(8, zh - 34),
          { color: palette.voice, lw: 1, alpha: 0.5, scale: 0.95 });
        const pps = iw / span, wN = N * pps, hpx = hop * pps;
        const yBot = zy + zh - 18, hc = Math.max(6, zh - 32);
        const x0 = ix + (span - N) * pps;
        ctx.save();
        ctx.strokeStyle = palette.voice;
        for (let k = 4; k >= 1; k--) {
          ctx.globalAlpha = 0.5 * (1 - k / 5.2);
          ctx.lineWidth = 1;
          winPath(ctx, x0 - k * hpx, wN, yBot, hc, useHann);
          ctx.stroke();
        }
        const ovw = wN - hpx;                       // zone de recouvrement avec la trame précédente
        if (ovw > 3) {
          ctx.globalAlpha = 0.1; ctx.fillStyle = palette.mix;
          ctx.fillRect(x0, yBot - hc, ovw, hc);
          ctx.globalAlpha = 1;
          if (ovw > 52 && zh > 70)
            tag(ctx, `recouvrement ${U.fmt.pct(1 - hopFrac)}`, x0 + ovw / 2, zy + 18,
              { color: palette.mix, size: 9, align: 'center' });
        }
        ctx.globalAlpha = 0.14; ctx.fillStyle = palette.voice;
        winPath(ctx, x0, wN, yBot, hc, useHann); ctx.fill();
        ctx.globalAlpha = 1; ctx.lineWidth = 1.6;
        winPath(ctx, x0, wN, yBot, hc, useHann); ctx.stroke();
        U.arrow(ctx, x0 - hpx, zy + zh - 9, x0, zy + zh - 9, { color: palette.blue, lw: 1.2, head: 5 });
        if (hpx > 40)
          tag(ctx, `hop = ${U.fmt.ms(hop / SR)}`, x0 - hpx / 2, zy + zh - 12,
            { color: palette.blue, size: 8.5, align: 'center' });
        ctx.restore();
        return { x0, x1: x0 + wN };
      }

      /* Panneau de lectures (mode large) — toutes les valeurs sont calculées */
      function drawReadouts(ctx, rx, ry, rw, rh, N, hop) {
        U.frame(ctx, rx, ry, rw, rh, 'Lectures — valeurs réelles');
        ctx.save();
        U.roundRect(ctx, rx + 1, ry + 1, rw - 2, rh - 2, 6);
        ctx.fillStyle = palette.panel; ctx.globalAlpha = 0.5; ctx.fill(); ctx.globalAlpha = 1;
        const px = rx + 10, pr = rx + rw - 10;
        let yy = ry + 18;
        const row = (labFull, labShort, val, color) => {
          U.text(ctx, rw > 260 ? labFull : labShort, px, yy, { size: 9.5, color: palette.dim });
          U.text(ctx, val, pr, yy, { size: 11, color, align: 'right', bold: true, mono: true });
          yy += 15;
        };
        row('Résolution temporelle Δt = N/sr', 'Δt = N/sr', U.fmt.ms(N / SR), palette.mix);
        row('Résolution fréquentielle Δf = sr/N', 'Δf = sr/N', U.fmt.hz(SR / N), palette.blue);
        if (rh > 150) {
          row('Cadence d’analyse sr/hop', 'sr/hop', `${(SR / hop).toFixed(1)} trames/s`, palette.dim);
          U.text(ctx, `Δt × Δf = ${((N / SR) * (SR / N)).toFixed(0)} — incertitude de Gabor`, px, yy,
            { size: 9, color: palette.faint });
          yy += 14;
        }
        yy += 4;
        U.text(ctx, 'LATENCE ALGORITHMIQUE', px, yy, { size: 9.5, color: palette.orange, bold: true });
        yy += Math.min(26, Math.max(18, rh * 0.16));
        U.text(ctx, U.fmt.ms(N / SR), px, yy,
          { size: Math.min(26, Math.max(16, rh * 0.17)), color: palette.orange, bold: true, mono: true });
        U.text(ctx, '= durée de fenêtre', pr, yy, { size: 9, color: palette.dim, align: 'right' });
        yy += 13;
        if (rh > 124) {
          U.text(ctx, '« ce que coûte cette fenêtre en latence temps réel »', px, yy,
            { size: 8.5, color: palette.dim });
        }
        if (rh > 96) {                               // jauge 0–70 ms, budget temps réel en vert
          const gy = ry + rh - 14, gw = rw - 20;
          const X = (ms) => px + U.clamp(ms / 70, 0, 1) * gw;
          ctx.fillStyle = palette.panel2; ctx.fillRect(px, gy, gw, 7);
          ctx.fillStyle = palette.green; ctx.globalAlpha = 0.3;
          ctx.fillRect(X(10), gy, X(20) - X(10), 7); ctx.globalAlpha = 1;
          U.text(ctx, '10–20 ms = budget temps réel', X(15) + 14, gy - 4, { size: 8, color: palette.green });
          const mx = X(sm.lat);
          ctx.fillStyle = palette.orange; ctx.fillRect(mx - 1, gy - 2, 2, 11);
          U.glowDot(ctx, mx, gy + 3.5, 2.5 + 2 * U.ease(pulse), palette.orange);
          U.text(ctx, '0', px, gy - 4, { size: 8, color: palette.faint });
          U.text(ctx, '70 ms', px + gw, gy - 4, { size: 8, color: palette.faint, align: 'right' });
        }
        ctx.restore();
      }

      /* Bandeau compact de lectures (mode étroit) */
      function drawStrip(ctx, x, y, w, h, N, hop) {
        ctx.save();
        U.roundRect(ctx, x, y, w, h, 6);
        ctx.fillStyle = palette.panel; ctx.fill();
        ctx.strokeStyle = palette.grid; ctx.stroke();
        U.text(ctx, `Δt = ${U.fmt.ms(N / SR)}`, x + 8, y + 16, { size: 10, color: palette.mix, mono: true, bold: true });
        U.text(ctx, `Δf = ${U.fmt.hz(SR / N)}`, x + w * 0.38, y + 16, { size: 10, color: palette.blue, mono: true, bold: true });
        U.text(ctx, `${(SR / hop).toFixed(0)} trames/s`, x + w - 8, y + 16, { size: 9, color: palette.dim, align: 'right', mono: true });
        U.text(ctx, 'Latence algorithmique :', x + 8, y + h - 9, { size: 9.5, color: palette.dim });
        U.text(ctx, U.fmt.ms(N / SR), x + 124, y + h - 9, { size: 13, color: palette.orange, bold: true, mono: true });
        if (w > 420)
          U.text(ctx, '← le coût temps réel de cette fenêtre', x + w - 8, y + h - 9,
            { size: 9, color: palette.faint, align: 'right' });
        ctx.restore();
      }

      /* ================================ boucle ================================ */
      stage.onFrame((t, dt) => {
        const ctx = stage.ctx, W = stage.W, H = stage.H;
        const N = +selN.value;
        const hopFrac = +selHop.value;
        const hop = Math.max(1, Math.round(N * hopFrac));
        const useHann = selWin.value === 'hann';
        const narrow = W < 560;

        /* — avancer le signal (échantillons réellement générés au fil de t) — */
        const now = Math.floor(t * SR) + VIS;
        if (now - absIdx > 2400) { absIdx = now - 2400; nextHop = Math.max(nextHop, absIdx); }
        while (absIdx <= now) { ring[absIdx % VIS] = U.gen.speech(absIdx / SR); absIdx++; }

        /* — trames STFT écoulées (compteur de hop) → colonnes du spectrogramme — */
        if (nextHop < now - 14 * hop) nextHop = now - 14 * hop;
        let guard = 14;
        while (nextHop <= now && guard-- > 0) {
          scroller.push(spectrumTo(nextHop, N, useHann, vmap[N]));
          nextHop += hop;
          pulse = 1;
        }
        /* — spectre instantané de la trame courante — */
        const spec = spectrumTo(now, N, useHann, vmap[N]);

        /* — easing (continue même en pause pour refléter les contrôles) — */
        sm.n = U.lerp(sm.n, N, 0.16);
        sm.hop = U.lerp(sm.hop, hop, 0.16);
        sm.rect = U.lerp(sm.rect, useHann ? 0 : 1, 0.12);
        sm.long = U.lerp(sm.long, (Math.log2(N) - 7) / 3, 0.1);
        sm.lat = U.lerp(sm.lat, (N / SR) * 1000, 0.15);
        pulse = Math.max(0, pulse - dt * 5);

        /* — mise en page (recalculée à chaque frame depuis W/H) — */
        stage.clear();
        const m = narrow ? 8 : 12;
        const top = narrow ? 22 : 30;
        const gap = narrow ? 20 : 24;
        const stripH = narrow ? 44 : 0;
        const innerW = W - 2 * m;
        const avail = Math.max(90, H - top - m - 2 * gap - (narrow ? stripH + 8 : 0));
        const h1 = Math.max(36, Math.round(avail * 0.30));
        const h2 = Math.max(44, Math.round(avail * 0.34));
        const h3 = Math.max(36, avail - h1 - h2);
        const y1 = top, y2 = y1 + h1 + gap;
        const yS = y2 + h2 + 8;
        const y3 = narrow ? yS + stripH + gap : y2 + h2 + gap;
        const zoomW = narrow ? 0 : Math.round(innerW * 0.40);
        const wfW = narrow ? innerW : innerW - zoomW - 14;
        const specW = narrow ? innerW : Math.round(innerW * 0.58);

        /* — entête — */
        if (!narrow) {
          let hx = m;
          hx += U.chip(ctx, 'voix — gen.speech', hx, 13, { color: palette.voice }) + 8;
          hx += U.chip(ctx, 'sr = 16 kHz', hx, 13, { color: palette.blue }) + 8;
          U.chip(ctx, `FFT ${N} points`, hx, 13, { color: palette.mix });
          U.text(ctx, 'STFT — Short-Time Fourier Transform', W - m, 17, { size: 11, color: palette.dim, align: 'right' });
        } else {
          U.text(ctx, 'STFT — fenêtre glissante + FFT', m, 14, { size: 10, color: palette.dim, bold: true });
        }

        /* — bandeau 1 : signal + fenêtre d'analyse — */
        U.frame(ctx, m, y1, wfW, h1, narrow ? 'Signal — 1,5 s' : 'Signal (pseudo-parole) — 1,5 s visibles');
        drawWaveBand(ctx, m, y1, wfW, h1, now);
        drawWindowsMain(ctx, m, y1, wfW, h1, useHann);
        if (!narrow) {
          tag(ctx, `fenêtre : N = ${N} éch. = ${U.fmt.ms(N / SR)}`, m + wfW - 6, y1 + 16, { color: palette.voice, size: 9, align: 'right' });
          tag(ctx, `hop = ${U.fmt.ms(hop / SR)} · recouvrement ${U.fmt.pct(1 - hopFrac)}`, m + wfW - 6, y1 + 29, { color: palette.blue, size: 9, align: 'right' });
          U.text(ctx, '−1,5 s', m + 5, y1 + h1 - 5, { size: 8.5, color: palette.faint });
          U.text(ctx, 'maintenant →', m + wfW - 5, y1 + h1 - 5, { size: 8.5, color: palette.faint, align: 'right' });
        } else {
          tag(ctx, `N = ${N} (${U.fmt.ms(N / SR)})`, m + wfW - 4, y1 + 14, { color: palette.voice, size: 8.5, align: 'right' });
        }
        if (!narrow) {
          const zr = drawZoom(ctx, m + wfW + 14, y1, zoomW, h1, now, N, hop, hopFrac, useHann);
          if (h1 > 56) {                            // rayons de loupe (marqueur → zoom)
            ctx.save();
            ctx.strokeStyle = palette.voice; ctx.globalAlpha = 0.16; ctx.setLineDash([3, 4]);
            const mxL = m + wfW - 1 - Math.max(2, sm.n * (wfW - 2) / VIS);
            ctx.beginPath();
            ctx.moveTo(mxL, y1 + 4); ctx.lineTo(zr.x0, y1 + 8);
            ctx.moveTo(m + wfW - 1, y1 + 4); ctx.lineTo(zr.x1, y1 + 8);
            ctx.stroke(); ctx.setLineDash([]);
            ctx.restore();
          }
        }

        /* — bandeau 2 : spectre instantané (vraie FFT) + lectures — */
        U.frame(ctx, m, y2, specW, h2, narrow ? 'Spectre de la trame' : 'Spectre instantané de la trame courante');
        U.bars(ctx, spec, m + 2, y2 + 4, specW - 4, Math.max(8, h2 - 22),
          { colorFn: (x01, v) => U.magma(v), gap: spec.length > 160 ? 0 : 1, alpha: 0.95 });
        const ticks = narrow ? [0, 4000, 8000] : [0, 2000, 4000, 6000, 8000];
        for (const f of ticks) {
          const tx = m + 2 + (f / (SR / 2)) * (specW - 4);
          U.text(ctx, U.fmt.hz(f), U.clamp(tx, m + 14, m + specW - 18), y2 + h2 - 6,
            { size: 9, color: palette.faint, align: 'center' });
        }
        if (!narrow) {
          tag(ctx, `1 case = Δf = ${U.fmt.hz(SR / N)}`, m + specW - 8, y2 + 16, { color: palette.blue, size: 9, align: 'right' });
          if (sm.rect > 0.02)
            tag(ctx, 'rectangulaire → fuites spectrales (lobes)', m + 8, y2 + 16,
              { color: palette.red, size: 9, alpha: U.smoothstep(sm.rect) });
          drawReadouts(ctx, m + specW + 14, y2, innerW - specW - 14, h2, N, hop);
        } else {
          drawStrip(ctx, m, yS, innerW, stripH, N, hop);
        }

        /* — bandeau 3 : spectrogramme défilant — */
        U.frame(ctx, m, y3, innerW, h3, narrow ? 'Spectrogramme |STFT|' : 'Spectrogramme |STFT| — 1 colonne par trame (hop)');
        scroller.draw(ctx, m + 1, y3 + 1, innerW - 2, h3 - 2);
        ctx.save();                                 // bord d'écriture, pulse à chaque trame
        ctx.globalAlpha = 0.25 + 0.6 * U.ease(pulse);
        ctx.fillStyle = palette.voice;
        ctx.fillRect(m + innerW - 3, y3 + 1, 2, h3 - 2);
        ctx.restore();
        tag(ctx, '8 kHz', m + 6, y3 + 14, { size: 8.5, color: palette.dim });
        tag(ctx, '0 Hz', m + 6, y3 + h3 - 5, { size: 8.5, color: palette.dim });
        if (!narrow) {
          tag(ctx, `1 col = ${U.fmt.ms(hop / SR)} → ${(SR / hop).toFixed(0)} col/s`, m + innerW / 2, y3 + h3 - 5,
            { size: 8.5, align: 'center', color: palette.dim });
          tag(ctx, 'stries verticales = plosives (transitoires)', m + innerW - 8, y3 + h3 - 5,
            { size: 8.5, align: 'right', color: palette.faint });
          /* annotation du compromis temps/fréquence, fondu selon N (easé) */
          const aL = U.smoothstep(sm.long);
          if (h3 > 96) {
            tag(ctx, 'compromis de Gabor : Δt ↔ Δf', m + innerW - 8, y3 + 16, { size: 9, align: 'right', color: palette.mix });
            tag(ctx, 'fenêtre longue : raies fines (Δf ↓) mais plosives étalées', m + innerW - 8, y3 + 29,
              { size: 9, align: 'right', color: palette.rest, alpha: 0.25 + 0.7 * aL });
            tag(ctx, 'fenêtre courte : plosives nettes (Δt ↓) mais raies épaisses', m + innerW - 8, y3 + 42,
              { size: 9, align: 'right', color: palette.rest, alpha: 0.25 + 0.7 * (1 - aL) });
          } else {
            tag(ctx, aL > 0.5 ? 'fenêtre longue : Δf fin, plosives étalées' : 'fenêtre courte : plosives nettes, Δf grossier',
              m + innerW - 8, y3 + 16, { size: 9, align: 'right', color: palette.rest });
          }
        }
      });
    },
  });
})();
