# Contrat des modules — Audio AI Atlas

Chaque module est UN fichier `modules/<id>.js`, JavaScript vanilla, chargé par `<script>` classique
(PAS de `import`/`export`/ESM). Tout le fichier est enveloppé dans une IIFE. Zéro dépendance externe,
zéro requête réseau, zéro `setInterval`/`setTimeout` (utiliser uniquement `stage.onFrame`).

## Enregistrement

```js
(function () {
  'use strict';
  const { U, palette } = Atlas;

  AtlasRegister({
    id: 'fft',                       // = nom du fichier sans .js
    title: 'FFT — du temps à la fréquence',
    category: 'signal',              // 'signal' | 'layers' | 'archi' | 'system'
    icon: 'ƒ',                       // 1–2 caractères (unicode ok)
    summary: 'Une phrase courte pour la carte d’accueil.',
    explain: `
      <p>2 à 4 paragraphes HTML en français. <strong>Termes clés</strong> en gras,
      termes techniques anglais standards conservés. Peut contenir <ul><li>…</li></ul>
      et <code>code</code>.</p>`,
    init(stage) {
      // créer l'état, les contrôles, puis :
      stage.onFrame((t, dt) => {
        stage.clear();               // fond — TOUJOURS en premier
        // ... dessiner avec stage.ctx en pixels CSS (DPR géré par le moteur)
      });
      // return () => {...}  // cleanup optionnel (rarement utile)
    },
  });
})();
```

## API `stage`

- `stage.ctx` — CanvasRenderingContext2D, déjà mis à l'échelle DPR : dessiner en pixels CSS.
- `stage.W`, `stage.H` — taille courante en px CSS (mise à jour auto au resize ; relire à chaque frame).
- `stage.onFrame(cb)` — `cb(t, dt)` appelé à chaque rAF. `t` = secondes cumulées (respecte pause/vitesse),
  `dt` = delta secondes (0 si en pause — le rendu doit continuer pour refléter les contrôles).
- `stage.onResize(cb)` — optionnel, pour reconstruire des buffers dépendant de la taille.
- `stage.clear(color?)` — remplit le fond (défaut `palette.stage`).
- Contrôles (ajoutés sous le canvas, dans l'ordre d'appel) :
  - `stage.addSlider({label, min, max, step, value, format?, onChange?})` → `{value}` (getter/setter)
  - `stage.addSelect({label, options: ['a','b'] | [{value,label}], value?, onChange?})` → `{value}` (string !)
  - `stage.addToggle({label, value?, onChange?})` → `{value}` (booléen)
  - `stage.addButton({label, onClick})`
  Lire `.value` dans onFrame suffit (pas besoin de onChange sauf reconstruction coûteuse).

## Utilitaires `Atlas.U`

Maths : `TAU, clamp(v,a,b), lerp(a,b,t), smoothstep(t), ease(t), hash(n)` (pseudo-aléa déterministe 0–1), `noise1(x)` (bruit lisse).
DSP : `rfftMag(Float32Array_pow2, hann=true)` → `Float32Array N/2` magnitudes ~0–1 ; `rfft(sig)` → `{re,im}` ;
`hann(N)` ; `fft(re,im)` in-place.
Colormaps : `magma(t)` / `viridis(t)` → string CSS ; `magmaRGB(t)` / `viridisRGB(t)` → `[r,g,b]`.
Signaux (déterministes, continus en temps absolu — échantillonner à 16 kHz en général) :
`gen.speech(t)` (pseudo-parole : harmoniques + plosives + fricatives), `gen.music(t)` (accords + basse + hi-hat),
`gen.noise(t)`, `gen.sine(t,f)`, `gen.chirp(t,f0,f1,T)`, `gen.buffer(fn,N,sr,t0)` → Float32Array.
Dessin : `text(ctx,s,x,y,{size,color,align,baseline,bold,mono})`, `roundRect(ctx,x,y,w,h,r)` (path seulement),
`node(ctx,x,y,w,h,{title,sub,color,active,fill,size})` (boîte de diagramme),
`arrow(ctx,x1,y1,x2,y2,{color,lw,head,dash,alpha})`, `wave(ctx,data,x,y,w,h,{color,lw,scale,alpha})`,
`bars(ctx,data,x,y,w,h,{color,colorFn:(x01,v)=>css,scale,gap,alpha})`, `frame(ctx,x,y,w,h,label?)`,
`chip(ctx,s,x,y,{color,size})` → largeur, `glowDot(ctx,x,y,r,color)`.
Formatage : `fmt.hz(f), fmt.ms(secondes), fmt.db(d), fmt.pct(p01), fmt.k(n)`.
Spectrogramme défilant : `new U.Scroller(wPx,hPx,cmapRGB?)` ; `.push(values01)` (index 0 = grave, 1 colonne/appel) ;
`.draw(ctx,x,y,w,h)`.

## Palette (`Atlas.palette`) — DA « Studio Mono », à respecter STRICTEMENT

Coque noir chaud + UN accent corail. Neutres : `bg #0c0c0e, stage #0e0e11, panel #161619, panel2 #1f1f23,
grid, text #f4f2ec, dim #9b988f, faint #605d57`.
**Accent de marque** : `accent #ff5c49` (corail) — réservé aux moments d'identité/interaction.
Sémantique audio constante dans toute l'app : **voice/corail `#ff5c49`** (= la marque), **rest (musique/effets)/ambre `#fbbf24`**,
**mix/violet `#a78bfa`**. Divers : `blue #60a5fa, pink #f472b6, red #f87171, green #34d399, orange #fb923c, yellow #facc15, teal #38bdf8`.

## Règles de qualité

1. **Responsive** : calculer TOUTE la mise en page depuis `stage.W/stage.H` à chaque frame (fractions + marges
   minimales). Doit rester lisible de 360×300 (iPhone portrait) à 1030×560 (desktop). Si la largeur < 560 px,
   simplifier (masquer les annotations secondaires, réduire les tailles de police, empiler verticalement).
2. **Performance 60 fps** : pré-allouer les buffers (pas de `new Float32Array` par frame sauf petit), FFT ≤ 1024
   par frame, Scroller ≤ 2–3 par module, pas de gradients/ombres sur des centaines d'éléments.
3. **Beauté** : fond `stage.clear()`, hiérarchie typographique (titres 12–13 px bold, annotations 10–11 px `dim`),
   alpha doux, halos (`glowDot`, `node({active:true})`) avec parcimonie, animations EASÉES (`U.ease`, `U.smoothstep`),
   transitions continues (jamais de saut sec : interpoler).
4. **Pédagogie** : chaque animation raconte UNE histoire ; annoter les éléments clés directement sur le canvas
   (en français) ; les nombres affichés doivent être VRAIS (calculés, pas décoratifs).
5. **Robustesse** : aucune exception même si W/H petit ; `dt` peut être 0 (pause) ; `t` peut être grand (heures) —
   utiliser des modulos pour les cycles ; ne JAMAIS toucher au DOM hors des helpers `stage.*`.
6. 2 à 4 contrôles interactifs pertinents, étiquettes en français.
