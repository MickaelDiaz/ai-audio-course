# Audio AI Atlas

**Les architectures d'IA audio, en mouvement.** Web-app 100 % statique, gratuite, sans compte,
sans tracking, installable sur iPhone (PWA) et utilisable hors-ligne. Du canvas et des maths,
zéro dépendance.

17 modules interactifs et animés, organisés en 4 catégories :

| Catégorie | Modules |
|---|---|
| **Signal & DSP** | Échantillonnage · FFT · STFT & spectrogramme · Filterbanks Mel/ERB · Masquage & cohérence additive |
| **Couches neuronales** | Conv1D · Conv2D · RNN/GRU/LSTM · Attention/Transformer · SSM/Mamba |
| **Architectures** | Encoder-Decoder/U-Net · DPRNN · Deep Filtering · Génératifs (GAN/Diffusion/Flow Matching) · Pipeline complet |
| **Temps réel & embarqué** | Latence & look-ahead · Quantification FP16/INT8 & NPU |

## Lancer en local

N'importe quel serveur statique fait l'affaire :

```bash
# Python
python -m http.server 4173
# ou Node
npx serve -l 4173
```

Puis ouvrir <http://localhost:4173>.

## Déployer gratuitement (pour l'avoir sur iPhone)

Le plus simple — **GitHub Pages** :

1. Créer un repo, pousser le contenu de ce dossier.
2. Settings → Pages → Source : branche `main`, dossier `/ (root)`.
3. L'app est servie en HTTPS sur `https://<user>.github.io/<repo>/`.

Alternative : **Cloudflare Pages** (glisser-déposer le dossier dans le dashboard) ou
**Netlify Drop**. Tout est statique, aucun build.

## Installer sur iPhone

1. Ouvrir l'URL déployée dans **Safari**.
2. Bouton **Partager** → **« Sur l'écran d'accueil »**.
3. L'app s'ouvre en plein écran, icône dédiée, et fonctionne **hors-ligne**
   (service worker, cache complet au premier chargement).

## Architecture du code

```
index.html            shell + chargement des modules
styles.css            thème sombre
app.js                routeur hash, accueil, vue module
core/viz-core.js      moteur (rAF, DPR, contrôles) + utilitaires (FFT réelle,
                      colormaps magma/viridis, générateurs de signaux, helpers canvas)
modules/<id>.js       un fichier = un module de visualisation (contrat dans docs/MODULE_SPEC.md)
sw.js                 service worker (cache-first, offline complet)
manifest.webmanifest  PWA
```

Ajouter un module : créer `modules/mon-id.js` suivant `docs/MODULE_SPEC.md`,
ajouter la balise `<script>` dans `index.html` et l'entrée dans `sw.js`.
