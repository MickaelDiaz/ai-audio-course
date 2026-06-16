/* ============================================================
   Audio AI Atlas — glossaire central
   Source unique de vérité pour les définitions de termes.
   Les modules référencent un terme via <dfn data-term="id">…</dfn>
   dans leur texte « explain » ; app.js affiche la définition au tap
   et la page #/glossary les liste toutes.

   Convention d'id : kebab-case, concept en clair (souvent l'anglais
   usuel du domaine). Réutiliser un id existant plutôt qu'en créer un
   doublon. Une définition = 1 à 3 phrases, sans présupposé, lisibles
   dans le métro.
   ============================================================ */
(function () {
  'use strict';
  if (!window.Atlas) { console.error('glossary.js chargé avant viz-core.js'); return; }

  Atlas.defineTerms({
    /* ---------- Signal & échantillonnage ---------- */
    'echantillonnage': {
      term: 'Échantillonnage',
      def: "Mesurer l'amplitude d'un son à intervalles réguliers pour le transformer en une suite de nombres (les « échantillons »). C'est le passage de l'analogique au numérique.",
    },
    'sample': {
      term: 'Échantillon (sample)',
      def: "Une mesure unique de l'amplitude du signal à un instant donné. Un son numérique n'est qu'une longue liste de ces nombres.",
    },
    'sample-rate': {
      term: "Fréquence d'échantillonnage",
      def: "Le nombre d'échantillons mesurés par seconde, en hertz (Hz). 16 kHz = 16 000 mesures par seconde, courant pour la voix ; 44,1 kHz pour la musique.",
    },
    'nyquist': {
      term: 'Fréquence de Nyquist',
      def: "La moitié de la fréquence d'échantillonnage. C'est la plus haute fréquence qu'on peut représenter fidèlement ; au-delà, le son se replie en parasites (aliasing).",
    },
    'aliasing': {
      term: 'Repliement (aliasing)',
      def: "Quand une fréquence dépasse la limite de Nyquist, elle réapparaît à une fausse fréquence plus basse — un artefact audible qu'on évite en filtrant avant d'échantillonner.",
    },
    'amplitude': {
      term: 'Amplitude',
      def: "La « hauteur » de l'onde à un instant donné : elle correspond au volume sonore. Souvent normalisée entre −1 et +1.",
    },
    'waveform': {
      term: "Forme d'onde",
      def: "La représentation du son dans le temps : l'amplitude en fonction du temps. C'est le signal brut, avant toute analyse fréquentielle.",
    },
    'frequence': {
      term: 'Fréquence',
      def: "Le nombre d'oscillations par seconde d'une onde, en hertz (Hz) : elle détermine la hauteur perçue (grave = basse fréquence, aigu = haute).",
    },
    'phase': {
      term: 'Phase',
      def: "Le décalage d'une onde dans le temps, mesuré en angle. Deux ondes de même fréquence mais de phases différentes sont décalées l'une par rapport à l'autre.",
    },

    /* ---------- Analyse fréquentielle ---------- */
    'fft': {
      term: 'FFT',
      def: "Transformée de Fourier rapide : un algorithme qui décompose un signal en ses fréquences constituantes. Elle répond à « quelles notes composent ce son ? ».",
    },
    'dft': {
      term: 'DFT',
      def: "Transformée de Fourier discrète : le calcul mathématique qui convertit des échantillons en spectre de fréquences. La FFT en est la version rapide.",
    },
    'stft': {
      term: 'STFT',
      def: "Transformée de Fourier à court terme : on découpe le son en petites fenêtres successives et on calcule la FFT de chacune. Résultat : comment le spectre évolue dans le temps.",
    },
    'spectre': {
      term: 'Spectre',
      def: "La répartition de l'énergie d'un son selon la fréquence, à un instant donné : l'« empreinte fréquentielle » du signal.",
    },
    'spectrogram': {
      term: 'Spectrogramme',
      def: "Une image du son : le temps en abscisse, la fréquence en ordonnée, et la couleur indiquant l'énergie. C'est une suite de spectres mis côte à côte (résultat de la STFT).",
    },
    'magnitude': {
      term: 'Magnitude',
      def: "L'intensité d'une fréquence dans le spectre (sa « force »). On l'oppose à la phase, qui en code le décalage temporel.",
    },
    'fft-bin': {
      term: 'Bin (de fréquence)',
      def: "Une case du spectre : la FFT répartit les fréquences en intervalles égaux, et chaque bin regroupe l'énergie d'une petite bande de fréquences.",
    },
    'window': {
      term: 'Fenêtre (de Hann)',
      def: "Avant la FFT, on multiplie chaque tranche du signal par une courbe en cloche qui s'éteint aux bords. Cela évite les discontinuités qui créeraient de fausses fréquences.",
    },
    'hop': {
      term: 'Pas (hop)',
      def: "Le décalage en échantillons entre deux fenêtres successives de la STFT. Un petit hop donne plus de trames (meilleure résolution temporelle), mais plus de calcul.",
    },
    'frame': {
      term: 'Trame',
      def: "Une fenêtre temporelle du signal traitée d'un bloc (par la STFT ou un réseau). Une colonne du spectrogramme correspond à une trame.",
    },
    'mel': {
      term: 'Échelle mel',
      def: "Une échelle de fréquences calquée sur l'oreille humaine : elle resserre les aigus et étire les graves, là où notre perception est plus fine. Base des spectrogrammes mel.",
    },
    'mfcc': {
      term: 'MFCC',
      def: "Coefficients cepstraux mel : une compression du spectre mel en une poignée de nombres décrivant la « forme » du timbre. Historiquement clé en reconnaissance vocale.",
    },
    'db': {
      term: 'Décibel (dB)',
      def: "Une échelle logarithmique pour l'intensité sonore : elle suit la perception (chaque +10 dB ≈ deux fois plus fort) et compresse une énorme plage de valeurs.",
    },
    'filterbank': {
      term: 'Banc de filtres',
      def: "Un ensemble de filtres qui découpent le son en bandes de fréquences, comme les curseurs d'un égaliseur. Brique de base des spectrogrammes mel et de nombreux encodeurs.",
    },

    /* ---------- Contenu de la parole ---------- */
    'f0': {
      term: 'Fréquence fondamentale (f0)',
      def: "La fréquence de vibration des cordes vocales : elle fixe la hauteur perçue de la voix. Les harmoniques en sont des multiples entiers.",
    },
    'harmonique': {
      term: 'Harmonique',
      def: "Une fréquence multiple de la fondamentale (×2, ×3, …). L'empilement des harmoniques donne au son sa richesse ; sur un spectrogramme, ce sont des lignes horizontales parallèles.",
    },
    'partial': {
      term: 'Partiel',
      def: "Une composante fréquentielle pure d'un son tenu (voyelle, note). Sur un spectrogramme, un partiel apparaît comme une raie horizontale stable.",
    },
    'formant': {
      term: 'Formant',
      def: "Une zone de fréquences renforcée par les résonances du conduit vocal. La position des formants distingue les voyelles (« a », « i », « ou »…).",
    },
    'transient': {
      term: 'Transitoire',
      def: "Un événement sonore bref et brutal (attaque, claquement) qui couvre une large bande de fréquences. Sur un spectrogramme : un trait vertical.",
    },
    'plosive': {
      term: 'Plosive',
      def: "Une consonne « explosive » (p, t, k, b, d, g) : un bref silence suivi d'un éclat large bande. C'est un transitoire marqué.",
    },
    'fricative': {
      term: 'Fricative',
      def: "Une consonne « soufflée » (s, f, ch) : un bruit de turbulence riche en hautes fréquences, sans hauteur définie.",
    },
    'voiced': {
      term: 'Voisé',
      def: "Se dit d'un son produit avec vibration des cordes vocales (voyelles, « z », « v ») : il a une fondamentale et des harmoniques nettes, contrairement aux sons non voisés.",
    },

    /* ---------- Couches neuronales ---------- */
    'kernel': {
      term: 'Noyau (kernel)',
      def: "Un petit tableau de poids qu'on fait glisser sur le signal ou l'image. À chaque position, il combine les valeurs locales pour produire une sortie : c'est le cœur d'une convolution.",
    },
    'filtre': {
      term: 'Filtre',
      def: "Synonyme de noyau dans un réseau de convolution : un détecteur de motif appris. Une couche en apprend des dizaines, chacun réagissant à un type de structure.",
    },
    'fir': {
      term: 'Filtre FIR',
      def: "Filtre à réponse impulsionnelle finie : sa sortie est une somme pondérée d'un nombre fini d'échantillons d'entrée. Une convolution 1D est exactement un filtre FIR — mais aux poids appris.",
    },
    'convolution': {
      term: 'Convolution',
      def: "Opération qui fait glisser un noyau sur l'entrée et calcule, à chaque position, le produit scalaire entre les poids et les valeurs couvertes. Base des CNN.",
    },
    'dot-product': {
      term: 'Produit scalaire',
      def: "Multiplier deux listes de nombres terme à terme puis tout additionner. Mesure à quel point un motif (le noyau) « ressemble » au morceau de signal couvert.",
    },
    'weight-sharing': {
      term: 'Partage de poids',
      def: "Le même petit jeu de poids est réutilisé à toutes les positions, au lieu d'en apprendre un par endroit. C'est ce qui rend les CNN économes et capables de détecter un motif où qu'il soit.",
    },
    'feature-map': {
      term: "Carte d'activation (feature map)",
      def: "La sortie d'un filtre : une carte qui s'allume là où le motif détecté est présent dans l'entrée. Un filtre = une carte = un canal de sortie.",
    },
    'channel': {
      term: 'Canal',
      def: "Une des cartes empilées en sortie d'une couche, produite par un filtre. Une couche à 32 canaux applique 32 filtres distincts et empile leurs 32 réponses.",
    },
    'receptive-field': {
      term: 'Champ réceptif',
      def: "La portion d'entrée qui influence une sortie donnée. Plus il est large, plus le réseau « voit » de contexte. On l'agrandit en empilant des couches ou via la dilation.",
    },
    'dilation': {
      term: 'Dilation',
      def: "Espacer les points captés par un noyau (un sur deux, un sur quatre…) pour couvrir un plus large contexte sans ajouter de poids ni de calcul. Idée centrale de WaveNet et des TCN.",
    },
    'causal': {
      term: 'Causalité',
      def: "Une couche causale ne regarde que le passé, jamais le futur. Indispensable en temps réel : sinon il faudrait attendre des échantillons à venir, donc ajouter de la latence.",
    },
    'lookahead': {
      term: 'Look-ahead',
      def: "Le nombre d'échantillons futurs qu'une couche non causale doit attendre pour calculer sa sortie. Il se paie directement en latence.",
    },
    'padding': {
      term: 'Padding',
      def: "Ajouter des valeurs (souvent des zéros) sur les bords de l'entrée pour que le noyau puisse s'appliquer jusqu'aux extrémités et contrôler la taille de sortie.",
    },
    'stride': {
      term: 'Pas (stride)',
      def: "De combien le noyau avance entre deux applications. Un stride de 2 saute une position sur deux : la sortie est deux fois plus courte (sous-échantillonnage).",
    },
    'depthwise-separable': {
      term: 'Convolution depthwise separable',
      def: "Décompose une convolution en deux étapes moins coûteuses : un filtre spatial par canal (depthwise), puis un mélange des canaux par convolution 1×1 (pointwise). Standard de l'audio embarqué.",
    },
    'pointwise': {
      term: 'Convolution pointwise (1×1)',
      def: "Une convolution de noyau 1×1 : elle ne regarde pas les voisins, elle se contente de recombiner les canaux entre eux à chaque position.",
    },
    'parameters': {
      term: 'Paramètres (poids)',
      def: "Les nombres appris d'un réseau. Leur quantité mesure la taille du modèle et pèse sur la mémoire et la vitesse — critique sur mobile et NPU.",
    },
    'gradient-descent': {
      term: 'Descente de gradient',
      def: "La méthode d'apprentissage : on mesure l'erreur, on calcule dans quel sens ajuster chaque poids pour la réduire, et on répète. Le SGD en est la version par petits lots.",
    },
    'activation': {
      term: "Fonction d'activation",
      def: "Une fonction non linéaire (ReLU, GELU…) appliquée après chaque couche. Sans elle, empiler des couches reviendrait à une seule transformation linéaire.",
    },

    /* ---------- Séquences & attention ---------- */
    'rnn': {
      term: 'RNN',
      def: "Réseau récurrent : il traite la séquence pas à pas en gardant un état (mémoire) mis à jour à chaque instant. Naturellement causal et léger, mais difficile à paralléliser.",
    },
    'gru': {
      term: 'GRU',
      def: "Variante de RNN munie de « portes » qui décident quoi garder ou oublier de l'état. Plus simple que le LSTM, très utilisée pour l'audio temps réel.",
    },
    'lstm': {
      term: 'LSTM',
      def: "Réseau récurrent à mémoire longue : des portes protègent un état interne sur de longues durées, ce qui aide à capter des dépendances éloignées dans le temps.",
    },
    'hidden-state': {
      term: 'État caché',
      def: "La mémoire d'un réseau récurrent : un vecteur qui résume tout le passé vu jusqu'ici et qu'on met à jour à chaque nouvel échantillon.",
    },
    'attention': {
      term: 'Attention',
      def: "Un mécanisme où chaque position regarde toutes les autres et les pondère selon leur pertinence. Le modèle « fait attention » aux instants utiles, proches ou lointains.",
    },
    'self-attention': {
      term: 'Self-attention',
      def: "L'attention d'une séquence sur elle-même : chaque instant se compare à tous les autres instants de la même séquence pour se réécrire en fonction du contexte.",
    },
    'qkv': {
      term: 'Query / Key / Value',
      def: "Les trois rôles de l'attention : la requête (query) cherche, les clés (keys) indexent, les valeurs (values) sont récupérées. Une position pondère les valeurs selon l'accord query-clé.",
    },
    'softmax': {
      term: 'Softmax',
      def: "Transforme une liste de scores en proportions positives qui somment à 1 — des « poids d'attention » ou des probabilités. Les grands scores dominent, les petits s'effacent.",
    },
    'transformer': {
      term: 'Transformer',
      def: "Architecture bâtie sur la self-attention, sans récurrence : elle traite toute la séquence en parallèle. Dominante en langage, puissante en audio mais coûteuse en mémoire.",
    },
    'ssm': {
      term: "Modèle à espace d'états (SSM)",
      def: "Une couche qui résume le passé dans un petit état continu mis à jour récurremment, comme un RNN, mais formulée pour s'entraîner en parallèle. Base de Mamba/S4.",
    },
    'tcn': {
      term: 'TCN',
      def: "Réseau convolutif temporel : une pile de convolutions 1D causales et dilatées qui couvre un large contexte temporel avec de petits noyaux. Alternative aux RNN.",
    },
    'wavenet': {
      term: 'WaveNet',
      def: "Modèle génératif de forme d'onde fondé sur des convolutions causales dilatées. Il a popularisé l'idée d'un champ réceptif croissant exponentiellement avec la profondeur.",
    },

    /* ---------- Architectures audio ---------- */
    'encoder-decoder': {
      term: 'Encodeur-décodeur',
      def: "Un schéma en deux temps : l'encodeur compresse l'entrée en une représentation compacte, le décodeur la reconstruit (ou la transforme). Charpente de la plupart des modèles de débruitage/séparation.",
    },
    'bottleneck': {
      term: "Goulot d'étranglement (bottleneck)",
      def: "La couche la plus compacte d'un encodeur-décodeur, au milieu. En forçant l'information à y transiter, on oblige le réseau à n'en garder que l'essentiel.",
    },
    'latent': {
      term: 'Représentation latente',
      def: "L'encodage interne et compact que le réseau se fabrique de l'entrée : non directement lisible, mais porteur de l'information utile à la tâche.",
    },
    'autoregressive': {
      term: 'Autorégressif',
      def: "Un modèle qui génère un échantillon à la fois, chacun conditionné par ceux déjà produits. Fidèle mais lent, car séquentiel par nature.",
    },
    'masking': {
      term: 'Masquage',
      def: "Séparer une source en estimant, pour chaque case temps-fréquence, quelle fraction garder. Le masque multiplie le spectre mêlé pour en extraire la voix (ou la rejeter).",
    },
    'ibm': {
      term: 'Masque binaire idéal (IBM)',
      def: "Un masque parfait à valeurs 0/1 : 1 si la cible domine la case temps-fréquence, 0 sinon. Référence théorique pour évaluer la séparation de sources.",
    },
    'irm': {
      term: 'Masque souple idéal (IRM)',
      def: "Comme l'IBM mais à valeurs continues entre 0 et 1, proportionnelles à la part de la cible dans chaque case. Plus doux, moins d'artefacts.",
    },
    'deep-filtering': {
      term: 'Deep filtering',
      def: "Au lieu d'un seul gain par case temps-fréquence, on applique un petit filtre qui combine plusieurs cases voisines (dans le temps et la fréquence). Plus fin que le masquage simple, il restaure mieux la phase.",
    },
    'dprnn': {
      term: 'DPRNN',
      def: "Réseau récurrent à double chemin : on découpe une longue séquence en blocs, puis on alterne un traitement intra-bloc (local) et inter-blocs (global). Permet aux RNN de gérer de très longues durées.",
    },
    'beamforming': {
      term: 'Formation de voies (beamforming)',
      def: "Avec plusieurs micros, combiner leurs signaux pour amplifier une direction et atténuer les autres — un « micro directionnel » virtuel et orientable.",
    },

    /* ---------- Temps réel & embarqué ---------- */
    'quantization': {
      term: 'Quantification',
      def: "Coder les poids et activations sur moins de bits (par ex. entiers 8 bits au lieu de flottants 32 bits). Le modèle devient plus petit et plus rapide, au prix d'un peu de précision.",
    },
    'int8': {
      term: 'INT8',
      def: "Représentation en entiers sur 8 bits (256 valeurs). Format de quantification le plus courant : ~4× plus compact que le flottant 32 bits, idéal pour NPU et mobile.",
    },
    'npu': {
      term: 'NPU',
      def: "Processeur neuronal : une puce spécialisée dans les multiplications-additions massives des réseaux, bien plus efficace en énergie qu'un CPU pour l'IA embarquée.",
    },
    'latency': {
      term: 'Latence',
      def: "Le délai entre l'entrée d'un son et la sortie traitée. En conversation ou en monitoring, au-delà de ~20–40 ms elle devient perceptible et gênante.",
    },
    'streaming': {
      term: 'Streaming (temps réel)',
      def: "Traiter l'audio au fil de l'eau, par petits blocs, sans attendre la fin de l'enregistrement. Impose des couches causales et une latence maîtrisée.",
    },
    'rtf': {
      term: 'Facteur temps réel (RTF)',
      def: "Temps de calcul divisé par durée d'audio traité. RTF < 1 = plus rapide que le temps réel (viable en direct) ; RTF > 1 = trop lent pour le streaming.",
    },
    'block-size': {
      term: 'Taille de bloc',
      def: "Le nombre d'échantillons traités d'un coup en streaming. Un gros bloc est plus efficace mais ajoute de la latence ; un petit bloc réagit vite mais coûte plus cher par échantillon.",
    },
  });

  /* ---------- Termes contribués par les modules (refonte mobile) ---------- */
  Atlas.defineTerms({
    'filtre-anti-repliement': { term: 'Filtre anti-repliement', def: "Un filtre passe-bas placé juste avant le convertisseur analogique-numérique. Il coupe les fréquences au-dessus de la limite de Nyquist pour les empêcher de se replier en parasites (aliasing) une fois le son échantillonné." },
    'bruit-quantification': { term: 'Bruit de quantification', def: "L'erreur produite quand on arrondit chaque échantillon au niveau le plus proche. Réparti de façon à peu près uniforme, il s'entend comme un léger souffle de fond ; plus on code sur de bits, plus il est faible." },
    'dither': { term: 'Dither', def: "Un bruit aléatoire de très faible niveau ajouté volontairement avant la quantification. Il « décorrèle » l'erreur d'arrondi du signal, ce qui rend le bruit de quantification plus naturel et moins audible aux faibles volumes." },
    'fourier': { term: 'Théorème de Fourier', def: "Résultat mathématique qui affirme que n'importe quel signal, même compliqué, peut être reconstitué en additionnant des ondes simples (des sinusoïdes) de fréquences différentes. C'est le fondement de toute l'analyse spectrale." },
    'sinusoide': { term: 'Sinusoïde', def: "L'onde la plus simple qui soit : une oscillation parfaitement régulière (la courbe du sinus). Un son pur, comme un diapason, en est une ; tout son complexe est une somme de telles ondes." },
    'resolution-frequentielle': { term: 'Résolution fréquentielle', def: "L'écart minimal entre deux fréquences que l'analyse peut séparer, noté Δf = fs/N. Plus on observe le signal longtemps (N grand), plus cet écart est fin et plus on distingue des fréquences proches." },
    'compromis-temps-frequence': { term: 'Compromis temps-fréquence', def: "On ne peut pas être à la fois très précis sur « quelle fréquence » et sur « à quel instant ». Observer plus longtemps affine les fréquences mais brouille le moment exact, et inversement. Ce dilemme est au cœur du spectrogramme." },
    'fenetre-rectangulaire': { term: 'Fenêtre rectangulaire', def: "Découper brutalement une tranche du signal pour l'analyser, sans adoucir les bords. Simple, mais les coupures nettes créent de fausses fréquences (fuites spectrales) : on lui préfère souvent une fenêtre en cloche comme celle de Hann." },
    'fuite-spectrale': { term: 'Fuite spectrale', def: "Quand on analyse une tranche du signal, l'énergie d'une fréquence « bave » sur les cases voisines du spectre à cause des coupures aux bords de la tranche. Une fenêtre douce (Hann) réduit fortement ce phénomène." },
    'gibbs': { term: 'Phénomène de Gibbs', def: "Quand on reconstitue un signal à fronts raides (comme un signal carré) avec un nombre fini d'ondes, des oscillations persistent près des fronts et le dépassement plafonne à environ 9 %, quel que soit le nombre d'ondes ajoutées." },
    'temps-frequence': { term: 'Compromis temps-fréquence (Gabor)', def: "On ne peut pas être précis à la fois dans le temps et en fréquence : une fenêtre d'analyse longue distingue finement les fréquences mais brouille les instants, une fenêtre courte fait l'inverse. Leur produit (Δt × Δf) reste constant, c'est l'incertitude de Heisenberg-Gabor." },
    'resolution-temporelle': { term: 'Résolution temporelle (Δt)', def: "La finesse avec laquelle on situe un événement dans le temps. En STFT elle vaut Δt = N/sr : plus la fenêtre est courte (petit N), plus on date précisément les attaques et transitoires." },
    'cola': { term: 'Condition COLA', def: "Constant OverLap-Add : quand les fenêtres successives, additionnées avec leur recouvrement, redonnent une somme constante. C'est la condition qui permet de reconstruire le signal sans déformation après traitement." },
    'overlap-add': { term: 'Overlap-add', def: "Méthode pour réassembler un signal traité fenêtre par fenêtre : on additionne les morceaux en les faisant se chevaucher. Combinée à la condition COLA, elle reconstitue le son d'origine." },
    'reconstruction-parfaite': { term: 'Reconstruction parfaite', def: "Propriété d'une analyse-synthèse qui, sans aucune modification entre les deux, restitue exactement le signal de départ. C'est la garantie de base avant tout traitement fréquentiel (débruitage, séparation)." },
    'istft': { term: 'STFT inverse (ISTFT)', def: "L'opération qui repasse du spectrogramme au signal temporel : on calcule la FFT inverse de chaque trame puis on les recombine par overlap-add. C'est elle qui « rejoue » le son après traitement." },
    'spectral-leakage': { term: 'Fuites spectrales (lobes latéraux)', def: "Quand on découpe brutalement le signal (fenêtre rectangulaire), les bords créent des discontinuités qui étalent l'énergie d'une fréquence sur ses voisines (lobes latéraux). La fenêtre de Hann adoucit les bords pour limiter ces fuites." },
    'algorithmic-latency': { term: 'Latence algorithmique', def: "Le retard incompressible dû au fait qu'on doit attendre d'avoir reçu toute une fenêtre (N échantillons, soit N/sr secondes) avant de pouvoir l'analyser. C'est pourquoi le temps réel utilise des fenêtres courtes de 10–20 ms." },
    'cochlee': { term: 'Cochlée', def: "L'organe en spirale de l'oreille interne qui transforme les vibrations sonores en signaux nerveux. Le long de sa membrane, chaque endroit répond à une fréquence précise : les graves d'un côté, les aigus de l'autre." },
    'erb': { term: 'ERB (Equivalent Rectangular Bandwidth)', def: "Une échelle de fréquences qui modélise la largeur des « filtres » de l'oreille : très étroits dans les graves, de plus en plus larges vers les aigus. Sert, comme l'échelle mel, à découper le son en bandes proches de notre perception." },
    'gain-par-bande': { term: 'Gain par bande', def: "Un facteur d'amplification (ou d'atténuation) appliqué à toute une bande de fréquences d'un coup, plutôt qu'à chaque fréquence séparément. Un réseau de débruitage prédit souvent quelques dizaines de gains de bande au lieu de centaines de valeurs fines." },
    'enveloppe-spectrale': { term: 'Enveloppe spectrale', def: "La forme générale du spectre : où se trouvent les zones de fréquences fortes et faibles, sans le détail fin. C'est elle qui porte l'essentiel du timbre et de l'intelligibilité d'un son." },
    'prior-perceptuel': { term: 'Prior perceptuel', def: "Une connaissance a priori, inspirée de l'audition humaine, qu'on intègre dans un modèle pour le guider. Découper le son sur une échelle mel ou ERB est un tel prior : il oriente le réseau vers ce que l'oreille perçoit vraiment." },
    'full-band': { term: 'Full-band', def: "Se dit d'un traitement audio qui couvre toute la bande passante audible, typiquement échantillonnée à 48 kHz (fréquences jusqu'à 24 kHz), par opposition aux systèmes limités à la voix (16 kHz, jusqu'à 8 kHz)." },
    'snr': { term: 'Rapport signal/bruit (SNR)', def: "La proportion entre l'énergie du son utile (la voix) et celle du bruit, exprimée en décibels. Plus le SNR est élevé, plus la voix ressort nettement ; un SNR négatif signifie que le bruit est plus fort que la voix." },
    'additivite': { term: 'Cohérence additive', def: "Propriété d'une séparation où les sorties estimées, une fois ré-additionnées, redonnent exactement le mélange d'entrée. Si on extrait la voix et le reste avec des masques complémentaires (qui somment à 1), aucune énergie n'est perdue ni inventée dans le remix." },
    'vanishing-gradient': { term: "Gradient qui s'évanouit", def: "Pendant l'apprentissage d'un réseau récurrent, le signal d'erreur est multiplié par un petit facteur à chaque pas de temps remonté. Après quelques pas il devient minuscule, et le réseau n'arrive plus à apprendre les dépendances lointaines. C'est le problème que les portes du GRU et du LSTM corrigent." },
    'gate': { term: 'Porte (gate)', def: "Dans un GRU ou un LSTM, une petite valeur entre 0 et 1 qui dose la quantité d'information laissée passer : 0 = on bloque, 1 = on laisse tout. Les portes décident quoi garder, oublier ou réécrire dans la mémoire à chaque pas." },
    'cell-state': { term: 'État de cellule (C)', def: "La mémoire « longue durée » d'un LSTM, séparée de l'état caché. Elle traverse le réseau comme un tapis roulant, modifiée seulement par des additions et la porte d'oubli, ce qui laisse l'information survivre sur de longues durées." },
    'rnnoise': { term: 'RNNoise', def: "Un débruiteur de voix temps réel, léger et open source, bâti autour d'un petit réseau récurrent (GRU). Conçu pour tourner en direct sur un appareil modeste pendant un appel." },
    'vad': { term: "VAD (détection d'activité vocale)", def: "Voice Activity Detection : repérer en continu les instants où quelqu'un parle dans un flux audio, par opposition au silence ou au bruit de fond. Brique de base des assistants vocaux et de la visioconférence." },
    'keyword-spotting': { term: 'Détection de mot-clé (keyword spotting)', def: "Faire tourner en permanence un tout petit modèle qui écoute un mot déclencheur précis (« Dis Siri », « OK Google ») pour réveiller un assistant, sans envoyer tout l'audio ailleurs. Doit être minuscule et économe en énergie." },
    'fp16': { term: 'FP16 (flottant 16 bits)', def: "Une façon de coder les nombres d'un réseau sur 16 bits au lieu de 32, soit deux fois moins de mémoire. Courant pour réduire la taille d'un modèle et de son état avec une perte de précision négligeable." },
    'attention-score': { term: "Score d'attention", def: "La mesure de pertinence entre deux positions d'une séquence, calculée par le produit scalaire entre une query et une key. Plus le score est élevé, plus les deux instants se « ressemblent » et plus l'un influencera l'autre." },
    'temperature': { term: 'Température (τ)', def: "Un facteur par lequel on divise les scores avant le softmax. Une température basse rend la distribution piquée (l'attention se concentre sur une ou deux positions) ; une température haute l'aplatit (l'attention se répartit)." },
    'multi-head': { term: "Têtes d'attention multiples", def: "On découpe les dimensions en plusieurs sous-espaces, et chaque « tête » calcule sa propre attention sur sa part. Les têtes se spécialisent (graves, transitoires…) et leurs sorties sont ensuite recombinées." },
    'masking-causal': { term: 'Masque causal', def: "En attention, interdit à chaque position de regarder le futur : on annule la moitié supérieure de la matrice d'attention. Indispensable en temps réel, où les instants à venir n'existent pas encore." },
    'kv-cache': { term: 'Cache K/V', def: "En attention, les clés (keys) et valeurs (values) déjà calculées sont conservées en mémoire pour ne pas les recalculer à chaque nouvelle position. Contrairement à l'état compact d'un RNN, ce cache grandit avec la durée traitée." },
    'cpu-fallback': { term: 'Repli sur CPU (fallback)', def: "Quand une puce accélératrice (NPU, GPU) ne sait pas exécuter une opération, le calcul retombe sur le processeur classique, beaucoup plus lent. Le softmax et les multiplications dynamiques de l'attention déclenchent souvent ce repli sur l'embarqué." },
    'local-attention': { term: 'Attention locale (fenêtrée)', def: "Une variante où chaque position ne regarde qu'une fenêtre voisine au lieu de toute la séquence. On perd le contexte lointain mais le coût redevient gérable en temps réel." },
    'linear-attention': { term: 'Attention linéaire', def: "Une reformulation de l'attention dont le coût croît proportionnellement à la longueur (et non au carré). Elle approxime le softmax pour éviter de construire la matrice T×T complète, au prix d'un peu de précision." },
    'mamba': { term: 'Mamba', def: "Une variante récente de modèle à espace d'états (SSM) qui rend ses paramètres dépendants de l'entrée : une « porte » apprise décide, à chaque instant, ce qui mérite d'entrer dans la mémoire. Efficace et rapide en streaming, elle rivalise avec les Transformers sur de longues séquences." },
    'selectivity': { term: 'Sélectivité', def: "Capacité d'une couche à filtrer l'information entrante selon son contenu, au lieu de tout intégrer de la même façon. Dans Mamba, c'est ce qui permet d'ignorer un bruit de fond et de ne mémoriser que les instants utiles." },
    'continuous-time': { term: 'Temps continu', def: "Une formulation où l'évolution du système est décrite par une équation différentielle (un taux de variation à chaque instant), avant d'être « discrétisée » en pas de temps réguliers pour tourner sur un ordinateur." },
    'discretization': { term: 'Discrétisation', def: "Transformer une équation à temps continu en une règle de mise à jour pas à pas, applicable à des échantillons. C'est l'étape qui rend un système continu exécutable sur une séquence numérique." },
    'time-constant': { term: 'Constante de temps (τ)', def: "La durée caractéristique pendant laquelle une mémoire ou une réponse s'estompe. Une petite τ oublie vite (réactif), une grande τ retient longtemps (mémoire lente)." },
    'eigenvalue': { term: 'Valeur propre (λ)', def: "Un nombre qui caractérise le comportement d'une matrice. Ici, chaque λ négatif fixe la vitesse à laquelle une composante de l'état décroît : son inverse, 1/|λ|, donne la constante de temps de cette mémoire." },
    'multiscale-memory': { term: 'Mémoire multi-échelle', def: "Combiner plusieurs mémoires de durées très différentes — quelques millisecondes à plusieurs secondes — pour capter à la fois les détails brefs (phonèmes) et les structures longues (prosodie, contexte)." },
    'parallel-scan': { term: 'Scan parallèle', def: "Une technique de calcul qui évalue une récurrence (un calcul pas à pas) en parallèle plutôt que séquentiellement, en exploitant son associativité. C'est ce qui permet d'entraîner un SSM aussi vite qu'un modèle non récurrent." },
    'phoneme': { term: 'Phonème', def: "La plus petite unité sonore qui distingue deux mots dans une langue, comme le « p » et le « b » de « pain » et « bain ». Une langue en compte quelques dizaines." },
    'crn': { term: 'CRN (réseau convolutif-récurrent)', def: "Une architecture qui combine des couches de convolution (pour analyser la structure locale du spectrogramme) et des couches récurrentes (pour suivre l'évolution dans le temps). Très courante pour le débruitage de la voix, comme DeepFilterNet." },
    'skip-connection': { term: 'Skip connection (connexion résiduelle)', def: "Un raccourci qui copie directement les informations d'une couche vers une couche plus profonde, en court-circuitant les couches intermédiaires. Cela préserve les détails fins et facilite l'entraînement des réseaux profonds." },
    'source-separation': { term: 'Séparation de sources', def: "Isoler des signaux mêlés dans un même enregistrement : par exemple extraire la voix d'un locuteur d'un fond bruyant, ou séparer plusieurs personnes qui parlent en même temps." },
    'tasnet': { term: 'TasNet', def: "Famille de modèles de séparation de sources qui travaillent directement sur la forme d'onde (« bout en bout »), sans passer par un spectrogramme. Un encodeur apprend sa propre représentation du son, séparée puis reconstruite par un décodeur." },
    'big-o': { term: 'Notation O(...)', def: "Une façon d'exprimer comment le coût d'un calcul grandit avec la taille de l'entrée. O(T) signifie « proportionnel à T » ; O(√T), « proportionnel à la racine de T », donc beaucoup moins cher quand T est grand." },
    'dual-path': { term: 'Double chemin (dual-path)', def: "Idée de découper une longue séquence en blocs disposés en grille 2D, puis d'alterner un traitement à l'intérieur de chaque bloc (local) et entre les blocs (global). Deux « chemins » courts remplacent un seul chemin très long." },
    'chunk': { term: 'Chunk (bloc)', def: "Un petit morceau contigu d'une longue séquence. Découper le signal en chunks permet de traiter chaque morceau localement avant de relier les morceaux entre eux." },
    'sepformer': { term: 'SepFormer', def: "Modèle de séparation de sources qui reprend le pliage en double chemin de DPRNN mais remplace les réseaux récurrents par de l'attention (Transformer). État de l'art en séparation de parole." },
    'complex-weight': { term: 'Poids complexe', def: "Un coefficient de filtre qui n'est pas un simple nombre, mais une amplitude ET un angle (une rotation de phase). Multiplier un signal par un poids complexe permet à la fois de le réduire en volume et de le décaler dans le temps — ce qu'un gain réel seul ne peut pas faire." },
    'deepfilternet': { term: 'DeepFilterNet', def: "Un modèle de débruitage de la parole en temps réel, léger, conçu pour tourner sur mobile. Il combine des gains grossiers par bandes de fréquences sur tout le spectre et un petit filtre complexe « deep filtering » sur les basses fréquences, là où vivent la voix et ses harmoniques." },
    'mac': { term: 'MAC (multiplication-accumulation)', def: "L'opération de base des réseaux de neurones : multiplier deux nombres et ajouter le résultat à un total. On compte les MACs par frame ou par seconde pour estimer le coût de calcul. Une multiplication de nombres complexes vaut 4 MACs réelles." },
    'modele-generatif': { term: 'Modèle génératif', def: "Un modèle qui crée de l'information nouvelle au lieu de seulement filtrer l'existant : il peut reconstruire des détails absents du signal d'entrée (fréquences perdues, échantillons coupés). Opposé à un modèle discriminatif, qui se contente de trier ou d'atténuer ce qui est déjà là." },
    'hallucination': { term: 'Hallucination', def: "Quand un modèle génératif invente un contenu plausible mais faux — par exemple un son de parole crédible qui ne correspond pas à ce qui a réellement été dit. C'est le risque inhérent à toute génération, à l'opposé du simple filtrage qui ne peut rien inventer." },
    'nfe': { term: 'NFE (Number of Function Evaluations)', def: "Le nombre de fois qu'il faut faire tourner le réseau (des « forwards ») pour produire une seule sortie. Plus le NFE est élevé, plus la génération est lente : c'est la mesure clé du coût des modèles génératifs en temps réel." },
    'gan': { term: 'GAN (réseau antagoniste)', def: "Un type de modèle génératif où deux réseaux s'affrontent : un générateur fabrique des données et un discriminateur tente de distinguer le vrai du faux. Le générateur produit une sortie en un seul forward (NFE = 1), mais cet entraînement par duel est réputé instable." },
    'diffusion': { term: 'Diffusion', def: "Un modèle génératif qui part de bruit pur et le débruite progressivement, étape par étape, jusqu'à obtenir un son propre. Très réaliste, mais coûteux : il faut souvent des dizaines à des centaines d'étapes (forwards) par sortie." },
    'flow-matching': { term: 'Flow matching', def: "Une famille récente de modèles génératifs qui apprend directement le « chemin » menant du bruit aux données sous forme d'un champ de vitesses. Les trajectoires étant presque droites, il faut très peu d'étapes (2 à 8) — un bon compromis vitesse/qualité, avec un entraînement stable." },
    'velocity-field': { term: 'Champ de vitesses', def: "Une fonction qui indique, en chaque point de l'espace, dans quelle direction et à quelle vitesse se déplacer. En flow matching, le modèle apprend ce champ : suivre les flèches depuis un point de bruit conduit jusqu'aux données propres." },
    'ode': { term: 'ODE (équation différentielle ordinaire)', def: "Une équation qui décrit comment une quantité évolue continûment quand on suit une vitesse donnée. La « résoudre » revient à avancer pas à pas le long de la trajectoire ; en flow matching, peu de pas suffisent car le chemin est presque droit." },
    'adversarial-training': { term: 'Entraînement adversarial', def: "Apprentissage par compétition entre deux réseaux : l'un essaie de tromper l'autre, qui essaie de ne pas l'être. Cette rivalité pousse à des résultats très réalistes, mais l'équilibre est fragile et difficile à stabiliser." },
    'mode-collapse': { term: 'Mode collapse', def: "Une panne classique des GAN : le générateur se met à produire toujours les mêmes quelques sorties au lieu de couvrir toute la variété des données possibles. Le modèle « triche » en exploitant une faille du discriminateur." },
    'one-step-distillation': { term: 'Distillation one-step', def: "Entraîner un modèle rapide (un seul forward, NFE = 1) à imiter un modèle lent mais excellent (diffusion ou flow à nombreuses étapes). On « distille » la qualité du gros modèle dans un petit, condition d'un usage temps réel." },
    'consistency-models': { term: 'Consistency models', def: "Une approche qui apprend à sauter en une seule étape d'un point bruité jusqu'au résultat propre, en imposant que tous les points d'une même trajectoire mènent à la même destination. Objectif : la qualité de la diffusion avec un coût proche de NFE = 1." },
    'bwe': { term: 'Extension de bande (BWE)', def: "Recréer les hautes fréquences manquantes d'un son qui en a été privé (téléphone, vieil enregistrement, compression). Le modèle génère un contenu plausible au-dessus de la bande disponible pour rendre la voix plus claire et naturelle." },
    'declipping': { term: 'Déclippage', def: "Réparer un signal « écrêté » : quand le son a été enregistré trop fort, ses crêtes ont été coupées (aplaties). Le déclippage reconstruit la forme d'onde originale au-dessus de ce plafond." },
    'neural-codec': { term: 'Codec neuronal', def: "Un compresseur audio fondé sur un réseau de neurones : il encode le son en très peu de données, puis le régénère à la lecture. À très bas débit, il reconstruit (et donc invente en partie) des détails plausibles." },
    'transport-distribution': { term: 'Transport de distribution', def: "Déplacer un nuage de points (par ex. du bruit aléatoire) pour qu'il épouse la forme d'un autre nuage (les vrais sons). Les modèles génératifs sont essentiellement différentes façons d'effectuer ce transport du bruit vers les données." },
    'lip-sync': { term: 'Synchronisation labiale (lip-sync)', def: "L'accord entre le son et le mouvement des lèvres à l'image. Si le son traité arrive trop tard (vers 100 ms de retard), le décalage devient visible et gênant à la télévision ou en visioconférence." },
    'comb-filter': { term: 'Effet de peigne (filtrage en peigne)', def: "Quand un son se superpose à une copie légèrement retardée de lui-même, certaines fréquences s'annulent et d'autres se renforcent, creusant le spectre comme les dents d'un peigne. C'est ce qui rend le son métallique quand le chemin traité d'une aide auditive arrive en retard sur le son direct." },
    'quant-grid': { term: 'Grille de quantification', def: "L'ensemble fini des valeurs autorisées une fois quantifié. Au lieu de tout réel possible, un poids doit tomber sur l'un de ces niveaux ; plus la grille est dense, plus la perte est faible." },
    'quant-step': { term: 'Pas de quantification', def: "L'écart entre deux niveaux voisins de la grille. Plus le pas est petit, plus la valeur d'origine est fidèlement représentée ; l'erreur maximale vaut la moitié d'un pas." },
    'sqnr': { term: 'SQNR', def: "Rapport signal sur bruit de quantification, en décibels : il compare l'énergie des poids à celle de l'erreur introduite en les arrondissant. Plus il est élevé, plus la quantification est fidèle (chaque bit gagné ajoute ~6 dB)." },
    'fp32': { term: 'FP32', def: "Nombre à virgule flottante sur 32 bits, le format de calcul par défaut à l'entraînement. Précis mais quatre fois plus lourd qu'un entier 8 bits à stocker et à déplacer." },
    'int4': { term: 'INT4', def: "Quantification en entiers sur 4 bits, soit seulement 16 niveaux. Deux fois plus compact qu'INT8, mais la grille est si grossière que la perte de précision devient difficile à maîtriser." },
    'outlier': { term: 'Outlier (valeur aberrante)', def: "Un poids isolé très éloigné de la masse des autres. En quantification uniforme, quelques outliers étirent la plage [min, max] et grossissent le pas, dégradant la précision de tous les autres poids." },
    'per-channel': { term: 'Quantification per-channel', def: "Donner à chaque canal sa propre plage min/max et sa propre grille, au lieu d'une seule pour tout le tenseur. Les canaux sans valeur aberrante gardent ainsi un pas fin et une meilleure précision." },
    'ptq': { term: 'PTQ (quantification post-entraînement)', def: "Quantifier un modèle déjà entraîné, sans le réentraîner. Rapide et courant, mais nécessite une phase de calibration pour mesurer les vraies plages des poids et activations." },
    'calibration': { term: 'Calibration', def: "Étape qui mesure les plages réelles de valeurs (min/max, par canal) sur des données représentatives, afin de placer correctement la grille de quantification. Indispensable à une bonne quantification post-entraînement." },
    'cache': { term: 'Cache (mémoire on-chip)', def: "Une petite mémoire très rapide intégrée à la puce. Si les poids du modèle y tiennent, ils sont chargés une fois ; sinon il faut les relire sans cesse depuis la mémoire externe, bien plus lente." },
    'dram': { term: 'DRAM', def: "La mémoire principale externe à la puce (RAM), grande mais lente d'accès. Y relire les poids à chaque inférence coûte du temps et de l'énergie : c'est souvent le vrai goulot de l'IA embarquée." },
    'memory-bandwidth': { term: 'Bande passante mémoire', def: "La quantité d'octets que la puce peut lire ou écrire par seconde entre le calcul et la mémoire. Quand un modèle doit relire ses poids à chaque trame, c'est elle, et non la puissance de calcul, qui plafonne la vitesse." },
  });
})();
