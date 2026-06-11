/* ============ Audio AI Atlas — shell applicatif ============ */
(function () {
  'use strict';

  const CATS = {
    signal: { label: 'Signal & DSP', color: '#60a5fa' },
    layers: { label: 'Couches neuronales', color: '#2dd4bf' },
    archi: { label: 'Architectures', color: '#a78bfa' },
    system: { label: 'Temps réel & embarqué', color: '#fbbf24' },
  };

  const app = document.getElementById('app');
  let currentHandle = null;
  let filter = 'all';

  function destroyCurrent() {
    if (currentHandle) { currentHandle.destroy(); currentHandle = null; }
  }

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  /* ---------- Accueil ---------- */
  function renderHome() {
    destroyCurrent();
    document.title = 'Audio AI Atlas';
    app.innerHTML = '';

    const hero = el('div', 'hero');
    hero.appendChild(el('h1', null, 'Audio AI Atlas'));
    hero.appendChild(el('p', null,
      "Les architectures d'IA audio, en mouvement — du signal brut au réseau embarqué. " +
      'Chaque module est interactif : règle, observe, comprends.'));
    app.appendChild(hero);

    const chips = el('div', 'chips');
    const mkChip = (key, label, color) => {
      const b = el('button', 'chip-btn' + (filter === key ? ' on' : ''), label);
      b.style.setProperty('--chipc', color || 'var(--blue)');
      b.addEventListener('click', () => { filter = key; renderHome(); });
      chips.appendChild(b);
    };
    mkChip('all', 'Tout', '#94a3b8');
    for (const [k, c] of Object.entries(CATS)) mkChip(k, c.label, c.color);
    app.appendChild(chips);

    const grid = el('div', 'grid');
    for (const def of Atlas.modules) {
      if (filter !== 'all' && def.category !== filter) continue;
      const cat = CATS[def.category] || CATS.signal;
      const card = el('div', 'card');
      card.style.setProperty('--cc', cat.color);
      const top = el('div', 'card-top');
      top.appendChild(el('div', 'card-icon', def.icon || '◆'));
      const tt = el('div');
      tt.appendChild(el('div', 'cat-tag', cat.label));
      tt.appendChild(el('h3', null, def.title));
      top.appendChild(tt);
      card.appendChild(top);
      card.appendChild(el('p', null, def.summary || ''));
      card.addEventListener('click', () => { location.hash = '#/m/' + def.id; });
      grid.appendChild(card);
    }
    app.appendChild(grid);

    app.appendChild(el('div', 'home-foot',
      '<strong>📱 Sur iPhone :</strong> ouvre cette page dans Safari → bouton Partager → ' +
      '« Sur l’écran d’accueil ». L’app fonctionne ensuite hors-ligne, en plein écran. ' +
      'Gratuit, sans compte, sans tracking — du canvas et des maths.'));
  }

  /* ---------- Vue module ---------- */
  function renderModule(id) {
    destroyCurrent();
    const def = Atlas.byId[id];
    if (!def) { location.hash = ''; return; }
    const cat = CATS[def.category] || CATS.signal;
    document.title = def.title + ' — Audio AI Atlas';
    app.innerHTML = '';
    window.scrollTo(0, 0);

    const bar = el('div', 'topbar');
    bar.style.setProperty('--cc', cat.color);
    const back = el('button', 'back-btn', '← Atlas');
    back.addEventListener('click', () => { location.hash = ''; });
    bar.appendChild(back);
    bar.appendChild(el('h2', null, def.title));
    bar.appendChild(el('span', 'cat-tag', cat.label));

    const tools = el('div', 'toolbar');
    const pauseBtn = el('button', 'tool-btn', '⏸');
    const speedSel = document.createElement('select');
    for (const s of [0.25, 0.5, 1, 1.5, 2]) {
      const o = document.createElement('option');
      o.value = s; o.textContent = s + '×';
      if (s === 1) o.selected = true;
      speedSel.appendChild(o);
    }
    tools.append(pauseBtn, speedSel);
    bar.appendChild(tools);
    app.appendChild(bar);

    const stageWrap = el('div', 'stage-wrap');
    app.appendChild(stageWrap);
    const controls = el('div', 'controls');
    app.appendChild(controls);

    if (def.explain) {
      const ex = el('section', 'explain');
      ex.appendChild(el('h2', null, 'Comprendre'));
      ex.insertAdjacentHTML('beforeend', def.explain);
      app.appendChild(ex);
    }

    // navigation précédent / suivant
    const idx = Atlas.modules.indexOf(def);
    const nav = el('div', 'mod-nav');
    const prev = Atlas.modules[idx - 1], next = Atlas.modules[idx + 1];
    if (prev) nav.appendChild(el('a', 'prev', `<span>← Précédent</span>${prev.title}`)).href = '#/m/' + prev.id;
    if (next) nav.appendChild(el('a', 'next', `<span>Suivant →</span>${next.title}`)).href = '#/m/' + next.id;
    app.appendChild(nav);

    currentHandle = Atlas.mount(def, stageWrap, controls);

    pauseBtn.addEventListener('click', () => {
      const st = currentHandle.stage;
      st.paused = !st.paused;
      pauseBtn.textContent = st.paused ? '▶' : '⏸';
    });
    speedSel.addEventListener('change', () => {
      currentHandle.stage.speed = +speedSel.value;
    });
  }

  /* ---------- Routage ---------- */
  function route() {
    const m = location.hash.match(/^#\/m\/([\w-]+)/);
    if (m && Atlas.byId[m[1]]) renderModule(m[1]);
    else renderHome();
  }
  window.addEventListener('hashchange', route);
  route();
})();
