/* ============ Audio AI Atlas — shell applicatif ============ */
(function () {
  'use strict';

  const CATS = {
    signal: { label: 'Signal & DSP', color: '#ff5c49' },
    layers: { label: 'Couches neuronales', color: '#60a5fa' },
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

  /* ---------- Glossaire : popover de définition au tap ---------- */
  let popEl = null, popBackdrop = null, popScrollY = 0;
  function hideTermPop() {
    if (popEl) { popEl.remove(); popEl = null; }
    if (popBackdrop) { popBackdrop.remove(); popBackdrop = null; }
    document.removeEventListener('keydown', onPopKey);
    window.removeEventListener('resize', hideTermPop);
    window.removeEventListener('scroll', onPopScroll, true);
  }
  function onPopKey(e) { if (e.key === 'Escape') hideTermPop(); }
  // desktop : ferme quand l'utilisateur fait réellement défiler (pas sur un scroll fantôme post-ouverture)
  function onPopScroll() {
    if (window.innerWidth > 640 && Math.abs((window.scrollY || 0) - popScrollY) > 4) hideTermPop();
  }

  function showTermPop(termId, anchor) {
    hideTermPop();
    const entry = Atlas.glossary[termId];
    if (!entry) { console.warn('Terme inconnu :', termId); return; }

    popBackdrop = el('div', 'term-pop-backdrop');
    popBackdrop.addEventListener('click', hideTermPop);
    document.body.appendChild(popBackdrop);

    popEl = el('div', 'term-pop');
    popEl.setAttribute('role', 'dialog');
    popEl.setAttribute('aria-label', 'Définition : ' + entry.term);
    const close = el('button', 'tp-close', '×');
    close.setAttribute('aria-label', 'Fermer');
    close.addEventListener('click', hideTermPop);
    popEl.appendChild(close);
    popEl.appendChild(el('div', 'tp-term', entry.term));
    popEl.appendChild(el('p', 'tp-def', entry.def));
    const more = el('a', 'tp-more', 'Voir dans le glossaire →');
    more.href = '#/glossary?t=' + encodeURIComponent(termId);
    more.addEventListener('click', hideTermPop);
    popEl.appendChild(more);
    document.body.appendChild(popEl);

    if (window.innerWidth > 640 && anchor) { // desktop : ancré au terme (mobile : feuille en bas, géré en CSS)
      const r = anchor.getBoundingClientRect();
      const pw = popEl.offsetWidth, ph = popEl.offsetHeight;
      let left = r.left + r.width / 2 - pw / 2;
      left = Math.max(8, Math.min(left, window.innerWidth - pw - 8));
      let top = r.bottom + 8;
      if (top + ph > window.innerHeight - 8) top = Math.max(8, r.top - ph - 8);
      popEl.style.left = left + 'px';
      popEl.style.top = top + 'px';
    }
    popScrollY = window.scrollY || 0;
    document.addEventListener('keydown', onPopKey);
    window.addEventListener('resize', hideTermPop);
    window.addEventListener('scroll', onPopScroll, true);
  }

  /* Rend cliquables les <… class="term" data-term="id"> d'un bloc explain. */
  function wireTerms(container) {
    const terms = container.querySelectorAll('.term[data-term]');
    if (!terms.length) return;
    const hint = el('p', 'term-hint',
      'Astuce — touchez les <span class="term-hint-dot">termes soulignés</span> pour leur définition.');
    const h2 = container.querySelector('h2');
    if (h2 && h2.nextSibling) container.insertBefore(hint, h2.nextSibling);
    else container.insertBefore(hint, container.firstChild);
    terms.forEach((t) => {
      const id = t.getAttribute('data-term');
      if (!Atlas.glossary[id]) return; // terme non défini : on laisse en texte simple
      t.setAttribute('tabindex', '0');
      t.setAttribute('role', 'button');
      const open = (e) => { e.preventDefault(); e.stopPropagation(); showTermPop(id, t); };
      t.addEventListener('click', open);
      t.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') open(e); });
    });
  }

  /* ---------- Page Glossaire ---------- */
  function renderGlossary() {
    destroyCurrent();
    hideTermPop();
    document.title = 'Glossaire — Audio AI Atlas';
    app.innerHTML = '';
    window.scrollTo(0, 0);

    const bar = el('div', 'topbar');
    const back = el('button', 'back-btn', '← Atlas');
    back.addEventListener('click', () => { location.hash = ''; });
    bar.appendChild(back);
    bar.appendChild(el('h2', null, 'Glossaire'));
    app.appendChild(bar);

    const wrap = el('div', 'gloss');
    const intro = el('p', null,
      'Tous les termes employés dans l’atlas, définis simplement — sans rien présupposer. ' +
      'Cherchez, ou parcourez la liste.');
    intro.style.cssText = 'color:var(--dim);font-size:14.5px;margin:14px 4px 0;line-height:1.55';
    wrap.appendChild(intro);

    const search = document.createElement('input');
    search.className = 'gloss-search';
    search.type = 'search';
    search.placeholder = 'Rechercher un terme ou une notion…';
    search.setAttribute('aria-label', 'Rechercher dans le glossaire');
    wrap.appendChild(search);

    const count = el('div', 'gloss-count');
    wrap.appendChild(count);

    const list = el('dl', 'gloss-list');
    wrap.appendChild(list);

    const back2 = el('a', 'gloss-foot-link', '← Retour à l’atlas');
    back2.href = '#';
    wrap.appendChild(back2);

    app.appendChild(wrap);

    const entries = Object.keys(Atlas.glossary)
      .map((id) => ({ id, term: Atlas.glossary[id].term, def: Atlas.glossary[id].def }))
      .sort((a, b) => a.term.localeCompare(b.term, 'fr', { sensitivity: 'base' }));

    function build(q) {
      list.innerHTML = '';
      const qq = (q || '').trim().toLowerCase();
      let n = 0;
      for (const e of entries) {
        if (qq && !(e.term.toLowerCase().includes(qq) || e.def.toLowerCase().includes(qq))) continue;
        const item = el('div', 'gloss-term');
        item.id = 'g-' + e.id;
        item.appendChild(el('dt', null, e.term));
        item.appendChild(el('dd', null, e.def));
        list.appendChild(item);
        n++;
      }
      count.textContent = qq
        ? n + ' terme' + (n > 1 ? 's' : '') + ' trouvé' + (n > 1 ? 's' : '')
        : n + ' termes';
      if (n === 0) list.appendChild(el('div', 'gloss-empty', 'Aucun terme ne correspond à votre recherche.'));
    }
    search.addEventListener('input', () => build(search.value));
    build('');

    // Ciblage d'un terme depuis « Voir dans le glossaire → »
    const tm = location.hash.match(/[?&]t=([^&]+)/);
    if (tm) {
      const node = document.getElementById('g-' + decodeURIComponent(tm[1]));
      if (node) {
        node.scrollIntoView({ block: 'center' });
        node.style.transition = 'background 0.3s';
        node.style.background = 'color-mix(in srgb, var(--accent) 14%, transparent)';
        setTimeout(() => { node.style.background = 'transparent'; }, 1300);
      }
    }
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
    mkChip('all', 'Tout', '#cfcabf');
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

    const foot = el('div', 'home-foot',
      '<strong>📖 Nouveau dans le domaine ?</strong> Chaque explication a ses termes techniques ' +
      'cliquables, et le <a href="#/glossary">glossaire complet</a> définit tout sans rien présupposer.' +
      '<br><br>' +
      '<strong>📱 Sur iPhone :</strong> ouvre cette page dans Safari → bouton Partager → ' +
      '« Sur l’écran d’accueil ». L’app fonctionne ensuite hors-ligne, en plein écran. ' +
      'Gratuit, sans compte, sans tracking — du canvas et des maths.');
    app.appendChild(foot);
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
    const glossBtn = el('button', 'gloss-btn', '📖');
    glossBtn.title = 'Glossaire';
    glossBtn.setAttribute('aria-label', 'Ouvrir le glossaire');
    glossBtn.addEventListener('click', () => { location.hash = '#/glossary'; });
    tools.appendChild(glossBtn);
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
      wireTerms(ex);
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
    hideTermPop();
    if (/^#\/glossary/.test(location.hash)) { renderGlossary(); return; }
    const m = location.hash.match(/^#\/m\/([\w-]+)/);
    if (m && Atlas.byId[m[1]]) renderModule(m[1]);
    else renderHome();
  }
  window.addEventListener('hashchange', route);
  route();
})();
