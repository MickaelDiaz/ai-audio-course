/* Audio AI Atlas — service worker (cache-first, offline complet) */
const VERSION = 'atlas-v1';
const ASSETS = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './manifest.webmanifest',
  './core/viz-core.js',
  './modules/sampling.js',
  './modules/fft.js',
  './modules/stft.js',
  './modules/filterbanks.js',
  './modules/masking.js',
  './modules/conv1d.js',
  './modules/conv2d.js',
  './modules/rnn.js',
  './modules/attention.js',
  './modules/ssm.js',
  './modules/encdec.js',
  './modules/dprnn.js',
  './modules/deepfilter.js',
  './modules/generative.js',
  './modules/pipeline.js',
  './modules/latency.js',
  './modules/quantization.js',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/apple-touch-icon.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then((hit) => {
      if (hit) return hit;
      return fetch(e.request).then((res) => {
        if (res.ok && new URL(e.request.url).origin === location.origin) {
          const clone = res.clone();
          caches.open(VERSION).then((c) => c.put(e.request, clone));
        }
        return res;
      });
    })
  );
});
