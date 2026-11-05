// Service worker: offline app shell + runtime cache for ONNX Runtime, fonts and the model file.
const VERSION = 'dsd-v1.0.0';
const SHELL = ['./', 'index.html', 'manifest.webmanifest', 'icon.svg', 'icon-192.png', 'icon-512.png'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION && k !== 'dsd-runtime').map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', (e) => {
  const req = e.request; if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const runtime = url.hostname === 'cdn.jsdelivr.net' || url.hostname.endsWith('fonts.googleapis.com') || url.hostname.endsWith('fonts.gstatic.com') || url.pathname.endsWith('.onnx');
  if (runtime) { // cache first
    e.respondWith(caches.open('dsd-runtime').then(async (c) => { const hit = await c.match(req); if (hit) return hit; const res = await fetch(req); if (res.ok || res.type === 'opaque') c.put(req, res.clone()); return res; }));
    return;
  }
  if (url.origin === location.origin) { // network first for the app so updates arrive, cache fallback offline
    e.respondWith(fetch(req).then((res) => { if (res.ok) caches.open(VERSION).then((c) => c.put(req, res.clone())); return res; }).catch(() => caches.match(req).then((r) => r || caches.match('index.html'))));
  }
});
