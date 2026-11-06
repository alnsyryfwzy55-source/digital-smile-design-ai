// Offline support: app files network-first (updates arrive), AI tools and fonts cache-first (large, versioned)
const VERSION = 'dsd2-20261106171532';
const SHELL = ['./', 'index.html', 'style.css', 'core.js', 'smile.js', 'sim.js', 'library.js', 'engine.js', 'app.js', 'capture.js', 'results.js', 'settings.js', 'worker.js', 'selftest.jpg', 'selftest.json', 'manifest.webmanifest', 'icon.svg', 'icon-192.png', 'icon-512.png'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith('dsd2-') && k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', (e) => {
  const req = e.request; if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const heavy = url.pathname.includes('/vendor/') || url.pathname.endsWith('.onnx') || /fonts\.(googleapis|gstatic)\.com|cdn\.jsdelivr\.net|storage\.googleapis\.com/.test(url.hostname);
  if (heavy) {
    e.respondWith(caches.open('dsd-runtime').then(async (c) => { const hit = await c.match(req); if (hit) return hit; const res = await fetch(req); if (res.ok || res.type === 'opaque') c.put(req, res.clone()); return res; }));
    return;
  }
  if (url.origin === location.origin) {
    e.respondWith(fetch(req).then((res) => { if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); } return res; }).catch(() => caches.match(req).then((r) => r || caches.match('index.html'))));
  }
});
