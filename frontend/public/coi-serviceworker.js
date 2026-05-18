/**
 * coi-serviceworker — GitHub Pages에서 Cross-Origin Isolation 활성화.
 * SharedArrayBuffer (WebGPU / WASM 스레드) 사용에 필요한
 * COOP / COEP 헤더를 service worker 레벨에서 주입한다.
 */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  if (e.request.cache === "only-if-cached" && e.request.mode !== "same-origin") {
    return;
  }
  e.respondWith(
    fetch(e.request).then((res) => {
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      return new Response(res.body, {
        status: res.status,
        statusText: res.statusText,
        headers,
      });
    })
  );
});
