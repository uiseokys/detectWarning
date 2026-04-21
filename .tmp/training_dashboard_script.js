
    (function () {
      try {
        const params = new URLSearchParams(window.location.search);
        const host = String(window.location.hostname || '').toLowerCase();
        const isTunnelHost =
          host.includes('trycloudflare') ||
          host.endsWith('.workers.dev');
        if (params.get('viewer') === '1' || isTunnelHost) {
          document.documentElement.classList.add('viewer-mode-page');
        }
      } catch (error) {}
    })();
  