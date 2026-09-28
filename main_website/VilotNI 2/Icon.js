/* VilotNI 2 visual identity icon.
 * UI-only. This file does not participate in inference or model behavior.
 */
(function attachVilotNI2Icon(global) {
  'use strict';

  const SVG = `
    <svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="vilotni2-icon-gradient" x1="3" y1="2" x2="29" y2="30" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#8300FB"/>
          <stop offset="1" stop-color="#FE0070"/>
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="9" fill="url(#vilotni2-icon-gradient)"/>
      <path d="M18.72 5.7 10.4 17.02h5.18l-2.2 9.28 8.23-11.47h-5.06l2.17-9.13Z" fill="white"/>
    </svg>`;

  function markup(options = {}) {
    const size = Number.isFinite(options.size) ? Math.max(16, options.size) : 32;
    const title = options.title || 'VilotNI 2';
    return `<span class="vilotni2-logo-mark" role="img" aria-label="${String(title).replace(/"/g, '&quot;')}" style="width:${size}px;height:${size}px">${SVG}</span>`;
  }

  function mount(target, options = {}) {
    const el = typeof target === 'string' ? document.querySelector(target) : target;
    if (!el) return false;
    el.innerHTML = markup(options);
    return true;
  }

  global.VilotNI2Icon = Object.freeze({ markup, mount });
})(window);
