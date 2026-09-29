// Wires up the theme-toggle button. The initial theme attribute is set by a
// tiny inline script in each page's <head> (before this file loads) so there's
// no flash of the wrong theme; this file only handles the click-to-toggle.
(function () {
  function currentTheme() {
    return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
  }

  function applyIcon(btn, theme) {
    // Shows the icon for the theme you'd switch TO.
    btn.textContent = theme === 'dark' ? '☀️' : '🌙';
  }

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    applyIcon(btn, currentTheme());
    btn.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) {}
      applyIcon(btn, next);
    });
  });
})();
