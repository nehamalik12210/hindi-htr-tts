/**
 * Phase 6 — Theme toggle (dark/light mode).
 * Dark mode is default. Persists choice in localStorage.
 */

(function () {
    const toggle = document.getElementById('themeToggle');
    const icon = document.getElementById('themeIcon');
    const html = document.documentElement;

    // Load saved theme
    const saved = localStorage.getItem('htr-theme') || 'dark';
    html.setAttribute('data-theme', saved);
    icon.textContent = saved === 'dark' ? '🌙' : '☀️';

    toggle.addEventListener('click', () => {
        const current = html.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', next);
        localStorage.setItem('htr-theme', next);
        icon.textContent = next === 'dark' ? '🌙' : '☀️';
    });
})();
