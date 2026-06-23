/* agastya.js — real API client, formatters, shared chrome, motion helpers.
   Talks to the live FastAPI backend (envelope { success, data, error, meta }).
   window.AGASTYA = { api, fmt, chrome, motion, cfg } */
(function () {
  "use strict";

  var qp = new URLSearchParams(location.search);
  var BASE = qp.get("api") || window.AGASTYA_API || "https://agastyaflip.onrender.com";

  /* ---------------- API ---------------- */
  function env(path) {
    return fetch(BASE + path, { headers: { Accept: "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (j) {
        if (j && j.success === false) throw new Error((j.error) || "request failed");
        return j; // { success, data, error, meta }
      });
  }
  function qs(p) {
    if (!p) return "";
    var u = new URLSearchParams();
    Object.keys(p).forEach(function (k) {
      var v = p[k];
      if (v !== null && v !== undefined && v !== "") u.set(k, v);
    });
    var s = u.toString();
    return s ? "?" + s : "";
  }
  var api = {
    base: BASE,
    health: function () { return env("/health"); },
    stats: function () { return env("/stats"); },
    metrics: function () { return env("/metrics"); },
    // params: violation_type, camera_id, plate, created_from, created_to,
    //         min_confidence, max_confidence, sort(newest|oldest), limit, offset
    violations: function (p) { return env("/violations" + qs(p)); },
    violation: function (id) { return env("/violations/" + id); },
    verify: function (id) { return env("/violations/" + id + "/verify"); },
    imageURL: function (id) { return BASE + "/violations/" + id + "/image"; },
  };

  /* ---------------- violation taxonomy (backend stores slugs only) ---------------- */
  var VT = {
    "no-helmet":           { label: "No Helmet",            tier: 2 },
    "triple-riding":       { label: "Triple Riding",        tier: 3 },
    "no-seatbelt":         { label: "No Seatbelt",          tier: 2 },
    "wrong-side-driving":  { label: "Wrong-Side Driving",   tier: 3 },
    "stop-line-violation": { label: "Stop-Line Violation",  tier: 1 },
    "red-light-violation": { label: "Red-Light Violation",  tier: 3 },
    "illegal-parking":     { label: "Illegal Parking",      tier: 1 },
  };
  // which slugs the live pipeline actually emits today
  var LIVE_TYPES = ["no-helmet", "triple-riding"];

  /* ---------------- formatters ---------------- */
  var MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  var fmt = {
    n: function (n) { return (n || 0).toLocaleString("en-IN"); },
    pct: function (x) { return Math.round((x || 0) * 100) + "%"; },
    vtLabel: function (v) { return (VT[v] && VT[v].label) || titleize(v); },
    vtTier: function (v) { return (VT[v] && VT[v].tier) || 1; },
    tierColor: function (t) { return t === 3 ? "#d8261c" : t === 2 ? "#e6a700" : "#1f9d4d"; },
    tierLabel: function (t) { return t === 3 ? "HIGH" : t === 2 ? "MED" : "LOW"; },
    confColor: function (c) { return c >= 0.85 ? "#1f9d4d" : c >= 0.72 ? "#e6a700" : "#d8261c"; },
    isLive: function (v) { return LIVE_TYPES.indexOf(v) >= 0; },
    dateTime: function (d) {
      if (!d) return "—"; var x = new Date(d);
      var hh = String(x.getUTCHours()).padStart(2, "0"), mm = String(x.getUTCMinutes()).padStart(2, "0");
      return MON[x.getUTCMonth()] + " " + x.getUTCDate() + ", " + x.getUTCFullYear() + " · " + hh + ":" + mm + " UTC";
    },
    day: function (d) { var x = new Date(d); return MON[x.getUTCMonth()] + " " + x.getUTCDate(); },
    ago: function (d) {
      if (!d) return "—";
      var s = Math.floor((Date.now() - new Date(d).getTime()) / 1000);
      if (s < 60) return "just now";
      if (s < 3600) return Math.floor(s / 60) + "m ago";
      if (s < 86400) return Math.floor(s / 3600) + "h ago";
      return Math.floor(s / 86400) + "d ago";
    },
    hash: function (h, n) { return (h || "").slice(0, n || 16) + "…"; },
  };
  function titleize(v) { return (v || "").split("-").map(function (w) { return w.charAt(0).toUpperCase() + w.slice(1); }).join(" "); }

  /* ---------------- shared chrome (rendered once per page) ---------------- */
  var LOGO = '<svg width="30" height="30" viewBox="0 0 48 48" fill="none"><circle cx="24" cy="24" r="21" stroke="#1f9d4d" stroke-width="2"/><circle cx="24" cy="24" r="15" stroke="#a0a096" stroke-width="1"/><path d="M24 9V39M9 24H39M14 14L34 34M34 14L14 34" stroke="#1f9d4d" stroke-width="1.3" opacity="0.5"/><circle cx="24" cy="24" r="4.5" fill="#eceae4" stroke="#1f9d4d" stroke-width="1.6"/><circle cx="24" cy="24" r="1.8" fill="#d8261c"/></svg>';
  var NAV = [
    { href: "dashboard.html", key: "dashboard", label: "DASHBOARD" },
    { href: "violations.html", key: "violations", label: "VIOLATIONS" },
    { href: "performance.html", key: "performance", label: "PERFORMANCE" },
    { href: "reports.html", key: "reports", label: "REPORTS" },
  ];
  var chrome = {
    header: function (active) {
      var links = NAV.map(function (n) {
        return '<a href="' + n.href + '"' + (n.key === active ? ' class="is-active"' : "") + ">" + n.label + "</a>";
      }).join("");
      return (
        '<div class="ag-tricolor"></div>' +
        '<header class="ag-header">' +
          '<a class="ag-brand" href="index.html">' + LOGO +
            '<span class="wordmark">AGASTYA</span>' +
            '<span class="sub">BENGALURU CITY TRAFFIC<br>ENFORCEMENT GRID</span>' +
          "</a>" +
          '<nav class="ag-nav">' + links +
            '<span class="ag-live" data-ag-live><span class="dot"></span>LIVE</span>' +
          "</nav>" +
        "</header>"
      );
    },
    footer: function () {
      var stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
      return (
        '<footer class="ag-footer">' +
          "<span>AGASTYA · TAMPER-EVIDENT ENFORCEMENT · BENGALURU CITY POLICE</span>" +
          "<span>SIGNED AT SOURCE · " + stamp + " UTC</span>" +
        "</footer>"
      );
    },
    // mount header into [data-ag-header], footer into [data-ag-footer]
    mount: function (active) {
      var h = document.querySelector("[data-ag-header]");
      if (h) h.innerHTML = chrome.header(active);
      var f = document.querySelector("[data-ag-footer]");
      if (f) f.innerHTML = chrome.footer();
    },
    offline: function () {
      var el = document.querySelector("[data-ag-live]");
      if (el) { el.classList.add("is-offline"); el.innerHTML = '<span class="dot"></span>OFFLINE'; }
    },
  };

  /* ---------------- motion helpers ---------------- */
  var motion = {
    intro: function (root) {
      var scope = root || document;
      var keys = scope.querySelectorAll("[data-k]");
      keys.forEach(function (el, i) {
        el.style.transition = "opacity .8s var(--ease), transform .8s var(--ease)";
        el.style.transitionDelay = (0.05 + i * 0.07) + "s";
        requestAnimationFrame(function () { requestAnimationFrame(function () { el.style.opacity = "1"; el.style.transform = "none"; }); });
      });
      motion.reveal(scope);
    },
    reveal: function (root) {
      var scope = root || document;
      var els = scope.querySelectorAll("[data-reveal]");
      if (!els.length) return;
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          if (e.isIntersecting) {
            var el = e.target;
            el.style.transition = "opacity .7s var(--ease), transform .7s var(--ease)";
            el.style.transitionDelay = ((+el.getAttribute("data-reveal-delay") || 0) / 1000) + "s";
            el.style.opacity = "1"; el.style.transform = "none";
            io.unobserve(el);
          }
        });
      }, { threshold: 0.08 });
      els.forEach(function (el) { io.observe(el); });
      setTimeout(function () { els.forEach(function (el) { el.style.opacity = "1"; el.style.transform = "none"; }); }, 1700);
    },
    countUp: function (el, target, dur) {
      if (!el) return; dur = dur || 1800;
      var start = performance.now();
      function step(t) {
        var p = Math.min(1, (t - start) / dur), e = 1 - Math.pow(1 - p, 3);
        el.textContent = fmt.n(Math.round(target * e));
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    },
  };

  window.AGASTYA = { api: api, fmt: fmt, chrome: chrome, motion: motion,
    cfg: { base: BASE, VT: VT, LIVE_TYPES: LIVE_TYPES } };
})();
