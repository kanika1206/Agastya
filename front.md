# AGASTYA — Frontend Build Prompt (Claude Design)

> **Paste this whole file into Claude design as the frontend brief. Build a production-grade dashboard for the AGASTYA traffic-violation enforcement backend. Every feature below is already implemented and live in the backend API — the frontend must surface all of them. Do not invent endpoints that are not listed. Do not ship a generic template.**

---

## 0. WHAT YOU ARE BUILDING

**AGASTYA is a quality-adaptive Restore → Detect → Verify cascade for automated Indian traffic-violation enforcement from photographic evidence.** **The backend ingests road images, runs a CV pipeline (gate → restore → detect → associate → OCR → calibrate), emits cryptographically tamper-evident violation records, and serves them over a read-only JSON HTTP API.** **You are building the operator-facing web dashboard that consumes that API: an enforcement review console where an officer browses violations, inspects annotated evidence, verifies the tamper-evident chain, and reads fleet-wide analytics.**

**The winning thesis the UI must communicate: every other team ships a plain YOLO + OCR table. AGASTYA wins on (a) robustness to degraded images and (b) cryptographically verifiable, explainable, legally-defensible evidence. The frontend must make the TRUST and EXPLAINABILITY story visible and obvious, not buried.**

---

## 1. THE BACKEND API (CONTRACT — BUILD AGAINST THIS EXACTLY)

**Base URL: configurable via env, default `http://localhost:8000`. The backend is FastAPI served by `uvicorn agastya.api.server:app`.**

**Every response uses the SAME envelope. Never assume a bare payload:**

```json
{ "success": true, "data": <payload>, "error": null, "meta": { } }
```

**On error: `success=false`, `data=null`, `error="<message>"`. HTTP 404 for missing violation. HTTP 422 for invalid query params (e.g. bad sort value).**

**CORS is already enabled server-side for `http://localhost:5173` and `http://localhost:3000` (override via `AGASTYA_CORS_ORIGINS`). Only `GET` is allowed.**

### **Endpoints (ALL must be used by the UI):**

**`GET /health`** — **liveness. `data: { "status": "ok" }`. Use for a connection/status indicator.**

**`GET /stats`** — **dashboard analytics. Returns:**
```json
{ "total": 1280, "by_type": { "no-helmet": 900, "triple-riding": 380 },
  "by_day": { "2026-06-18": 120, "2026-06-19": 160 },
  "mean_confidence": 0.91 }
```
**`by_day` is a date→count trend series. `by_type` is the violation-type breakdown. `mean_confidence` is fleet-wide calibrated mean.**

**`GET /violations`** — **paginated, filterable list. Query params (ALL optional, ALL must be wired into the UI):**
- **`violation_type`** — **exact match (`no-helmet` | `triple-riding`)**
- **`camera_id`** — **exact match**
- **`plate`** — **substring search (case-insensitive LIKE), empty string = no filter**
- **`created_from`** / **`created_to`** — **ISO datetime range on record creation**
- **`min_confidence`** / **`max_confidence`** — **floats 0.0–1.0**
- **`sort`** — **`newest` (default) | `oldest`**
- **`limit`** — **1–500, default 50**
- **`offset`** — **pagination offset, default 0**

**Response `data` is an array of list rows:**
```json
{ "id": 1, "content_hash": "54639e...", "violation_type": "no-helmet",
  "confidence": 0.88, "plate": "KA01AB1234", "camera_id": "CAM-7",
  "captured_at": "2026-06-19T10:00:00Z", "evidence_root": "a3f1...",
  "created_at": "2026-06-19T22:07:00+00:00", "image_path": "evidence/1.jpg" }
```
**`meta` carries `{ total, limit, offset, sort }` — drive pagination + result count from `meta.total`.**

**`GET /violations/{id}`** — **full evidence bundle for one violation: `{ id, content_hash, evidence_root, credential, audit_chain, created_at, image_path }`. `credential` is the C2PA-style signed manifest (`alg`, `signature`, `manifest{...}`). `audit_chain` is the hash-chained immutable log (array of `{seq,event,payload_hash,prev_hash,entry_hash}`). 404 if missing.**

**`GET /violations/{id}/image`** — **the annotated evidence JPEG (FastAPI FileResponse, `content-type: image/jpeg`). The detector boxes + violation label + confidence are already drawn onto the frame. 404 if no image on disk. Render this directly as the evidence photo.**

**`GET /violations/{id}/verify`** — **the TRUST endpoint. Returns:**
```json
{ "audit_chain_valid": true, "credential_signature_valid": true,
  "signing_key_configured": true }
```
**`audit_chain_valid` = the hash-chain log is intact (no tamper/reorder). `credential_signature_valid` = the HMAC-signed manifest matches the pixels+metadata (null if no signing key configured). This is the cryptographic tamper-evidence proof — surface it prominently with strong pass/fail visual states.**

---

## 2. SCREENS / FEATURES (BUILD EVERY ONE)

### **2.1 Analytics Dashboard (landing)**
- **Hero KPI band: total violations, breakdown by type, mean calibrated confidence — all from `/stats`.**
- **Trend chart of `by_day` (violations over time) — treat the chart as a first-class designed element, not a stock library default.**
- **Violation-type split (`by_type`) as a deliberate visualization (donut / bar with real hierarchy).**
- **Live backend status pill driven by `/health`.**

### **2.2 Violations Console (core screen)**
- **Filterable, sortable, paginated table/grid of violations from `/violations`.**
- **Every filter wired: type, camera id, plate substring search, date range, confidence min/max slider, sort newest/oldest.**
- **Filters belong in the URL (shareable state): a filtered view should be linkable and survive refresh.**
- **Result count + pagination driven by `meta.total` / `limit` / `offset`.**
- **Each row shows a thumbnail (from `/violations/{id}/image`), type, confidence, plate, camera, timestamp, and a verify badge.**
- **Confidence rendered semantically (calibrated score → color/state), not as raw text only.**

### **2.3 Violation Detail / Evidence Inspector**
- **Large annotated evidence image (`/violations/{id}/image`) with the drawn detection boxes.**
- **Full record metadata from `/violations/{id}`: violation type, calibrated confidence, plate, camera, captured/created timestamps, content hash, evidence root, model versions (from `credential.manifest.model_versions`).**
- **Calibration / uncertainty block: if present in the manifest metadata — `raw_confidence`, `conformal_set`, `human_review` flag — show them; surface a clear "NEEDS HUMAN REVIEW" state when flagged (statistical abstention is a headline feature).**

### **2.4 Trust & Tamper-Evidence Panel (the differentiator)**
- **A dedicated, visually strong verification panel powered by `/violations/{id}/verify`.**
- **Two explicit cryptographic checks with bold pass/fail states: AUDIT CHAIN VALID and CREDENTIAL SIGNATURE VALID.**
- **Show the evidence root / content hash and the C2PA-style credential.**
- **Render the audit chain (`seq`, `event`, hashes) as an immutable timeline — chain-of-custody made visible.**
- **Reference the standards-compliance story (ISO/IEC 27037/27041/27042/27043, NIST SP 800-86, eIDAS) — these live in the manifest; present them as a legal-defensibility badge row.**

### **2.5 Operator quality-of-life**
- **Loading, empty, and error states for every API call (the envelope's `error` field drives error UI).**
- **Keyboard-navigable table, accessible focus states, reduced-motion support.**
- **Responsive: 320 / 768 / 1024 / 1440. No overflow at any breakpoint.**

---

## 3. DOMAIN VOCABULARY (use correct labels)

- **Violation types (7 total). Two are detector-driven and ALWAYS active; five are scene-rule-driven and fire when per-camera SceneContext (zones / stop-line / signal state / heading) is configured. Display all seven with human labels and a "live vs requires-config" indicator:**
  - **`no-helmet` → "No Helmet" (LIVE — detector + confidence gate)**
  - **`triple-riding` → "Triple Riding" (LIVE — ≥3 persons on one motorcycle, box association)**
  - **`seatbelt` → "Seatbelt" (rule — fires on `no-seatbelt` detector class; needs that class/data)**
  - **`illegal-parking` → "Illegal Parking" (rule — vehicle inside a configured no-parking zone polygon)**
  - **`stop-line` → "Stop-Line Violation" (rule — vehicle crosses configured stop-line)**
  - **`red-light` → "Red-Light Violation" (rule — vehicle crosses stop-line while signal state = red)**
  - **`wrong-side` → "Wrong-Side Driving" (rule — vehicle heading opposite the allowed direction; needs heading/multi-frame)**
- **`plate` = Indian license plate string (may be null when OCR abstained — show "—" / "Not read", never blank-crash).**
- **`confidence` = post-hoc calibrated (temperature-scaled) score, NOT raw model output.**
- **`conformal_set` / `human_review` = conformal-prediction abstention: when the prediction set is not a clean singleton, the case is flagged for human review.**
- **`content_hash` = SHA-256 over evidence pixels. `evidence_root` = Merkle/binding root. `credential` = HMAC-signed (C2PA-style) manifest.**

---

## 4. DESIGN DIRECTION (NON-NEGOTIABLE — NO GENERIC TEMPLATE)

- **Pick a specific, opinionated direction: a serious "civic / enforcement command console" — think defensible-evidence control room, not a SaaS marketing page. Dark luxury or disciplined Swiss/International both fit; choose one and commit.**
- **Define a real palette as design tokens (CSS custom properties). Use color SEMANTICALLY: violation severity, confidence level, verified-vs-tampered, needs-review.**
- **Deliberate typography pairing with real scale contrast for hierarchy (KPIs huge, metadata quiet).**
- **Depth/layering for the evidence inspector; the verification panel should feel weighty and authoritative.**
- **Designed hover/focus/active states everywhere; motion that clarifies flow, never decorative churn (animate transform/opacity only).**
- **Data viz (trend + type split) is part of the design system, not a default chart drop-in.**
- **Banned: default card grid with uniform spacing, stock centered hero, unmodified library defaults, gray-on-white with one accent, dashboard-by-numbers with no point of view.**

---

## 5. TECH EXPECTATIONS

- **Modern React (Vite) or equivalent; TypeScript.**
- **Server state via TanStack Query / SWR (the API is read-only, stale-while-revalidate fits). Do NOT mirror server state into a client store.**
- **URL as state for all filters/sort/pagination.**
- **Typed API client matching the envelope `{ success, data, error, meta }` and every response shape above.**
- **One central `API_BASE_URL` from env (default `http://localhost:8000`).**
- **Performance budget: landing JS < 150kb gz; explicit image dimensions; lazy-load below-the-fold evidence thumbnails; eager + high priority only on the detail hero image.**

---

## 6. ACCEPTANCE CRITERIA

- **Dashboard renders real `/stats` (totals, by_type, by_day trend, mean confidence).**
- **Console lists `/violations` with EVERY filter (type, camera, plate, date range, confidence range), sort, and working pagination off `meta.total`.**
- **Detail screen shows annotated evidence image + full bundle metadata + calibration/review state.**
- **Verify panel calls `/violations/{id}/verify` and shows bold AUDIT CHAIN and SIGNATURE pass/fail, plus the audit-chain timeline and standards badges.**
- **Every screen has loading / empty / error states driven by the envelope.**
- **Null plate, no-evidence-image (404), and disabled-signing-key (null signature) edge cases all handled gracefully.**
- **Responsive 320→1440, accessible, no console errors.**
