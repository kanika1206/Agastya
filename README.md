# AGASTYA

**Automated, tamper-evident traffic-violation enforcement from photographic evidence.**

AGASTYA is a quality-adaptive **Restore → Detect → Verify** computer-vision cascade for Indian roads. It ingests traffic images, repairs degraded inputs *for machine readability* (not for the human eye), detects vehicles and road users, classifies violations with calibrated confidence, reads number plates, and seals every result into **cryptographically tamper-evident, explainable evidence** that can be independently verified.

> Winning thesis: every other approach ships a plain YOLO + OCR table that collapses under low light, rain, shadow and motion blur. AGASTYA wins on (a) robustness to degraded images and (b) court-admissible, verifiable evidence.

## Architecture — four-stage cascade

```
IMAGE → [0 Quality Gate] → [1 Restore] → [2 Detect + Associate] → [3 Verify] → [4 Evidence Chain]
         ARNIQA (route       NAFNet ·        YOLO26 (P2, NMS-free)    PARSeq OCR    hash-bind → HMAC
         only degraded)      Real-ESRGAN ·   · SAHI · SAM2/trapezium  · classify ·   sign → Merkle
                             LCDNet + task    triple-riding rule       temp-scaling   chain → Grad-CAM
                             OCR loss                                  + conformal
                                                                       abstention
```

Stage backends are config-selected factories — swap NAFNet→passthrough or SAM2→box-overlap for low-VRAM without touching the pipeline.

## Components

- `agastya/stages/` — gate · restore · detect · associate · ocr · evidence · violations (pluggable backends)
- `agastya/verify/` — temperature scaling + conformal calibration
- `agastya/eval/` — E2E evaluation, ablations, degradation stratification, P/R/F1
- `agastya/store/` — SQLite evidence store
- `agastya/api/` — FastAPI service
- `web/` — 7-page operator dashboard (Landing · Dashboard · Violations · Detail · Verify · Performance · Reports)

## Quick start

```bash
# 1. install
pip install -e .

# 2. configure
cp .env.example .env            # then set AGASTYA_SIGNING_KEY

# 3. seed demo evidence (140 signed violations)
AGASTYA_SIGNING_KEY=dev-key PYTHONPATH=. python3 scripts/seed_demo.py

# 4. run the API
AGASTYA_SIGNING_KEY=dev-key PYTHONPATH=. python3 -m uvicorn agastya.api.server:app --port 8000

# 5. serve the dashboard
cd web && python3 -m http.server 3000   # open http://localhost:3000
```

## API

Read-only JSON, envelope `{ success, data, error, meta }`:

`GET /health` · `GET /stats` · `GET /metrics` · `GET /violations` (filter/paginate) · `GET /violations/{id}` · `GET /violations/{id}/verify` · `GET /violations/{id}/image`

## Evidence & trust

Every violation record is hash-bound over its evidence pixels (SHA-256), signed (HMAC-256), and appended to a Merkle audit chain. `GET /violations/{id}/verify` recomputes the hash, signature, and chain — tamper one byte and verification fails. A standards manifest references ISO/IEC 27037/27041/27042/27043, NIST, and eIDAS.

## Testing

```bash
pytest            # 269 tests
```

## Scope

Demo is a deep vertical slice — **no-helmet + triple-riding + ANPR + tamper-evident evidence**. The remaining five violation types (seatbelt, wrong-side, stop-line, red-light, illegal parking) are documented extension points; the temporal ones require the optional multi-frame path.

## License

MIT — see [LICENSE](LICENSE).
