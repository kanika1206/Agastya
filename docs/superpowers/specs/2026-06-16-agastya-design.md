# AGASTYA — Design Spec

*Quality-adaptive, evidence-grade traffic-violation vision system for Indian roads.*

Renamed from DRISHTI. Source: `ps`, `researchdone`, `DRISHTI_master_build_prompt_v2.md`. Naming: pure brand, no backronym.

## 1. Thesis & Scope

Competing baseline (YOLO + Tesseract) collapses under low-light, rain, shadows, motion blur, and small/distant/occluded objects. AGASTYA wins on (a) degradation robustness and (b) cryptographically tamper-evident, explainable evidence. Scientific novelty: restoration optimized for downstream machine perception (mAP/OCR), not human PSNR, inside a closed Restore→Detect→Verify loop with statistically-guaranteed abstention.

- **On paper (full system):** helmet, seatbelt, triple-riding, wrong-side, stop-line, red-light, illegal parking + ANPR + analytics.
- **Demo (vertical slice):** helmet non-compliance + triple riding + ANPR + tamper-evident evidence. Everything else = documented extension points.

## 2. Architecture — Four-Stage Cascade

### Stage 0 — Quality Gating
ARNIQA (WACV 2024, arXiv:2310.14918) no-reference IQA scores each detected crop. Only crops below threshold route to restoration; clean crops bypass. Optional region-aware gating via CLIP-IQA+/QualiCLIP.

### Stage 1 — Restoration
- NAFNet-width32 (ECCV 2022, arXiv:2204.04676) — motion blur + noise.
- Real-ESRGAN ×4 (ICCV-W 2021) fine-tuned on synthetic high-order degradation over plate/rider crops.
- LCDNet (arXiv:2505.06393) — license-plate-specific, character-aware SR.
- Restoration-aware / task-driven loss (OCR-as-discriminator term) = headline loss novelty. Optimize for detector + OCR, not PSNR/SSIM.
- Hallucination guard (legal-critical): OCR-confidence threshold + plate-format/character-count consistency + optional cross-frame voting + DINOv2/CLIP feature proxy. Abstain over emitting low-confidence plate.
- Optional: MWFormer/EDTR/TeReDiff/DTRDNet (multi-weather), SeeSR/SUPIR (diffusion SR) — never on the OCR/evidence path.

### Stage 2 — Detection + Association
- Primary detector: YOLO26-m (arXiv:2509.25164, NMS-free E2E, DFL removed). YOLO26-s low-VRAM fallback. YOLOv11-m stability fallback.
- Native P2 small-object head (`yolo26-p2.yaml`); native ProgLoss + STAL + MuSGD.
- Classes: `{motorcycle, rider, helmet, no-helmet, person, car, truck, bus, auto-rickshaw, license-plate}`.
- SAHI sliced inference for small/distant objects. Caveat: SAHI stitch uses NMS-style merge; run YOLO26 one-to-many branch (`end2end=False`) for slicing. Validate in Phase 1.
- Custom IoU loss (MPDIoU/WIoUv3/Focaler-IoU) = ablation-only (ProgLoss+STAL already target small objects).
- Augmentation: Mosaic + CutMix + Copy-Paste (minority classes) + weather aug.
- Rider–motorcycle association: trapezium box (arXiv:2204.08364) and/or SAM 2 mask. Triple-riding = ≥3 person instances on one motorcycle mask. SAM 2 zero-shot, prompted by detector boxes; MobileSAM/EfficientSAM if VRAM-bound.
- Ablation detectors: YOLOv11-m, RT-DETRv2 (arXiv:2407.17140).

### Stage 3 — Recognition + Classification + Confidence
- PARSeq (ECCV 2022, arXiv:2207.06966) plate OCR, LoRA fine-tune on Indian plates.
- Violation-classification head → violation type.
- Confidence: temperature scaling + conformal prediction sets (APS/RAPS) for coverage guarantees and principled abstention. OOD detection for distribution shift.

### Stage 4 — Evidence Chain
- Per violation: annotated image + JSON metadata (timestamp, GPS/camera id, violation class, calibrated confidence + conformal set, plate string, model versions, human-review flag).
- C2PA content credential (Spec 2.4): SHA-256 hard binding over evidence pixels + X.509-signed manifest.
- Grad-CAM explainability overlays.
- Standards-compliance manifest (ISO/IEC 27037/27041/27042/27043, NIST, eIDAS).
- Local Merkle / hash-chain audit log of evidence creation/access.

### Extra cheap wins
TTA + Weighted Box Fusion; DINOv2 features for gate/distillation; optional knowledge distillation.

### Optional multi-frame path (AI City Track 5 is video)
Cross-frame voting / temporal consistency; ByteTrack (BoT-SORT/StrongSORT for heavy occlusion); ReID lowest priority.

## 3. Data Flow

```
image
  → ARNIQA gate
  → [restore if degraded: NAFNet + Real-ESRGAN/LCDNet]
  → YOLO26 + SAHI detect
  → SAM 2 associate (rider↔motorcycle)
  → crop plate → restore plate → PARSeq OCR
  → classify + calibrate (temp scaling + conformal)
  → evidence chain (C2PA + Merkle + Grad-CAM + standards manifest)
  → SQLite/Parquet store
  → analytics dashboard
```

## 4. Train vs Pretrained
- Train / fine-tune: YOLO26 (+P2), NAFNet, Real-ESRGAN, LCDNet, PARSeq (LoRA), ARNIQA only if adapting.
- Pretrained / zero-shot: SAM 2 / MobileSAM, base ARNIQA, base PARSeq, C2PA tooling, DINOv2/CLIP proxies, optional diffusion SR, optional MWFormer/EDTR/TeReDiff.

## 5. Datasets
- IDD / IDD-Detection (IIIT-H) — flagship, ~47k images, 40 classes.
- IDD motorcycle-violation derivative (arXiv:2204.08364) — helmet + trapezium triple-riding labels.
- AI City Challenge 2024 Track 5 — 100 train + 100 test videos, 20 s @ 10 fps, 1920×1080; credibility anchor Rank-1 Co-DETR 0.4860 mAP.
- Indian ANPR set (16,192-image 4-point + Kaggle) — plate detection + PARSeq fine-tune. Indian plates ≠ CCPD; fine-tune.
- Optional: DriveIndia (arXiv:2507.19912), IITH Helmet 1/2.
- Restoration/SR: GoPro (NAFNet); synthetic degradation over plate/rider crops (Real-ESRGAN/LCDNet).
- Seatbelt extension (weak data): Roboflow windshield sets + windshield-ROI→seatbelt-classify.
- Annotation-gap handling: unify all to one YOLO-format schema; SAM 2-bootstrap + manual review; domain adaptation + heavy augmentation for IDD↔AI City bias.

## 6. Evaluation (mapped to PS metrics)
- Detection: mAP@50, mAP@50–95, per-class P/R/F1 on IDD + AI City test.
- Degradation-stratified mAP (clean/blur/low-light/low-res) + object-size-stratified — proves restoration novelty.
- Restoration ablation: detector mAP + OCR accuracy with/without NAFNet/SR/LCDNet; PSNR↑ ≠ mAP↑.
- Quality-gate ablation: ARNIQA-gated vs always-on vs never (accuracy + latency).
- Association ablation: triple-riding P/R, SAM 2 masks vs box-IoU.
- OCR: plate + character accuracy vs resolution curve.
- Loss ablation: YOLO26 native vs +MPDIoU vs +WIoUv3 vs +Focaler-IoU on small-object AP.
- Calibration: ECE + reliability diagrams; conformal coverage @ target risk; abstention rate.
- Efficiency: FPS, per-stage latency, peak VRAM, params/FLOPs; show gate cuts average latency; INT8/FP16 numbers.
- Novelty-proof rule: every novel block beats the honest YOLO+OCR baseline on ≥1 PS metric, in a clean ablation table.

## 7. Roadmap (single GPU, ~4 weeks)
- Phase 1 (wk1): baseline + data — assemble IDD + AI City + ANPR, unify schema; train YOLO26-m baseline (+P2), AMP, 640px; validate SAHI↔NMS-free; establish baseline metrics (the control).
- Phase 2 (wk2): restoration + gating — fine-tune NAFNet + Real-ESRGAN + LCDNet; build ARNIQA gate; measure restoration-conditioned mAP + OCR-vs-resolution + hallucination guard.
- Phase 3 (wk3): association + OCR + uncertainty — SAM 2/MobileSAM association + trapezium triple-riding; PARSeq LoRA OCR; temperature scaling + conformal; copy-paste/weather aug; TTA+WBF.
- Phase 4 (wk4): evidence + analytics + ablations — C2PA + Grad-CAM + standards manifest + Merkle log; SQLite/Parquet store + dashboard; full ablation matrix. Optional: custom-IoU ablation, RT-DETRv2/YOLOv11 comparison, multi-frame, diffusion SR.

## 8. Repo Structure & Constraints
Many small focused modules:
```
agastya/
  stages/{gate,restore,detect,associate,recognize,evidence}/
  data/        (download, schema-unify, annotation-bootstrap)
  eval/        (metrics, stratified sweeps, ablation harness)
  analytics/   (store, dashboard)
  configs/     (model + pipeline configs, yolo26-p2.yaml)
  pipeline.py  (orchestrates the cascade)
```
- Single consumer GPU, 12–24 GB VRAM. AMP everywhere; auto-batch; gradient accumulation/checkpointing if needed.
- Low-VRAM (<10 GB) fallback: YOLO26-s, NAFNet-width16, lightweight SR (IMDN/RFDN), MobileSAM, skip diffusion, tile inputs.
- Report peak VRAM, params, FLOPs, FPS, per-stage latency per model.
- Zero code comments in this repo (project rule).
- Workflow: nothing trains/downloads/executes destructively without explicit written "go" per step.

## 9. Do-Not-Include
Qwen-HS (does not exist); DINO-HS as plug-in module; DAIR as restoration; Hyperledger/blockchain (prototype); KAN/efficient-KAN/GR-KAN on critical path (optional honest ablation only); Group DRO / SAMURAI crowd modules / stacked RFLA+QueryDet+Bi-AFPN (ablation-only).

## 10. Open Risks
- SAHI ↔ YOLO26 NMS-free interaction — explicit Phase 1 test, most likely silent breakage.
- External arXiv IDs except YOLO26 accepted on prior verification — confirm before publication.
- Diffusion SR character fabrication — excluded from evidence path.
- No public Indian windshield-seatbelt dataset — seatbelt extension weaker than helmet/triple-riding slice.
