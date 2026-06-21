# AGASTYA — MASTER BUILD PROMPT (v2, reviewer-integrated + verified)

> Paste this whole document as the project brief / CLAUDE.md for the build agent. It specifies the complete architecture, every technology, every dataset, every research paper (with verified arXiv IDs), the evaluation plan, the single-GPU constraints, and the phased roadmap. **Do not omit any component listed here, and do not add anything from the DO-NOT-INCLUDE list in Section 10.**
>
> Changes vs v1: primary detector upgraded to **YOLO26** (P2 head + ProgLoss + STAL are NATIVE); quality gate upgraded to **ARNIQA**; **LCDNet** added for plate-specific SR; **SAHI**, **conformal prediction**, **Grad-CAM + standards manifest + Merkle audit log** added; custom IoU loss demoted to ablation-only; optional multi-frame path added; explicit DO-NOT-INCLUDE list added.

---

## 0. ROLE & PRIME DIRECTIVE

You are a senior computer vision researcher + ML engineer. Build **AGASTYA**: a quality-adaptive **Restore → Detect → Verify** cascade for automated Indian traffic-violation enforcement from photographic evidence.

**Winning thesis:** every competing team builds the obvious YOLO + Tesseract baseline, which collapses under low light, rain, shadows, motion blur, and small/distant/occluded objects. AGASTYA wins on (a) robustness to degradation and (b) cryptographically tamper-evident, explainable evidence. Scientific novelty: **restoration optimized for downstream machine perception (mAP/OCR), not human PSNR**, inside a closed restore→detect→verify loop, with statistically-guaranteed abstention.

**Scope rule:** Full system on paper (helmet, seatbelt, triple-riding, wrong-side, stop-line, red-light, illegal parking + ANPR + analytics). The **demo is a deep vertical slice**: *helmet non-compliance + triple riding + ANPR + tamper-evident evidence*. Everything else = documented extension points, not shipped shallow.

**Workflow constraint (NON-NEGOTIABLE):** Nothing executes — no training, no download, no destructive command — without explicit discussion and my written confirmation at each step. Propose, wait for "go," then act.

---

## 1. HARDWARE / RUNTIME CONSTRAINTS
- Single consumer GPU, target **12–24 GB VRAM**, built on my own PC.
- **Mixed precision (AMP)** everywhere; auto-batch; gradient accumulation + gradient checkpointing if needed.
- **Low-VRAM fallback (<10 GB):** YOLO26-s; NAFNet-width16; lightweight SR (IMDN/RFDN/ShuffleMixer) instead of Real-ESRGAN ×4; MobileSAM/EfficientSAM instead of SAM 2; skip diffusion SR; tile high-res inputs.
- Report peak VRAM, params, FLOPs, FPS, per-stage latency for every model.

---

## 2. ARCHITECTURE — FOUR-STAGE CASCADE

### Stage 0 — Quality Gating (efficiency centerpiece)
- **ARNIQA** (WACV 2024, arXiv:2310.14918) no-reference IQA as the gate — robust to unseen distortions, lightweight. Replaces BRISQUE/MANIQA.
- Optional region-aware gating: CLIP-IQA+ / QualiCLIP.
- Routes ONLY degraded crops to restoration; clean crops bypass (blanket restoration hurts clean images + wastes latency).

### Stage 1 — Restoration (Restore)
- **NAFNet-width32** (ECCV 2022, arXiv:2204.04676) for motion blur + noise (SimpleGate + Simplified Channel Attention).
- **Real-ESRGAN ×4** (ICCV-W 2021) fine-tuned on a synthetic high-order degradation model (motion blur + downscale + JPEG) over plate/rider crops.
- **LCDNet** (arXiv:2505.06393) for license-plate-specific, layout/character-aware SR — directly OCR-optimized; use on plate crops where it beats generic SR.
- **Restoration-aware / task-driven loss** = headline loss novelty: OCR-as-discriminator / character-aware term (LPSRGAN/LCDNet-style) so outputs are optimized for detector + OCR, NOT PSNR/SSIM.
- **SR hallucination guarding (legal-critical):** OCR-confidence threshold + plate-format/character-count consistency check + (multi-frame path) cross-frame voting + optional DINOv2/CLIP feature-similarity proxy. Abstain rather than emit a low-confidence plate.
- **Optional multi-weather / text-aware restoration:** MWFormer (arXiv:2411.17226), EDTR (arXiv:2507.22459), TeReDiff (arXiv:2506.09993), DTRDNet (arXiv:2509.00925) — add only if rain/haze/text-region eval shows NAFNet is insufficient.
- **Optional max-quality diffusion SR:** SeeSR (arXiv:2311.16518) / SUPIR (arXiv:2401.13627) / StableSR — NEVER on the OCR/evidence path (hallucinates characters). Inference-only.

### Stage 2 — Detection + Association (Detect)
- **PRIMARY DETECTOR = YOLO26-m** (arXiv:2509.25164; Ultralytics official, NMS-free end-to-end, DFL removed). YOLO26-s for low-VRAM fallback. **YOLOv11-m kept only as a stability fallback** if YOLO26 shows training instability mid-build.
  - **Native P2 small-object head** via `yolo26-p2.yaml` (instantiate from YAML, train; no pretrained P2 weights shipped).
  - **Native small-object losses ProgLoss + STAL** + MuSGD optimizer — these come WITH YOLO26; do not re-implement.
- Classes: `{motorcycle, rider, helmet, no-helmet, person, car, truck, bus, auto-rickshaw, license-plate}`.
- **SAHI (Slicing Aided Hyper Inference)** at inference for distant/small plates & helmets — biggest small-object win for least effort.
  - **CAVEAT:** SAHI stitches slices with an NMS-style merge. YOLO26's default one-to-one head is NMS-free → run the **one-to-many branch (`end2end=False`)** for SAHI, or apply NMS only at the slice-stitch stage. Validate this in Phase 1.
- **Custom IoU loss = ABLATION-ONLY** (MPDIoU arXiv:2307.07662 / WIoUv3 / Focaler-IoU arXiv:2401.10525). Demoted because ProgLoss+STAL already target small objects natively; report as a comparison, don't make it the build's spine.
- **Augmentation (cheap win):** Mosaic + CutMix + **Copy-Paste** (for minority classes helmet-violation/triple-riding) + weather augmentation.
- **Rider–motorcycle association:** trapezium box (IDD method, arXiv:2204.08364) and/or SAM 2 mask. **Triple riding = ≥3 person instances associated with one motorcycle mask.** Plus explicit association logic (AI City-winner style) + per-class thresholding to suppress crowd false positives.
- **SAM 2** (Meta 2024), prompted by detector boxes — zero-shot, inference-only, never trained — for masks (association + clean evidence overlays). **Swap to MobileSAM/EfficientSAM** if VRAM-bound.
- **Ablation detectors (rigor):** YOLOv11-m, RT-DETRv2 (arXiv:2407.17140).

### Stage 3 — Recognition + Classification + Confidence (Verify pt.1)
- **PARSeq** (ECCV 2022, arXiv:2207.06966) plate OCR, fine-tuned on Indian plates (LoRA fine-tune = cheap, 1–2% OCR gain).
- Violation-classification head → violation type.
- **Confidence (PS requires it):** temperature scaling (post-hoc calibration, low ECE) + **conformal prediction sets (APS/RAPS)** for statistical coverage guarantees and principled abstention logging. OOD detection + adaptive calibration for distribution shift.

### Stage 4 — Evidence Chain (Verify pt.2, trust centerpiece)
- Per violation: annotated image + JSON metadata (timestamp, GPS/camera id, violation class, calibrated confidence + conformal set, plate string, model versions, human-review flag).
- **C2PA content credential** (Spec 2.4): SHA-256 hard binding over evidence pixels + X.509-signed manifest → tamper-evident.
- **Grad-CAM explainability overlays** per violation decision (XAI requirement; SHAP on the classification head only if time allows — SHAP on detection is too heavy).
- **Standards-compliance manifest** referencing ISO/IEC 27037 / 27041 / 27042 / 27043, NIST, eIDAS — near-free text, high legal-defensibility/judge impact.
- **Local Merkle / hash-chain audit log** of evidence creation/access (immutable, independently verifiable) + input hashing + chain-of-custody logging.

### Extra cheap wins (keep)
- **Test-Time Augmentation + Weighted Box Fusion (WBF)** — 1–3% mAP (AI City winners).
- **DINOv2 features for the quality gate / distillation** (NOT as the detector backbone — that blows VRAM/training budget). Optional.
- **Knowledge distillation** to keep models small/fast on single GPU. Optional.

### Optional multi-frame path (because AI City Track 5 is VIDEO)
- Core pipeline runs single-photo (matches PS). Expose an optional multi-frame flag for the AI City split:
  - **Cross-frame voting / temporal consistency** for detection+OCR — cheap, suppresses SR hallucination, boosts recall.
  - **ByteTrack** for multi-object tracking/association (BoT-SORT/StrongSORT for heavy occlusion). 
  - Vehicle/rider ReID = lowest priority, analytics polish only.

---

## 3. WHAT TO TRAIN vs USE PRETRAINED
- **Train / fine-tune:** YOLO26 detector (+P2), NAFNet, Real-ESRGAN, LCDNet, PARSeq (LoRA), ARNIQA only if adapting.
- **Pretrained / zero-shot:** SAM 2 / MobileSAM, base ARNIQA, base PARSeq, C2PA tooling, DINOv2/CLIP proxies, (optional) diffusion SR, (optional) MWFormer/EDTR/TeReDiff.

---

## 4. DATASETS (Indian-first, multi-source, annotation-gap-handled)
- **IDD / IDD-Detection** (IIIT-H) — flagship; ~47k images, 40 classes, unstructured Indian roads. Core vehicle/road-user detection.
- **IDD motorcycle-violation derivative** (arXiv:2204.08364) — helmet/no-helmet + trapezium triple-riding labels; reusable.
- **AI City Challenge 2024 Track 5** — 100 train + 100 test videos, 20 s @ 10 fps, 1920×1080, Indian city; per-rider boxes (≤4/motorcycle) + helmet status, 9 fine-grained classes. Credibility anchor: Rank-1 Co-DETR 0.4860 mAP (Vo et al., CVPR-W 2024). Enables the optional multi-frame path.
- **DriveIndia** (arXiv:2507.19912) — 66,986 images, 24 classes, weather/illumination diversity. Optional vehicle-robustness expansion.
- **IITH Helmet 1/2** — real CCTV helmet videos.
- **ANPR:** Indian plate set (e.g. 16,192-image, 4-point) + Kaggle Indian LP sets for plate detection + PARSeq fine-tuning. Indian plates ≠ CCPD (China); fine-tune (Nadiminti et al., arXiv:2207.06657).
- **Restoration/SR training:** GoPro (NAFNet deblur); synthetic degradation over plate/rider crops (Real-ESRGAN/LCDNet fine-tune).
- **Seatbelt (extension, weak data):** no public Indian windshield set; Roboflow windshield sets (IPB ~9.87k, akaike) + windshield-ROI→seatbelt-classify; watch minority-class recall collapse.
- **Annotation-gap handling:** IDD lacks helmet labels → add helmet + trapezium boxes, SAM 2-bootstrapped + manual review. AI City has missing annotations → sample + correct. Unify all to one YOLO-format schema. Domain adaptation (InterAug) + cross-domain validation + heavy augmentation to fight IDD↔AI City bias.

---

## 5. EVALUATION (map to PS metrics + prove novelty)
- **Detection:** mAP@50, mAP@50–95, per-class P/R/F1 on IDD + AI City test.
- **Degradation-stratified mAP** (clean/blur/low-light/low-res) + object-size-stratified — proves the restoration novelty.
- **Restoration ablation:** detector mAP + OCR char/plate accuracy WITH vs WITHOUT NAFNet/SR/LCDNet; show PSNR↑ ≠ mAP↑, and task-driven loss beats PSNR-only restoration downstream.
- **Quality-gate ablation:** ARNIQA-gated vs always-on vs never restoration (accuracy + average latency).
- **Association ablation:** triple-riding P/R with SAM 2 masks vs box-IoU heuristic.
- **OCR:** plate + character accuracy vs resolution curve (SR lifts low-res tail; precedent 65%→74.5%).
- **Loss ablation:** YOLO26 native (ProgLoss+STAL) vs +MPDIoU vs +WIoUv3 vs +Focaler-IoU on small-object AP.
- **Calibration/uncertainty:** ECE + reliability diagrams; conformal coverage @ target risk; abstention rate.
- **Efficiency/scalability:** FPS, per-stage latency, peak VRAM, params/FLOPs; show the gate cuts average latency; report quantized (INT8/FP16) numbers.
- **Novelty proof rule:** every novel block must beat the honest YOLO+OCR baseline on ≥1 PS metric, in a clean ablation table.
- **Analytics/reporting:** SQLite/Parquet violation store, searchable records, statistics + trend dashboard.

---

## 6. ROADMAP (single GPU, ~4 weeks)
- **Phase 1 (wk1) — Baseline + data:** assemble IDD + AI City + ANPR, unify schema; train **YOLO26-m baseline (+P2 via yolo26-p2.yaml)**, AMP, 640px; **validate SAHI ↔ NMS-free interaction**; establish baseline mAP/P/R/F1 (the control).
- **Phase 2 (wk2) — Restoration + gating:** fine-tune NAFNet-width32 + Real-ESRGAN + LCDNet (plates); build ARNIQA gate; measure restoration-conditioned mAP + OCR-vs-resolution + hallucination-guard behavior.
- **Phase 3 (wk3) — Association + OCR + uncertainty:** SAM 2/MobileSAM association + trapezium triple-riding logic; PARSeq (LoRA) OCR; temperature scaling + conformal prediction; copy-paste/weather aug; TTA+WBF.
- **Phase 4 (wk4) — Evidence + analytics + ablations:** C2PA + Grad-CAM + standards manifest + Merkle log; SQLite/Parquet store + dashboard; full ablation matrix. Optional: custom-IoU-loss ablation, RT-DETRv2/YOLOv11-m comparison, multi-frame (ByteTrack + cross-frame voting), diffusion SR branch.

---

## 7. VERIFIED PAPER / METHOD LIST
- **YOLO26** — Ultralytics, arXiv:2509.25164 (NMS-free E2E, DFL removed, ProgLoss+STAL+MuSGD, P2 head, 5 sizes). PRIMARY detector. [Self-verified.]
- **ProgLoss + STAL** — introduced in the YOLO26 paper (arXiv:2509.25164); STAL also arXiv:2407.08362. Native to YOLO26.
- **YOLOv11** — Ultralytics 2024. Stability fallback.
- **RT-DETRv2** — arXiv:2407.17140. Ablation.
- **NAFNet** — Chen et al., ECCV 2022, arXiv:2204.04676.
- **Real-ESRGAN** — Wang et al., ICCV-W 2021.
- **LCDNet** — arXiv:2505.06393 (license-plate, character-aware SR).
- **MWFormer** — arXiv:2411.17226; **EDTR** — arXiv:2507.22459; **TeReDiff** — arXiv:2506.09993; **DTRDNet** — arXiv:2509.00925 (optional restoration).
- **SeeSR** — arXiv:2311.16518; **SUPIR** — arXiv:2401.13627 (optional diffusion SR, off evidence path).
- **ARNIQA** — WACV 2024, arXiv:2310.14918 (NR-IQA gate).
- **SAM 2** — Ravi et al., Meta 2024; **MobileSAM/EfficientSAM** (lightweight swaps).
- **MPDIoU** — Ma & Xu, CVPR 2023, arXiv:2307.07662; **Focaler-IoU** — arXiv:2401.10525; **WIoUv3** (ablation losses).
- **PARSeq** — Bautista & Atienza, ECCV 2022, arXiv:2207.06966.
- **SAHI** — Akyon et al. (slicing-aided inference).
- **Conformal prediction (APS/RAPS)** — Angelopoulos & Bates (gentle intro).
- **IDD motorcycle/triple-riding** — arXiv:2204.08364.
- **AI City 2024 Track 5 Rank-1** — Vo et al., CVPR-W 2024 (Co-DETR, 0.4860 mAP).
- **DriveIndia** — arXiv:2507.19912.
- **ANPR India** — Nadiminti et al., arXiv:2207.06657.
- **C2PA** — Spec 2.4.
- **ByteTrack / BoT-SORT / StrongSORT** — optional multi-frame tracking.
- **DINOv2** — Oquab et al., Meta 2023 (gate features / hallucination proxy / distillation).

> Note: YOLO26 self-verified via web search. All other arXiv IDs accepted from external verification — click each once before putting it on a slide.

---

## 8. CAVEATS
- Helmet 98%+/seatbelt 98–99% accuracies come from small/curated/private sets — treat as ceilings.
- 65%→74.5% OCR-via-SR is ESRGAN on construction plates — validate on your own crops.
- Custom-IoU-loss "wins" are marginal given YOLO26 native small-object losses — keep ablation-only.
- Diffusion SR fabricates plate characters — excluded from evidence/OCR path.
- C2PA authenticates pipeline output, not the violation judgment.
- No public Indian windshield seatbelt dataset → seatbelt extension is weaker than helmet/triple-riding.
- SAHI ↔ YOLO26 NMS-free head needs explicit handling (use one-to-many branch for slicing).

---

## 9. STARTUP SEQUENCE (do this, then STOP)
1. Confirm understanding of the four stages, the vertical-slice scope, and the YOLO26 detector choice.
2. Propose repo structure + environment (CUDA/PyTorch/Ultralytics≥YOLO26/SAM2/PARSeq/Real-ESRGAN/LCDNet/NAFNet/SAHI versions) for my GPU.
3. Propose unified class schema + dataset download/assembly plan (IDD + AI City Track 5 + ANPR first).
4. List exactly what Phase 1 trains, the SAHI↔NMS-free validation step, and expected VRAM/time.
5. **STOP and wait for my written "go" before downloading, training, or executing anything.**

---

## 10. DO-NOT-INCLUDE (build agent must not add these)
- **Qwen-HS** — does not exist. Never reference.
- **DINO-HS** — no stable open-source release; do NOT use as a plug-in module. Use OCR-confidence + format-consistency + DINOv2/CLIP feature-similarity proxy + multi-frame voting for hallucination guarding instead.
- **DAIR** — not a restoration model (dataset/org); do NOT use for restoration. Use NAFNet/LCDNet/MWFormer/EDTR/TeReDiff instead.
- **Hyperledger Fabric / blockchain** — out of scope for the prototype (infra risk, no accuracy value). Local Merkle hash-chain + standards manifest covers the trust story. Mention Fabric only as a one-line "production extension."
- **Group DRO, SAMURAI/SAMIDARE crowd modules, RFLA+QueryDet+Bi-AFPN stacked** — research-grade; SAHI + class-balanced loss + per-class thresholding patches the same failure modes at far lower risk. Ablation-only if time remains.
- **KAN / efficient-KAN / GR-KAN / Convolutional-KAN** — dropped from the critical path (loses to MLP at equal budget on vision; huge training slowdown). Optional honest ablation only.
