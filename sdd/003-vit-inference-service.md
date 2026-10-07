# Ticket #003 — ViT Model and Inference Service

| Field | Value |
|---|---|
| Issue | #6 (closed) |
| Owner | Marlon Valencia Velosa |
| Estimate | 5 SP · due 2026-09-24 |
| Source spec | [`docs/propuesta.md`](../docs/propuesta.md) |

## Spec
Load `Anwarkh1/Skin_Cancer-Image_Classification` with `ViTForImageClassification`, return the
probability vector for the 7 classes and the top-1 diagnosis with its confidence.

## Acceptance criteria
- [x] CPU inference takes less than 3.0 seconds.
- [x] Correct label mapping (`mel`, `nv`, `bcc`, `akiec`, `bkl`, `df`, `vasc`).

## Implementation
PR #11: [`inference.py`](../src/skin_lesion_classifier/inference.py) and
[`model_loader.py`](../src/skin_lesion_classifier/model_loader.py) (offline-first loading and
7-class integrity validation).

## Verification
- Latency: 0.054 s p95 on CPU ([`REPORTE_VALIDACION_MODELO.md`](../docs/REPORTE_VALIDACION_MODELO.md), section 7).
- Tests: `test_inference.py`, `test_model_loader.py`, `test_labels.py`.
