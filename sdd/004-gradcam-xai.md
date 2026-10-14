# Ticket #004 — Grad-CAM Visual Explainability for Transformers

| Field | Value |
|---|---|
| Issue | #7 (closed) |
| Owner | Miguel Ángel Ortiz Roldán |
| Estimate | 8 SP (critical task) · due 2026-09-26 |
| Source spec | [`docs/propuesta.md`](../docs/propuesta.md) · detailed tasks: [`odd/tasks/ticket-004-gradcam-xai.md`](../odd/tasks/ticket-004-gradcam-xai.md) |

## Spec
Extract activations from the last self-attention layer of the ViT, compute gradients with respect
to the predicted class and overlay the heatmap on the original image.

## Acceptance criteria
- [x] Produces a superimposed heatmap as an RGB image compatible with the UI.
- [ ] Does not block the main execution thread. **Not verified** — explanation runs synchronously inside `diagnose()`.

## Implementation
PR #14: [`gradcam.py`](../src/skin_lesion_classifier/gradcam.py) — hook on
`vit.encoder.layer[-1].layernorm_before`, `[CLS]` token excluded, 196 patches reshaped to a
14×14 grid, ReLU, bilinear upsampling to 224×224 and JET overlay. Hooks are always released in `finally`.

## Verification
`tests/test_gradcam.py` (85% coverage on `gradcam.py`).
