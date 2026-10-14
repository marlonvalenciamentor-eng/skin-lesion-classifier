# Ticket #002 — Image Ingestion and Preprocessing

| Field | Value |
|---|---|
| Issue | #5 (closed) |
| Owner | Miguel Ángel Ortiz Roldán |
| Estimate | 3 SP · due 2026-09-22 |
| Source spec | [`docs/propuesta.md`](../docs/propuesta.md) |

## Spec
Implement `src/preprocessing.py`, accepting PIL images or byte buffers, validating integrity and
returning the normalized ViT tensor (shape `[1, 3, 224, 224]`).

## Acceptance criteria
- [x] Rejects unsupported or corrupt files.
- [x] Returns a normalized 32-bit float tensor.
- [ ] `test_preprocessing.py` passes with `pytest`. **Not met as written — see deviation.**

## Implementation
PR #11. Preprocessing lives in `InferenceService.predict()` in
[`inference.py`](../src/skin_lesion_classifier/inference.py): RGB conversion, the ViT image
processor (resize + normalization to `[1, 3, 224, 224]`) and domain errors (`InferenceError`)
for invalid or empty images. The UI shows a friendly error for unreadable uploads (`app.py`).

## Deviation
No separate `preprocessing.py` module or `test_preprocessing.py` was created. The Hugging Face
image processor already performs the normalization the model expects, so preprocessing was kept
inside the inference service; it is covered by [`tests/test_inference.py`](../tests/test_inference.py)
(89% coverage on `inference.py`).
