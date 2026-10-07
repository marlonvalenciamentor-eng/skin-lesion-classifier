# Ticket #008 — Model Validation and Final Report

| Field | Value |
|---|---|
| Issue | #18 (closed) |
| Owner | Marlon Valencia Velosa |
| Estimate | 5 SP · due 2026-10-01 |
| Source spec | Issue #18 · detailed tasks: [`odd/tasks/ticket-008-model-evaluation.md`](../odd/tasks/ticket-008-model-evaluation.md) |

## Spec
Run the CRISP-DM evaluation phase: measure the ViT on HAM10000, check the success metrics and
document the results in a final report.

## Acceptance criteria
- [x] Melanoma recall reported and compared with the > 90% target.
- [x] CPU latency reported and compared with the < 3 s target.
- [x] Confusion matrix and per-class metrics documented in `docs/`.
- [x] Robustness results (low light / noise) documented.
- [x] README explains how to run the Streamlit app.
- [x] `pytest`, `ruff check` and `ruff format --check` pass.

## Implementation
PR #23, fixes in PR #24 and #25: `evaluation*.py`, `perturbations.py`; outputs in
[`docs/evaluation/`](../docs/evaluation/).

## Outcome
Report: [`REPORTE_VALIDACION_MODELO.md`](../docs/REPORTE_VALIDACION_MODELO.md).
- Melanoma recall **87.5%** (126/144) — **target not met**.
- Latency 0.054 s p95 — target met.
- Known limitation: the test split overlaps with training data, so results are optimistic.
