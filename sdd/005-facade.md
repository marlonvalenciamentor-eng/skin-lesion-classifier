# Ticket #005 — Orchestrating Facade

| Field | Value |
|---|---|
| Issue | #8 (closed) |
| Owner | Miguel Ángel Ortiz Roldán & Marlon Valencia Velosa |
| Estimate | 3 SP · due 2026-09-27 |
| Source spec | [`docs/propuesta.md`](../docs/propuesta.md) · detailed tasks: [`odd/tasks/ticket-005-facade-integrator.md`](../odd/tasks/ticket-005-facade-integrator.md) |

## Spec
Design `DermatologyDiagnosticFacade` exposing a single entry point that orchestrates reading,
preprocessing, inference and Grad-CAM.

## Acceptance criteria
- [x] The UI contains no `torch` or `transformers` imports.
- [x] The facade returns a structured `DiagnosticResult` with class, confidence and heatmap.

## Implementation
PR #15, integrated in PR #17: [`facade.py`](../src/skin_lesion_classifier/facade.py). Includes the
fix that translates HAM10000 codes to model labels before calling `explain()`.

## Deviation
The spec named `src/integrator.py` and `diagnose_image(file_path)`; the implementation is
`facade.py` with `diagnose(image)`, which takes an in-memory image so the UI never writes uploads to disk.

## Verification
`app.py` has no `torch`/`transformers` imports (blocking rule in [`AGENTS.md`](../AGENTS.md)); `tests/test_facade.py` (100% coverage).
