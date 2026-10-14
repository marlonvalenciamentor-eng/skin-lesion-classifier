# Ticket #007 — Automated Tests and Quality Assurance

| Field | Value |
|---|---|
| Issue | #10 (closed) |
| Owner | Miguel Ángel Ortiz Roldán |
| Estimate | 5 SP · due 2026-09-30 |
| Source spec | [`docs/propuesta.md`](../docs/propuesta.md) |

## Spec
Unit and integration tests in `tests/` covering method contracts, exception handling and response times.

## Acceptance criteria
- [x] Test coverage above 80% on `src/` modules.
- [x] `uv run pytest` passes.

## Implementation
Tests were delivered with each feature PR (AAA structure, no network; real-model tests are
marked `integration`). CI runs `ruff check`, `ruff format --check` and `pytest` on every PR.

## Verification (2026-10-06)
`uv run pytest --cov=src/skin_lesion_classifier`: **100 passed, 94% coverage** (704 statements, 44 missed).
