# Ticket #009 — Inference API, Pydantic Contracts and Docker

| Field | Value |
|---|---|
| Issue | #19 (closed) |
| Owner | Marlon Valencia Velosa |
| Estimate | 8 SP · due 2026-10-01 |
| Source spec | Issue #19 · detailed tasks: [`odd/tasks/ticket-009-inference-api-docker.md`](../odd/tasks/ticket-009-inference-api-docker.md) |

## Spec
Split the solution into a FastAPI inference service and a Streamlit UI consuming it over HTTP,
with Pydantic v2 contracts and separate Docker images orchestrated by `docker compose`.

## Acceptance criteria (as delivered in PR #20)
- [x] `POST /predict` and `GET /health` with validated Pydantic schemas and a single `ErrorResponse` envelope.
- [x] The UI consumes the API without importing `torch`, `transformers` or the facade.
- [x] `docker compose build` builds both images in CI.
- [x] Unit tests for contracts, API and client without network; coverage ≥ 80%.
- [x] `pytest`, `ruff` and `mypy --strict` pass.

## Status: superseded
PR #26 reverted to a single service. The facade already decoupled the UI from the model, and two
processes added operational cost without benefit. Decision record:
[`odd/tasks/revert-to-single-service.md`](../odd/tasks/revert-to-single-service.md). Docker was kept
as a single image ([`docker/Dockerfile`](../docker/Dockerfile)), built in CI on every PR.
