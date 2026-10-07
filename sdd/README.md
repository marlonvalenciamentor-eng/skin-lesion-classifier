# Spec-Driven Development (SDD)

This folder consolidates the specification of every ticket in one place, following the
flow mandated by [`CONSTITUTION.md`](../CONSTITUTION.md):
**Explore → Propose → Spec → Implement & Verify**.

> **Consolidation note (2026-10-06):** these files were assembled on 2026-10-06 from
> artifacts that already existed — the ticket specs and acceptance criteria in
> [`docs/propuesta.md`](../docs/propuesta.md) and the GitHub issues, the task documents
> in [`odd/tasks/`](../odd/tasks/), the merged pull requests and the validation report.
> No specification was rewritten after the fact; deviations are recorded as found.

## Flow → artifacts

| SDD phase | Artifact |
|---|---|
| Explore | Mind maps in [`docs/`](../docs/), [`docs/CONTEXTO_SESION_FUTURA.md`](../docs/CONTEXTO_SESION_FUTURA.md) |
| Propose | [`docs/propuesta.md`](../docs/propuesta.md) (objectives, schedule, critical path) |
| Spec | One file per ticket in this folder; architecture contracts in [`CONSTITUTION.md`](../CONSTITUTION.md) |
| Implement | One pull request per ticket (linked in each spec) |
| Verify | `pytest` suite (100 tests, 94% coverage on `src/`), CI on every PR, [`docs/REPORTE_VALIDACION_MODELO.md`](../docs/REPORTE_VALIDACION_MODELO.md) |

## Ticket index

GitHub issues and pull requests share one counter, so ticket numbers differ from issue numbers.

| Ticket | Spec | Issue | Implementation | Status |
|---|---|---|---|---|
| #001 | [Base architecture and uv environment](001-base-architecture-uv.md) | #4 | `10d4deb` | Done |
| #002 | [Image ingestion and preprocessing](002-image-preprocessing.md) | #5 | PR #11 | Done with deviation |
| #003 | [ViT model and inference service](003-vit-inference-service.md) | #6 | PR #11 | Done |
| #004 | [Grad-CAM explainability](004-gradcam-xai.md) | #7 | PR #14 | Done |
| #005 | [Orchestrating facade](005-facade.md) | #8 | PR #15, #17 | Done with deviation |
| #006 | [Streamlit web interface](006-streamlit-ui.md) | #9 | PR #16, #17, #26 | Done |
| #007 | [Automated tests and QA](007-tests-qa.md) | #10 | Across all PRs | Done |
| #008 | [Model validation and final report](008-model-validation.md) | #18 | PR #23, #24, #25 | Done — melanoma target not met |
| #009 | [Inference API, Pydantic contracts and Docker](009-inference-api-docker.md) | #19 | PR #20 | Superseded by PR #26 |
