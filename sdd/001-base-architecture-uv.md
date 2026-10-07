# Ticket #001 — Base Architecture and uv Environment

| Field | Value |
|---|---|
| Issue | #4 (closed) |
| Owner | Marlon Valencia Velosa |
| Estimate | 2 SP · due 2026-09-20 |
| Source spec | [`docs/propuesta.md`](../docs/propuesta.md) |

## Spec
Initialize a deterministic environment with `uv`, configure `pyproject.toml` and `.gitignore`,
and define the Clean Architecture folder layout (`src/`, `tests/`, `docs/`).

## Acceptance criteria
- [x] `uv sync` runs without broken dependencies on Linux/macOS.
- [x] No `.safetensors` files or cache folders are tracked in Git.

## Implementation
Commit `10d4deb` (2026-09-17), "scaffold project with uv, pytest, ruff and CI pipeline", made before the PR flow started.

## Verification
- `uv sync --locked` runs in CI on every PR; local `uv sync` verified on macOS on 2026-10-06.
- `models/` is ignored in `.gitignore`; committing weights is a blocking rule in [`AGENTS.md`](../AGENTS.md).
