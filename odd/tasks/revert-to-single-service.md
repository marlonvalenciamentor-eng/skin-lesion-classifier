# Revertir a servicio único (monólito) + Docker

## Objetivo

Revertir la separación API (FastAPI) + UI (Streamlit por HTTP) para volver al enfoque simple: la UI consume la fachada `DermatologyDiagnosticFacade` directamente, en un único proceso y un único contenedor Docker. Se conserva la evaluación del modelo.

## Problema

La arquitectura se separó en dos servicios (API + UI por HTTP), añadiendo fricción operativa innecesaria para el alcance del proyecto (un modelo, una UI demostrativa). La fachada ya resolvía el desacoplamiento UI↔modelo en un solo proceso.

## Alcance autorizado

- Revertir `app.py` a la versión que consume la fachada directamente.
- Eliminar `api.py`, `api_client.py`, `schemas.py` y sus tests (`test_api.py`, `test_api_client.py`, `test_schemas.py`).
- Simplificar `pyproject.toml`: quitar `fastapi`, `uvicorn`, `httpx`, `pydantic`, `python-multipart`.
- Reemplazar `docker/api.Dockerfile` + `docker/ui.Dockerfile` por un único Dockerfile (app + modelo + Streamlit), y `docker-compose.yml` a un solo servicio.
- Actualizar `README.md`.
- Conservar la evaluación (`evaluation*.py`, `perturbations.py`) y los fixes recientes.
- Mantener la implementación en la rama `feature/single-service`.

## Restricciones

- `app.py` no importa `torch` ni `transformers` directamente: consume la fachada.
- La evaluación sigue usando `pyarrow` y `huggingface-hub` (se conservan como dependencias).
- Un solo contenedor: el modelo se descarga/cachea por volumen, no se hornea en la imagen.
- Usar `uv` para las verificaciones; Ruff (línea 100) y mypy strict en `src`.

## Tareas

- [x] T1 Revertir `app.py` a fachada directa.
- [x] T2 Eliminar `api.py`, `api_client.py`, `schemas.py` + tests de API.
- [x] T3 Simplificar `pyproject.toml` (quitar deps de API).
- [x] T4 Dockerfile único + `docker-compose.yml` de un servicio.
- [x] T5 Actualizar `README.md`.
- [x] T6 `uv sync` + verificaciones (pytest, ruff, mypy).

## Criterios de aceptación

- `app.py` usa `DermatologyDiagnosticFacade` y no `api_client`.
- No quedan referencias a `api.py`, `api_client.py` ni `schemas.py`.
- `pyproject.toml` no declara `fastapi`, `uvicorn`, `httpx`, `pydantic` ni `python-multipart`.
- Un único Dockerfile levanta la app con el modelo.
- `uv run pytest -v`, `uv run ruff check .`, `uv run mypy --strict src tests` pasan.

## Checks aplicables

- `uv run pytest -v`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy --strict src tests`

## Progreso

- Rama: `feature/single-service` (desde `main` @ 39f1c0d).
- T1: `app.py` restaurado a la versión que consume la fachada directamente (sin `api_client`).
- T2: eliminados `api.py`, `api_client.py`, `schemas.py` + `test_api.py`, `test_api_client.py`, `test_schemas.py`; `test_app.py` restaurado a la versión de fachada directa.
- T3: `pyproject.toml` sin `fastapi`, `uvicorn`, `httpx`, `pydantic`, `python-multipart`; `uv.lock` regenerado (83 packages).
- T4: `docker/Dockerfile` único (Streamlit + fachada + modelo); `docker-compose.yml` de un servicio.
- T5: `README.md` actualizado (sin API, un solo comando, un solo contenedor).
- T6: Evidencia de verificaciones:
  - `uv run pytest -v` → 98 passed, 2 deselected.
  - `uv run ruff check .` → All checks passed.
  - `uv run ruff format --check .` → 38 files formatted.
  - `uv run mypy --strict src tests` → Success (22 source files).
  - Smoke end-to-end: `ISIC_0000000.jpg` → `nv` (Nevus melanocítico, benigno, 99.7%).
- Commit work-unit: pendiente.