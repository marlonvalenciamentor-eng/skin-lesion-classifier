# Ticket #005 — Fachada Orquestadora (`DermatologyDiagnosticFacade`)

## Objetivo

Crear la fachada orquestadora que centraliza el flujo completo de diagnóstico: carga del servicio de inferencia, predicción clínica y explicabilidad Grad-CAM, exponiendo un único método `diagnose(image) -> DiagnosticResult` para que la capa de presentación (Streamlit) nunca importe `torch` ni `transformers`.

## Problema

`model_loader`, `inference` y `gradcam` son módulos independientes y correctos, pero la futura interfaz web no debe conocerlos ni importar librerías de bajo nivel. Falta el punto único de entrada que orquesta el pipeline completo y devuelve un DTO limpio e inmutable.

## Alcance autorizado

- Crear `src/skin_lesion_classifier/facade.py`.
- Crear `tests/test_facade.py` con pruebas unitarias desacopladas de red.
- Exponer `model` y `processor` como propiedades read-only en `InferenceService` (cambio mínimo en `src/skin_lesion_classifier/inference.py`) para que la fachada construya el `ViTGradCAM` con los mismos componentes inyectados, sin duplicar la carga.
- Mantener la implementación en la rama `feature/facade-integrator`.
- No descargar ni incorporar pesos o datasets al repositorio.

## Restricciones

- `DiagnosticResult` debe ser un DTO `@dataclass(frozen=True)` sin tipos de bajo nivel (sin `torch.Tensor` ni objetos `transformers`).
- La fachada aplica inyección de dependencias: acepta `inference_service` y `explainer` opcionales y hace lazy-load con `load_inference_service()` si no se proveen.
- Por defecto, Grad-CAM explica la clase Top-1 predicha por la inferencia (consistencia clínica); se admite `target_class` explícito.
- Los errores de dominio (`InferenceError`, `GradCAMError`) se propagan tal cual; la fachada no añade una capa de error innecesaria.
- Artefactos técnicos en inglés salvo el encabezado institucional y mensajes de dominio, que siguen la convención existente en español.
- Usar `uv` para las verificaciones requeridas.

## Tareas

- [x] T005-1 Exponer `model` y `processor` como propiedades read-only en `InferenceService`.
- [x] T005-2 Implementar `DiagnosticResult` y `DermatologyDiagnosticFacade` en `facade.py`.
- [x] T005-3 Crear `tests/test_facade.py` con pruebas AAA desacopladas de red (combinación, Top-1 por defecto, clase explícita, propagación de errores, lazy-load).
- [x] T005-4 Ejecutar las verificaciones (pytest, ruff check, ruff format, mypy strict) y registrar evidencia.

## Criterios de aceptación

- `DermatologyDiagnosticFacade.diagnose(image) -> DiagnosticResult` existe y orquesta predicción + Grad-CAM en un solo llamado.
- `DiagnosticResult` contiene `label`, `confidence`, `probabilities`, `heatmap`, `superimposed_image` y `target_class`, sin tipos de bajo nivel.
- La fachada construye su `ViTGradCAM` a partir del servicio cargado (sin duplicar carga) o acepta un explainer inyectado.
- Las pruebas unitarias pasan sin conectividad ni pesos reales.
- `uv run pytest -v`, `uv run ruff check .`, `uv run ruff format --check .` y `uv run mypy --strict src tests` pasan.

## Checks aplicables

- `uv run pytest -v --junitxml=tests/reports/unit_tests.xml`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy --strict src tests`

## Progreso

- Rama: `feature/facade-integrator` (actualizada a `576ac53` con `git merge --ff-only main`).
- T005-1: `InferenceService` ahora expone `model` y `processor` como propiedades read-only (atributos anotados a nivel de clase para mypy strict).
- T005-2: `facade.py` implementa `DiagnosticResult` (frozen, sin tipos torch/transformers) y `DermatologyDiagnosticFacade` con inyección de dependencias, lazy-load vía `load_inference_service()`, construcción del `ViTGradCAM` desde los componentes del servicio y `target_class` por defecto = Top-1 predicho.
- T005-3: `tests/test_facade.py` con 7 pruebas AAA desacopladas de red (fakes sin torch).
- T005-4: Evidencia de verificaciones:
  - `uv run pytest -v` → 36 passed, 2 deselected (incluye 7 nuevas de test_facade).
  - `uv run ruff check .` → All checks passed.
  - `uv run ruff format .` → 2 files reformatted; `--check` limpio.
  - `uv run mypy --strict src tests` → Success: no issues found in 11 source files.
  - `uv run pytest -m integration -v` → 2 passed (smoke test con modelo local).
- Commit work-unit: pendiente.