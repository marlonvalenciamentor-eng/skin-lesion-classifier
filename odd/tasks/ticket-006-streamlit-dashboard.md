# Ticket #006 — Interfaz Web Médica Reactiva en Streamlit

## Objetivo

Construir la interfaz de usuario en Streamlit que consume la fachada `DermatologyDiagnosticFacade` (Ticket #005) para mostrar diagnóstico clínico y explicabilidad Grad-CAM, sin importar jamás `torch` ni `transformers`.

## Problema

El backend (carga, inferencia, Grad-CAM y fachada) está completo, pero no existe un punto de interacción para el usuario final. Falta la capa de presentación que cargue imágenes dermatoscópicas, muestre el mapa de calor y las probabilidades, y maneje errores de forma amigable.

## Alcance autorizado

- Crear `app.py` en la raíz del proyecto (UI Streamlit).
- Crear `src/skin_lesion_classifier/labels.py` con el mapeo de taxonomía clínica (nombre legible en español y severidad por clase HAM10000).
- Crear `tests/test_labels.py` (mapeo, sin red) y `tests/test_app.py` (smoke con `AppTest`, sin cargar el modelo).
- Mantener la implementación en la rama `feature/streamlit-dashboard` (basada en `feature/facade-integrator` para heredar la fachada).
- No descargar ni incorporar pesos o datasets al repositorio.

## Restricciones

- La capa de presentación **nunca** importa `torch` ni `transformers`; consume únicamente `DiagnosticResult` de la fachada.
- Cargar la fachada con `st.cache_resource` para evitar recargar el ViT en cada rerun.
- Manejo amigable de errores: capturar `InferenceError`, `GradCAMError` y `ModelLoadingError` y mostrarlos con `st.error`.
- Incluir descargo de responsabilidad médica (CDSS) visible.
- Artefactos técnicos en inglés salvo el encabezado institucional y mensajes de dominio/UI en español.
- Usar `uv` para las verificaciones requeridas; respetar Ruff (línea 100) y mypy strict en `src`.

## Tareas

- [x] T006-1 Implementar `labels.py` con `LABEL_TO_NAME`, `LABEL_TO_SEVERITY` y `SEVERITY_BADGE`.
- [x] T006-2 Implementar `app.py` con carga de archivos, muestras pre-cargadas, dos columnas (original vs Grad-CAM), badge de severidad, métricas, barras de probabilidad y disclaimer.
- [x] T006-3 Crear `tests/test_labels.py` y `tests/test_app.py` (smoke AppTest).
- [x] T006-4 Ejecutar verificaciones (pytest, ruff check, ruff format, mypy strict) y registrar evidencia.

## Criterios de aceptación

- La UI permite subir PNG/JPG/JPEG y seleccionar muestras ISIC de `data/samples/`.
- Layout de dos columnas: imagen original y mapa térmico Grad-CAM.
- Diagnóstico Top-1 con badge de severidad (maligno/precanceroso/benigno).
- Barras de probabilidad para las 7 clases HAM10000 con nombre legible.
- Latencia y descargo de responsabilidad médica visibles.
- Errores de entrada o de modelo se muestran con mensajes claros, sin traceback.
- `app.py` no importa `torch` ni `transformers`.

## Checks aplicables

- `uv run pytest -v`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy --strict src tests`

## Progreso

- Rama: `feature/streamlit-dashboard` (desde `feature/facade-integrator` @ 2d80c53).
- T006-1: `labels.py` con 7 clases HAM10000, severidad (maligno/precanceroso/benigno) y estilos de badge (color + icono Material).
- T006-2: `app.py` con `st.cache_resource` para el ViT, `st.file_uploader`, selector de muestras `data/samples/`, dos columnas (original vs Grad-CAM), badge de severidad, métricas de confianza/latencia, barras de probabilidad y disclaimer CDSS. No importa `torch` ni `transformers`.
- T006-3: `tests/test_labels.py` (5 pruebas) y `tests/test_app.py` (smoke AppTest sin cargar modelo).
- T006-4: Evidencia de verificaciones:
  - `uv run pytest -v` → 43 passed, 2 deselected.
  - `uv run ruff check .` → All checks passed.
  - `uv run ruff format --check .` → limpio.
  - `uv run mypy --strict src tests` → Success (14 source files).
  - Smoke end-to-end: `ISIC_0000000.jpg` → `nv` (Nevus melanocítico, benigno, 99.7%), heatmap (224,224), overlay RGB.
- Fix transversal (Ticket #005): `facade.py` ahora traduce código HAM10000 a etiqueta del modelo (`CODE_TO_MODEL_LABEL`) antes de `explain()`. Se detectó con el smoke end-to-end.
- Commit work-unit: pendiente.