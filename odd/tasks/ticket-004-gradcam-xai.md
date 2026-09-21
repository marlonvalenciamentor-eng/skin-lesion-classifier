# Ticket #004 — Grad-CAM XAI para Vision Transformer

## Objetivo

Implementar un componente de explicabilidad visual Grad-CAM para el ViT de clasificación de lesiones cutáneas, manteniendo Clean Architecture, responsabilidad única y tipado estricto.

## Problema

El servicio de inferencia produce una clase HAM10000, pero todavía no expone qué regiones espaciales de la imagen influyeron en el resultado.

## Alcance autorizado

- Crear `src/skin_lesion_classifier/gradcam.py`.
- Crear `tests/test_gradcam.py` con pruebas unitarias desacopladas de red.
- Mantener la implementación en la rama `feature/gradcam-xai`.
- No descargar ni incorporar pesos o datasets al repositorio.

## Restricciones

- Hook en `model.vit.encoder.layer[-1].layernorm_before` o capa equivalente validada.
- Excluir `[CLS]`, convertir los 196 parches a `14x14` y producir un mapa interpolado a la resolución de entrada.
- Retirar hooks en `finally`, incluso ante excepciones.
- Usar `uv` para las verificaciones requeridas.
- Artefactos técnicos en inglés salvo el encabezado institucional y mensajes de dominio, que siguen la convención existente en español.

## Tareas

- [x] T004-1 Implementar `GradCAMError`, `GradCAMResult` y `ViTGradCAM` con validaciones y limpieza segura de hooks.
- [x] T004-2 Crear pruebas unitarias AAA para dimensiones, rango, imagen superpuesta, errores y limpieza.
- [x] T004-3 Ejecutar pytest, ruff, formato, mypy estricto e integración; las cinco verificaciones solicitadas pasaron, incluida `uv run pytest -m integration -v` con 1 passed y 23 deselected.

## Criterios de aceptación

- `GradCAMResult` contiene `heatmap`, `superimposed_image` y `target_class`.
- El heatmap es `np.ndarray`, tiene resolución de entrada y valores en `[0, 1]`.
- La superposición es una imagen PIL RGB válida.
- La clase objetivo puede resolverse por etiqueta o por predicción top-1.
- Los errores operativos se exponen como `GradCAMError`.
- Todos los hooks registrados son eliminados en éxito y error.
- Las verificaciones solicitadas por el ticket pasan.

## Checks aplicables

- `uv run pytest -v --junitxml=tests/reports/unit_tests.xml`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy --strict src tests`
- `uv run pytest -m integration -v`

## Progreso

- Rama: `feature/gradcam-xai`.
- T004-1, T004-2 y las cinco verificaciones solicitadas completados; la integración terminó correctamente.
- Evidencia: implementación inyecta modelo/procesador, usa `layernorm_before`, excluye `[CLS]`, interpola bilinealmente y retira handles en `finally`.
- Evidencia de pruebas: `uv run pytest -q tests/test_gradcam.py` → 6 passed; `uv run pytest -q` → 23 passed, 1 deselected.
- Evidencia de calidad: `uv run ruff check src/skin_lesion_classifier/gradcam.py tests/test_gradcam.py` → All checks passed; `uv run ruff format --check src/skin_lesion_classifier/gradcam.py tests/test_gradcam.py` → 2 files already formatted; `uv run mypy --strict src tests` → Success: no issues found in 9 source files.
- Integración `uv run pytest -m integration -v` → 1 passed, 23 deselected.
- Commit work-unit: `6d64660 feat(xai): add ViT Grad-CAM explanations`.
- Próximo paso: entrega del resultado bajo la política ordinaria del repositorio.
