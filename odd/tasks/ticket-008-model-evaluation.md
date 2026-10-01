# Ticket #008 — Validación del Modelo y Reporte Final

## Objetivo

Ejecutar la fase de Evaluación (CRISP-DM, tarea Q2, hito H3) del modelo ViT
(`Anwarkh1/Skin_Cancer-Image_Classification`) con un script reproducible (`uv run ...`)
que mida desempeño por clase, sensibilidad de Melanoma, latencia en CPU y robustez, y
documentar los resultados en un reporte final en `docs/`. Cierra el issue #18.

## Problema

No existe código de evaluación ni datos de validación locales. El único dataset
autorizado (`marmal88/skin_cancer`, HAM10000) es el mismo con el que se entrenó el modelo,
y sus splits se solapan (hallazgo verificado el 2026-10-01 leyendo solo columnas de IDs):

- `test`: 1285 imágenes; 1025 (80 %) son el mismo `image_id` que en `train`.
- Solo 28 imágenes de `test` pertenecen a lesiones nunca vistas en `train`/`validation`
  (1 melanoma).
- `validation` se usó para selección del modelo (tarjeta del modelo, 5 épocas).

Decisión del usuario (opción 1): evaluar sobre `test` completo como **techo optimista**,
reportar aparte el subconjunto de 260 imágenes cuyo `image_id` no está en `train`, y
documentar explícitamente la contaminación. No se afirma generalización.

## Alcance autorizado

- Descargar al caché de Hugging Face (nunca al repositorio), con revisiones fijadas:
  - Modelo `Anwarkh1/Skin_Cancer-Image_Classification@e37ebda4a662db26d9221c78f6c72b1cb8736ce0`.
  - Procesador `google/vit-base-patch16-224-in21k`.
  - Split `test` de `marmal88/skin_cancer@bdd59e10860746202dfdc452e0ac3a9eaa25268d`
    (`data/test-00000-of-00001-61e7cf54bf274ae2.parquet`, 354 MB).
  - Lectura remota solo de las columnas `image_id`/`lesion_id` de `train` y `validation`
    para calcular el solapamiento.
- Crear módulos de evaluación bajo `src/skin_lesion_classifier/` con sus tests.
- Generar resultados (JSON + PNG de matriz de confusión) y el reporte
  `docs/REPORTE_VALIDACION_MODELO.md`.
- Añadir al `README.md` una sección de evaluación (Streamlit ya documentado en línea 54).

## Restricciones

- Sin pesos, datasets ni imágenes clínicas en git (`data/`, `models/` y `reports/` ignorados).
- Sin dependencias nuevas: métricas con `numpy`, figuras con `matplotlib`, parquet con
  `pyarrow` (ya presente por `transformers`), descarga con `huggingface_hub`.
- Reglas de `AGENTS.md`/`CONSTITUTION.md`: módulos `.py` bajo `src/`, una responsabilidad
  por módulo, sin `utils.py`, dependencias inyectadas, salidas `@dataclass(frozen=True)`,
  excepciones de dominio con mensaje en español y `raise ... from err`, type hints
  completos, `mypy --strict`, `ruff check` y `ruff format` limpios.
- Los tests unitarios nunca cargan el modelo real ni tocan la red (servicios falsos y
  parquet sintético en `tmp_path`).
- Artefactos técnicos en inglés; mensajes de dominio y reporte final en español.
- TDD: activo (fuente: precedente del ticket 009; no hay configuración declarada en el
  proyecto). Runner: `uv run pytest`. RED observado antes de cada implementación.

## Tareas

- [x] T008-1 Métricas puras (`evaluation_metrics.py`): matriz de confusión, precisión,
      recall, F1 y soporte por clase en orden `HAM10000_CODES`, exactitud global,
      comparación de recall `mel` contra la meta (> 90 %). Ruta: delegada (writer).
- [x] T008-2 Perturbaciones de robustez (`perturbations.py`): baja luminosidad y ruido
      gaussiano deterministas (semilla). Ruta: delegada (writer).
- [x] T008-3 Carga del split (`evaluation_dataset.py`): lectura del parquet fijado desde el
      caché HF, mapeo `dx` → código corto, conjunto de `image_id` de `train` para el
      subconjunto sin solapamiento. Ruta: delegada (writer).
- [x] T008-4 Ejecutor y CLI (`evaluation.py`, `uv run python -m
      skin_lesion_classifier.evaluation`): evalúa con `PredictionProvider` inyectado, mide
      latencia por imagen, ejecuta escenarios de robustez, guarda JSON + PNG y overlays
      Grad-CAM de muestra en `reports/evaluation/`. Ruta: delegada (writer).
- [x] T008-5 Ejecutar la evaluación real, revisar Grad-CAM cualitativamente, escribir
      `docs/REPORTE_VALIDACION_MODELO.md` y la sección de evaluación del `README.md`.
      Ruta: inline (ejecución + documentación).
- [x] T008-6 Checks completos y cierre del ticket.

## Criterios de aceptación

- [x] Recall de `mel` reportado y comparado con la meta > 90 %.
- [x] Latencia de inferencia en CPU reportada y comparada con la meta < 3 s por imagen.
- [x] Matriz de confusión y métricas por clase documentadas en `docs/`.
- [x] Resultados de robustez (baja luminosidad / ruido) documentados.
- [x] README indica cómo ejecutar la aplicación Streamlit.
- [x] `uv run pytest`, `uv run ruff check .` y `uv run ruff format --check .` pasan.

## Checks aplicables

- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy --strict src tests`

## Progreso

Rama: `feature/model-evaluation` (desde `main` @ `669e68d`).

Entrega: pronóstico ~900 líneas (real ~1500 tras T008-4, supera ~400). Estrategia
elegida por el usuario: `single-pr` (un solo PR hacia `main`, por la fecha límite).

Espejo Engram `odd/ticket-008-model-evaluation/tasks`: **pendiente** (servidor engram no
conectado en esta sesión).

Commits por tarea (rama `feature/model-evaluation`): T008-1 `08b7ee1`, T008-2 `c86720a`,
T008-3 `a665666`, T008-4 `f56eedb` (incluye `revision` opcional en `load_inference_service`
y `/reports/` en `.gitignore`). Dependencias: `20c6831` (`pyarrow` y `huggingface-hub`
declarados en el grupo `api`, regla 5 de `AGENTS.md`). T008-5/T008-6: commit del reporte.

Resultado (2026-10-01): recall `mel` 87,5 % (meta > 90 %: **no cumplida**); latencia p95
0,054 s (meta < 3 s: cumplida); baja luminosidad hunde el recall `mel` a 9,7 %; Grad-CAM
coherente en 1 de 6 muestras. Detalle en `docs/REPORTE_VALIDACION_MODELO.md`.

### Evidencia de verificación

- `uv run pytest`: 128 passed, 2 deselected.
- `uv run pytest -m integration`: 2 passed (modelo real).
- `uv run ruff check .`: All checks passed.
- `uv run ruff format --check .`: 43 files already formatted.
- `uv run mypy --strict src tests`: no issues found in 28 source files.
- Evaluación real: `uv run python -m skin_lesion_classifier.evaluation`, exit 0, ≈ 6 min.
- gga: aprobó los commits de T008-1..4 sin bloqueos.
- Revisión nativa (RDD), rango `669e68d..0b0e786`: riesgo alto (`pyproject.toml`),
  consentimiento otorgado por el usuario, 4 lentes, **aprobada** y reconocida
  (lineage `review-6dd9b7fb91a44c8e`, autoridad consumida). 0 hallazgos bloqueantes.

### Seguimiento (hallazgos no bloqueantes de la revisión, trabajo posterior)

- R2-001: el CLI no calcula el solapamiento por `lesion_id` ni contra `validation` (las
  cifras de 28 imágenes / 1 melanoma del reporte vienen de un análisis puntual; el reporte
  ya lo aclara).
- R3/R4: el solapamiento y Grad-CAM corren después de los 3 escenarios y `metrics.json` se
  escribe al final; un fallo de red tardío pierde ≈ 6 min. Mover `fetch_train_ids` al inicio
  o escribir resultados parciales.
- R4: `main` no captura `DatasetLoadingError`/`ModelLoadingError` (traceback crudo sin red).
- R3: `--limit 0` o negativo evalúa 0 imágenes y reporta la meta de latencia como cumplida.
- R3: el ruido usa la misma semilla para todas las imágenes (un solo patrón de ruido).
- R3: `in_train` asume `image_id` únicos en `test`.
- R2: `targets` duplicado en `metrics.json`; factor 0,4 de oscurecimiento definido dos veces.

### Desviaciones respecto al diseño original

- La meta de latencia se evalúa sobre p95 (no la media) para ser más exigente.
- `.gitignore` usa `/reports/` anclado a la raíz para no ignorar `tests/reports/`.
- El procesador no tiene revisión fijada (el cargador no la admite); queda como próximo paso.
- `pyarrow` y `huggingface-hub` se declararon explícitamente (antes eran transitivas), en
  contra de la restricción inicial "sin dependencias nuevas", para cumplir la regla 5.
- Los overlays Grad-CAM no se versionan (imágenes clínicas); se regeneran con el CLI.
