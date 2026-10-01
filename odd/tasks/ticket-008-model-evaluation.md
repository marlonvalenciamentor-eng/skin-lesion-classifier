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
- [ ] T008-3 Carga del split (`evaluation_dataset.py`): lectura del parquet fijado desde el
      caché HF, mapeo `dx` → código corto, conjunto de `image_id` de `train` para el
      subconjunto sin solapamiento. Ruta: delegada (writer).
- [ ] T008-4 Ejecutor y CLI (`evaluation.py`, `uv run python -m
      skin_lesion_classifier.evaluation`): evalúa con `PredictionProvider` inyectado, mide
      latencia por imagen, ejecuta escenarios de robustez, guarda JSON + PNG y overlays
      Grad-CAM de muestra en `reports/evaluation/`. Ruta: delegada (writer).
- [ ] T008-5 Ejecutar la evaluación real, revisar Grad-CAM cualitativamente, escribir
      `docs/REPORTE_VALIDACION_MODELO.md` y la sección de evaluación del `README.md`.
      Ruta: inline (ejecución + documentación).
- [ ] T008-6 Checks completos y cierre del ticket.

## Criterios de aceptación

- [ ] Recall de `mel` reportado y comparado con la meta > 90 %.
- [ ] Latencia de inferencia en CPU reportada y comparada con la meta < 3 s por imagen.
- [ ] Matriz de confusión y métricas por clase documentadas en `docs/`.
- [ ] Resultados de robustez (baja luminosidad / ruido) documentados.
- [ ] README indica cómo ejecutar la aplicación Streamlit.
- [ ] `uv run pytest`, `uv run ruff check .` y `uv run ruff format --check .` pasan.

## Checks aplicables

- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy --strict src tests`

## Progreso

Rama: `feature/model-evaluation` (desde `main` @ `669e68d`).

Entrega: pronóstico ~900 líneas (supera ~400); la estrategia de PR (`stacked-to-main` o
`feature-branch-chain`) queda pendiente de confirmación del usuario. Los commits por
unidad de trabajo sirven para ambas.

Espejo Engram `odd/ticket-008-model-evaluation/tasks`: **pendiente** (servidor engram no
conectado en esta sesión).

### Evidencia de verificación

(pendiente)

### Desviaciones respecto al diseño original

(ninguna aún)
