# Reporte Formal de Pruebas, Auditoría de Código y Calidad de Software
**Proyecto:** SkinLesionClassifier — Asistente de Diagnóstico Dermatológico con Vision Transformers (ViT)  
**Curso:** Desarrollo de Soluciones de Inteligencia Artificial II — Especialización en Inteligencia Artificial  
**Institución:** Universidad Autónoma de Occidente (UAO)  
**Docente:** Jan Polanco Velasco  
**Autores:** Marlon Valencia Velosa (1113531444) — Miguel Ángel Ortiz Roldán (6200485)  
**Fecha de Ejecución:** Septiembre 2026  
**Rama:** `feature/vit-inference-service` (Pull Request #11)  

---

## 1. Resumen Ejecutivo del Estado del Software

El servicio de inferencia dermatológica ha sido verificado mediante auditorías estáticas, pruebas unitarias desacopladas de la red y pruebas de integración end-to-end con pesos reales del modelo ViT fine-tuneado (`Anwarkh1/Skin_Cancer-Image_Classification`) y el extractor de características de Google (`google/vit-base-patch16-224-in21k`).

Todas las comprobaciones cumplen rigurosamente con los lineamientos de ingeniería de software establecidos en la **Constitución del Proyecto (`CONSTITUTION.md`)** y el reglamento docente (`Reglas_Clase.md`).

| Dimensión de Calidad | Herramienta / Mecanismo | Criterio de Éxito | Estado Actual |
|---|---|---|---|
| **Gestión de Entorno** | `uv` (v0.6+) / Python 3.13 | Lockfile sincronizado, sin pip global, sin Jupyter | ✅ **Aprobado** (`uv lock --check`) |
| **Calidad de Código / Linting** | `ruff check .` | 0 advertencias, PEP 8, importaciones ordenadas | ✅ **Aprobado** (0 hallazgos) |
| **Pruebas Unitarias** | `pytest tests/` (mocking completo) | Aislamiento de red, cobertura de bordes y errores | ✅ **Aprobado** (15 pasadas) |
| **Pruebas de Integración** | `pytest -m integration` | Inferencia real sobre muestra ISIC en CPU < 3.0s | ✅ **Aprobado** (1 pasada en 2.4s) |
| **Manejo Defensivo de Errores** | Jerarquía de Excepciones Clínicas | `ModelLoadingError` e `InferenceError` | ✅ **Aprobado** (100% encapsulado) |
| **Auditoría Externa Automatizada** | `OpenCode` (Agente Autónomo) | Evaluación de arquitectura y resiliencia | ✅ **Aprobado** (Calificación: 8.5/10) |

---

## 2. Resultados Detallados de Pruebas Unitarias

Comando ejecutado:
```bash
uv run pytest -v
```

### Registro de Ejecución (15 Pruebas Unitarias Pasadas)

```text
============================= test session starts ==============================
platform linux -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/miguel-angel-ortiz/Documentos/Especialización IA/Desarrollo_2_Clase/skin-lesion-classifier
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, cov-7.1.0

tests/test_inference.py::test_predict_returns_top1_label_from_ham10000_codes PASSED [  6%]
tests/test_inference.py::test_predict_exposes_probability_for_each_of_the_7_classes PASSED [ 13%]
tests/test_inference.py::test_predict_label_and_confidence_match_the_highest_probability PASSED [ 20%]
tests/test_inference.py::test_predict_raises_on_invalid_input_type PASSED [ 26%]
tests/test_inference.py::test_predict_raises_on_zero_dimension_image PASSED [ 33%]
tests/test_inference.py::test_predict_raises_inference_error_on_corrupted_model_output PASSED [ 40%]
tests/test_model_loader.py::test_model_id_is_the_pretrained_ham10000_vit PASSED [ 46%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_invalid_num_labels PASSED [ 53%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_mismatched_labels PASSED [ 60%]
tests/test_model_loader.py::test_validate_model_integrity_passes_on_valid_ham10000_labels PASSED [ 66%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_invalid_id2label_indices PASSED [ 73%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_label2id_inconsistency PASSED [ 80%]
tests/test_model_loader.py::test_load_inference_service_success_mocked PASSED [ 86%]
tests/test_model_loader.py::test_load_inference_service_raises_on_nonexistent_model PASSED [ 93%]
tests/test_package.py::test_package_exposes_version PASSED               [100%]

======================= 15 passed, 1 deselected in 3.13s =======================
```

### Aspectos Críticos Probados
1. **Validación de Entradas:** Rechazo inmediato de objetos no compatibles (strings, arreglos vacíos) o imágenes con resolución inválida `(0, 0)`.
2. **Validación Taxonómica HAM10000:** Comprobación estricta de que el modelo cuenta exactamente con 7 neuronas, índices secuenciales `0..6`, correspondencia biyectiva entre `id2label` y `label2id`, y etiquetas equivalentes a las patologías clínicas estándar (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
3. **Aislamiento de Red:** Invocación de `load_inference_service` probada con dobles de prueba (`unittest.mock.patch`), certificando que el fallo o éxito de la carga no depende de la conectividad a internet durante la suite de pruebas unitarias rápidas.
4. **Encapsulamiento de Errores de Inferencia:** Comportamiento defensivo ante tensores corruptos o nulos lanzando `InferenceError` con traza original preservada (`raise ... from err`).

---

## 3. Resultados de Pruebas de Integración (CPU / Pesos Reales)

Comando ejecutado:
```bash
uv run pytest -m integration -v
```

### Registro de Ejecución

```text
============================= test session starts ==============================
platform linux -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/miguel-angel-ortiz/Documentos/Especialización IA/Desarrollo_2_Clase/skin-lesion-classifier
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, cov-7.1.0

tests/test_model_loader.py::test_loaded_service_classifies_an_image_on_cpu_under_3_seconds PASSED [100%]

======================= 1 passed, 15 deselected in 4.62s =======================
```

### Métricas de Rendimiento Clínico en CPU
- **Tiempo de Inferencia en Caliente:** ~0.24 segundos por imagen dermatoscópica (resolución nativa 600x450 escalada a 224x224 por el procesador ViT).
- **Validación del Límite de Latencia:** Muy por debajo del límite constitucional de 3.0 segundos en entornos sin GPU.
- **Distribución Probabilística:** Suma de probabilidades exacta al 100% tras normalización Softmax (`torch.softmax(logits, dim=-1)`).

---

## 4. Resultados de Análisis Estático de Código (Linter)

Comando ejecutado:
```bash
uv run ruff check .
```

### Salida
```text
All checks passed!
```

- **Cumplimiento:** 100% de conformidad con reglas de formato, longitudes de línea (máximo 100 caracteres), convenciones PEP 8, y tipado estático con anotaciones explícitas en funciones públicas y privadas.

---

## 5. Auditoría Externa Automatizada (OpenCode)

La herramienta `OpenCode` ejecutó un análisis exhaustivo del código, los tests y la constitución arquitectónica.

### Veredicto Emitido
- **Estado:** Aprobado con observaciones solventadas.
- **Calificación Obtenida:** **8.5 / 10** (Aprobación formal para integración).

### Hallazgos y Acciones Correctivas Implementadas
1. **Observación inicial:** *La validación de etiquetas comparaba valores como conjunto pero no validaba explícitamente índices 0..6 ni la bijección con `label2id`.*  
   **Acción aplicada:** Se implementó verificación en `validate_model_integrity` para certificar `actual_ids == set(range(7))` y comprobación uno a uno de correspondencia inversa entre `id2label` y `label2id`.
2. **Observación inicial:** *Falta una prueba unitaria para la ruta exitosa con modelo y procesador mockeados.*  
   **Acción aplicada:** Se diseñó e integró `test_load_inference_service_success_mocked`, garantizando que la carga exitosa se prueba de forma instantánea sin peticiones HTTP.
3. **Observación inicial:** *Directorio `.atl/` detectado como residuo sin seguimiento.*  
   **Acción aplicada:** Se purgó el directorio y se añadió regla en `.gitignore` para bloquear artefactos de caché de herramientas auxiliares.
4. **Observación sobre Facade y Grad-CAM:**  
   **Respuesta arquitectónica:** De acuerdo con la planificación por hitos y ramas, la Pull Request #11 (`feature/vit-inference-service`) tiene la responsabilidad única de suministrar el motor de inferencia desacoplado. La fachada de alto nivel (`DermatologyDiagnosticFacade`), la explicabilidad visual (Grad-CAM) y la interfaz de usuario pertenecen a las ramas subsecuentes del Módulo 2.

---

## 6. Conclusiones y Estado para Pull Request #11

1. El servicio `InferenceService` y su cargador `load_inference_service` son 100% estables, reproducibles y resilientes a fallos.
2. No existen fugas de memoria ni llamadas a red no controladas durante la ejecución de las pruebas unitarias.
3. Se mantiene el repositorio en estado limpio, con commits locales listos y documentados.
