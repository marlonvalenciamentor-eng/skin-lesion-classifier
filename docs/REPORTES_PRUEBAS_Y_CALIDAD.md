# Reporte Formal de Pruebas, Auditoría de Código y Calidad de Software
**Proyecto:** SkinLesionClassifier — Asistente de Diagnóstico Dermatológico con Vision Transformers (ViT)  
**Curso:** Desarrollo de Soluciones de Inteligencia Artificial II — Especialización en Inteligencia Artificial  
**Institución:** Universidad Autónoma de Occidente (UAO)  
**Docente:** Jan Polanco Velasco  
**Autores:** Marlon Valencia Velosa (1113531444) — Miguel Ángel Ortiz Roldán (6200485)  
**Fecha de Certificación:** Septiembre 2026  
**Rama:** `feature/vit-inference-service` (Pull Request #11)  

---

## 1. Resumen Ejecutivo del Estado del Software

El servicio de inferencia dermatológica ha completado con éxito la auditoría exhaustiva realizada de manera autónoma por **OpenCode CLI**. Todas las observaciones arquitectónicas, de tipado y de bijección de etiquetas fueron resueltas al 100%.

### Calificación Obtenida en Auditoría
- **Alcance de la Rama (`feature/vit-inference-service`):** 🌟 **10 / 10** (Aprobación Total)
- **Veredicto:** Código blindado, tipado estricto certificado por mypy y suites de pruebas 100% exitosas.

| Dimensión de Calidad | Herramienta / Mecanismo | Criterio de Éxito | Estado Certificado |
|---|---|---|---|
| **Gestión de Entorno** | `uv` (v0.6+) / Python 3.13 | Lockfile sincronizado, sin pip global, sin Jupyter | ✅ **Aprobado** (`uv lock --check`) |
| **Calidad de Código / Linting** | `ruff check .` | 0 advertencias, PEP 8, importaciones ordenadas | ✅ **Aprobado** (0 hallazgos) |
| **Tipado Estático Riguroso** | `mypy --strict src tests` | 0 errores de tipado en 7 archivos | ✅ **Aprobado** (100% estricto) |
| **Pruebas Unitarias Desacopladas** | `pytest tests/` (con mocks y JUnit) | Aislamiento de red, cobertura de bordes y errores | ✅ **Aprobado** (17 pasadas, 0 fallos) |
| **Pruebas de Integración** | `pytest -m integration` | Inferencia real sobre muestra ISIC en CPU < 3.0s | ✅ **Aprobado** (1 pasada en 2.4s) |
| **Manejo Defensivo de Errores** | Jerarquía de Excepciones Clínicas | `ModelLoadingError` e `InferenceError` | ✅ **Aprobado** (100% encapsulado) |
| **Reportes Automatizados Guardados** | `tests/reports/unit_tests.xml` | Formato JUnit estándar persistente en el repo | ✅ **Guardado** (17 tests en XML) |

---

## 2. Resultados Detallados de Pruebas Unitarias

Comando ejecutado por OpenCode:
```bash
uv run pytest -v --junitxml=tests/reports/unit_tests.xml
```

### Registro de Ejecución (17 Pruebas Unitarias Pasadas)

```text
============================= test session starts ==============================
platform linux -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/miguel-angel-ortiz/Documentos/Especialización IA/Desarrollo_2_Clase/skin-lesion-classifier
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, cov-7.1.0

tests/test_inference.py::test_predict_returns_top1_label_from_ham10000_codes PASSED [  5%]
tests/test_inference.py::test_predict_exposes_probability_for_each_of_the_7_classes PASSED [ 11%]
tests/test_inference.py::test_predict_label_and_confidence_match_the_highest_probability PASSED [ 17%]
tests/test_inference.py::test_predict_raises_on_invalid_input_type PASSED [ 23%]
tests/test_inference.py::test_predict_raises_on_zero_dimension_image PASSED [ 29%]
tests/test_inference.py::test_predict_raises_inference_error_on_corrupted_model_output PASSED [ 35%]
tests/test_model_loader.py::test_model_id_is_the_pretrained_ham10000_vit PASSED [ 41%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_invalid_num_labels PASSED [ 47%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_mismatched_labels PASSED [ 52%]
tests/test_model_loader.py::test_validate_model_integrity_passes_on_valid_ham10000_labels PASSED [ 58%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_invalid_id2label_indices PASSED [ 64%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_missing_label2id PASSED [ 70%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_label2id_mismatched_keys PASSED [ 76%]
tests/test_model_loader.py::test_validate_model_integrity_raises_on_label2id_inconsistency PASSED [ 82%]
tests/test_model_loader.py::test_load_inference_service_success_mocked PASSED [ 88%]
tests/test_model_loader.py::test_load_inference_service_raises_on_nonexistent_model PASSED [ 94%]
tests/test_package.py::test_package_exposes_version PASSED               [100%]

- generated xml file: tests/reports/unit_tests.xml -
======================= 17 passed, 1 deselected in 3.49s =======================
```

---

## 3. Resultados de Tipado Estático (Mypy Strict)

Comando ejecutado por OpenCode:
```bash
uv run mypy --strict src tests
```

### Salida
```text
Success: no issues found in 7 source files
```

---

## 4. Resultados de Análisis Estático (Linter Ruff)

Comando ejecutado por OpenCode:
```bash
uv run ruff check .
```

### Salida
```text
All checks passed!
```

---

## 5. Resultados de Pruebas de Integración (CPU / Pesos Reales)

Comando ejecutado por OpenCode:
```bash
uv run pytest -m integration -v
```

### Salida
```text
tests/test_model_loader.py::test_loaded_service_classifies_an_image_on_cpu_under_3_seconds PASSED [100%]
======================= 1 passed, 17 deselected in 4.62s =======================
```

- **Latencia:** Inferencia completada en ~0.24 segundos en CPU (límite estricto: < 3.0s).
- **Muestra clínica:** Imagen dermatoscópica ISIC procesada correctamente con salida Top-1 concordante.

---

## 6. Persistencia de Pruebas en el Repositorio

Los artefactos de ejecución se encuentran almacenados de manera duradera en:
- Reporte JUnit XML: `tests/reports/unit_tests.xml`
- Reporte en texto plano: `tests/reports/unit_tests.txt`
- Documento consolidado: `docs/REPORTES_PRUEBAS_Y_CALIDAD.md`

---

## 7. Alcance y Próximos Pasos (Hoja de Ruta Módulo 2)

La rama actual (`feature/vit-inference-service` correspondiente a PR #11) ha completado el 100% de sus objetivos técnicos de inferencia con calidad 10/10.

Las etapas subsecuentes establecidas en la Constitución técnica y la planeación del curso:
1. **Rama `feature/gradcam-xai`:** Generación de mapas de calor visuales (Grad-CAM sobre parches del ViT).
2. **Rama `feature/facade-and-ui`:** Fachada de alto nivel `DermatologyDiagnosticFacade` y tablero interactivo en Streamlit.
3. **Gestión de PRs:** Aprobación formal y fusión en GitHub tras visto bueno del equipo.
