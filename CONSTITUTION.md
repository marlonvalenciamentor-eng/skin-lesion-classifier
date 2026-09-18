# CONSTITUCIÓN DEL PROYECTO: Clasificador de Lesiones de Piel

> **Contexto del Proyecto:** Sistema Inteligente de apoyo al diagnóstico dermatológico basado en Inteligencia Artificial con Vision Transformers (ViT) y Explicabilidad Visual (Grad-CAM).  
> **Equipo:** Miguel Ángel Ortiz Roldán (6200485) & Marlon Valencia Velosa (1113531444)  
> **Institución:** Universidad Autónoma de Occidente (UAO) — Especialización en Inteligencia Artificial

---

## 📌 Principios de Arquitectura de Software

1. **Clean Architecture (Patrón Fachada):**
   - Separación estricta de capas: **Presentación** (UI Web con Streamlit), **Orquestación** (Fachada / Service Layer) y **Lógica de Dominio** (Modelos de Inferencia, Preprocesamiento y Explicabilidad XAI).
   - Un módulo orquestador (`src/service.py` o `src/integrator.py`) actúa como la única Fachada. La interfaz web (`app.py`) **nunca** debe importar librerías de Deep Learning (`torch`, `transformers`) directamente ni manipular tensores.

2. **Alta Cohesión y Bajo Acoplamiento (SOLID):**
   - Cada módulo tiene una responsabilidad única:
     - `src/io_handler.py`: Carga, validación de formato (PNG, JPEG, DICOM/Metadata) y verificación de dimensiones mínimas.
     - `src/preprocessing.py`: Redimensionamiento (224x224), normalización acorde a pesos pre-entrenados ImageNet y conversión a tensores.
     - `src/model_loader.py`: Carga desacoplada del modelo y tokenizador/extractor de características desde Hugging Face (`Anwarkh1/Skin_Cancer-Image_Classification`).
     - `src/grad_cam.py`: Extracción de gradientes de atención en la última capa del Vision Transformer para proyectar mapas de calor sobre la imagen.

---

## 🛠️ Stack Técnico Oficial

| Categoría | Tecnología Seleccionada | Justificación Técnica |
|---|---|---|
| **Lenguaje** | Python 3.11 / 3.12 / 3.13 | Estándar de la industria para ecosistemas de IA. |
| **Gestor de Entorno y Paquetes** | `uv` | Gestión ultra-rápida, determinista vía `uv.lock` (prohibido `pip` suelto). |
| **Arquitectura del Modelo** | Vision Transformer (`google/vit-base-patch16-224-in21k`) | Estado del arte en Computer Vision mediante atención de parches. |
| **Modelo Pre-entrenado** | `Anwarkh1/Skin_Cancer-Image_Classification` | Fine-tuneado en dataset derivado de HAM10000 con 96.95% de exactitud. |
| **Formato de Pesos** | `safetensors` | Carga rápida, segura contra deserialización maliciosa y bajo consumo de memoria (343 MB). |
| **Framework de Deep Learning** | `PyTorch` + `Hugging Face Transformers` | Ecosistema nativo para Transformers y cálculo de gradientes. |
| **Explicabilidad (XAI)** | `Grad-CAM` / `PyTorch-Grad-CAM` | Mapas de calor para transparencia clínica y reducción del sesgo de "caja negra". |
| **Interfaz de Usuario** | `Streamlit` | Interfaz reactiva web, moderna, liviana y con capacidad de despliegue cloud. |
| **Calidad y Testing** | `pytest` + `pytest-cov` | Cobertura de pruebas unitarias y de integración continua. |

---

## ✅ Reglas de Calidad, Git y MLOps

1. **Gestión de Artefactos y Modelos Pesados:**
   - Queda **estrictamente prohibido** subir archivos binarios pesados (`.safetensors`, `.bin`, `.pt`) al repositorio Git. Deben descargarse bajo demanda con caché de Hugging Face o especificarse en `.gitignore`.
2. **Convención de Commits (Conventional Commits):**
   - Todo commit debe mantener la nomenclatura estándar:
     - `feat:` Nuevas funcionalidades (ej. endpoint de inferencia, widget UI).
     - `fix:` Corrección de errores o bugs.
     - `test:` Inclusión o ajuste de pruebas unitarias.
     - `docs:` Documentación, especificaciones o actualización del README/propuesta.
     - `refactor:` Mejoras en código sin alterar comportamiento.
3. **Calidad de Código y Tipado:**
   - Todo módulo dentro de `src/` debe contar con **Type Hints** completos (PEP 484) y **Docstrings estilo Google** (`Args:`, `Returns:`, `Raises:`).
4. **Pruebas Automatizadas:**
   - La suite de pruebas debe ejecutarse limpiamente mediante `uv run pytest` antes de fusionar cualquier rama a `main`.

---

## 🚀 Flujo Diario de Trabajo (Workflow)

```bash
# 1. Sincronizar el entorno virtual con uv
uv sync

# 2. Ejecutar la suite de pruebas unitarias
uv run pytest -v

# 3. Lanzar la aplicación web en entorno de desarrollo
uv run streamlit run app.py
```
