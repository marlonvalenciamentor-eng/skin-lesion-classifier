# Guion de exposición — SkinLesionClassifier

Detección temprana de cáncer de piel con Vision Transformers (ViT) y Grad-CAM.

---

## 1. Apertura (30 seg)

> "SkinLesionClassifier es un sistema asistivo para la detección temprana de cáncer de piel. Subís una imagen dermatoscópica y, en segundos, te devuelve el diagnóstico probable, la severidad clínica, y un mapa de calor Grad-CAM que muestra **dónde** el modelo está mirando para decidir. Todo bajo Clean Architecture, con Python 3.13 y Streamlit."

## 2. Arquitectura en una frase

> "El sistema sigue el patrón **Fachada**. La interfaz (Streamlit) **nunca** importa `torch` ni `transformers`: habla solo con `DermatologyDiagnosticFacade`, que es el único componente que toca el modelo. Eso se llama desacoplamiento UI ↔ modelo."

## 3. El recorrido paso a paso (el corazón de la demo)

### Paso 1 — Abrir la app

Ejecutás `uv run streamlit run app.py`. Se renderiza `app.py`: el título institucional y dos campos de entrada.

### Paso 2 — Subir la imagen

Con `st.file_uploader` elegís un PNG/JPG. Con `st.text_input` escribís el nombre del paciente (opcional). Acá todavía **no se tocó el modelo**.

### Paso 3 — Cargar la fachada (una sola vez)

Al diagnosticar, se llama `get_facade()`, decorada con `@st.cache_resource`. Esto es clave: el ViT pesa **~343 MB**, así que se carga **una única vez** y queda cacheado para todas las consultas siguientes. Internamente `get_facade()` construye `DermatologyDiagnosticFacade()`, que a su vez llama a `load_inference_service()` de `model_loader.py`:

> "`model_loader.py` aplica política **offline-first**: si existe `models/vit-skin-cancer/` con `model.safetensors` completo, lo carga del disco; si no, descarga de Hugging Face. Y antes de usar el modelo, `validate_model_integrity()` comprueba que tenga exactamente las **7 clases HAM10000** y que `id2label`/`label2id` sean biyectivos — si no, lanza `ModelLoadingError`."

### Paso 4 — `facade.diagnose(image)`

La fachada orquesta dos cosas en secuencia:

- **(a) Inferencia** — `InferenceService.predict(image)` de `inference.py`:

  > "Convierte la imagen a RGB, la redimensiona a **224×224** (lo que espera el ViT), la pasa por el procesador que la convierte en tensor `[1, 3, 224, 224]`. Hace el *forward pass* en `torch.inference_mode()` (sin gradientes, más rápido). Aplica **softmax** a los logits para obtener probabilidades, mapea las etiquetas largas de Hugging Face a los **códigos cortos HAM10000** (`mel`, `nv`, `bcc`…), y se queda con el **Top-1**."

- **(b) Explicabilidad** — `ViTGradCAM.explain(image)` de `gradcam.py`:

  > "Registra un **hook** en la última capa de auto-atención del encoder (`vit.encoder.layer[-1].layernorm_before`), hace un *backward* para capturar los **gradientes** de la clase objetivo, excluye el token `[CLS]`, toma los **196 parches** y los organiza en una grilla **14×14**. Pondera las activaciones por los gradientes, aplica ReLU (para quedarse solo con lo que contribuye positivamente), interpola bilinealmente a 224×224, y superpone un colormap **JET** (55% imagen + 45% calor)."

  > "Acá hay un detalle fino: la fachada traduce el código corto (`nv`) a la etiqueta larga del modelo (`melanocytic_Nevi`) antes de llamar a `explain()`, porque el explainer resuelve clases contra `id2label`. Lo hacemos con `CODE_TO_MODEL_LABEL`."

El resultado se combina en un DTO inmutable `DiagnosticResult` (label, confidence, probabilities, heatmap, superimposed_image).

### Paso 5 — Pintar los resultados (3 columnas)

`app.py` muestra:

- Columna 1: **imagen original**.
- Columna 2: **mapa de calor Grad-CAM**. Acá usamos `resize_overlay_to_match()`: como el overlay sale **cuadrado (224×224)** pero la foto original es rectangular, lo reescalamos al **aspect ratio de la original** para que se vean del mismo tamaño.
- Columna 3: **diagnóstico** — badge de severidad (`severity_badge` de `labels.py`: rojo = maligno, naranja = precanceroso, verde = benigno), métricas de confianza y latencia, y la tabla de probabilidades de las 7 patologías (con `ProgressColumn`).

### Paso 6 — Reporte PDF

El botón "Descargar reporte (PDF)" llama a `build_report_pdf()` de `report.py`, que con `matplotlib` genera un **A4** con: nombre del paciente, fecha, diagnóstico, severidad, confianza, ambas imágenes y las probabilidades. Todo sin tocar el modelo.

## 4. Tabla resumen (para mostrar en diapositiva)

| Módulo | Responsabilidad | Función clave |
|---|---|---|
| `app.py` | Presentación (Streamlit) | `get_facade()`, `resize_overlay_to_match()` |
| `facade.py` | Orquestación | `diagnose(image) -> DiagnosticResult` |
| `model_loader.py` | Carga + validación del modelo | `load_inference_service()`, `validate_model_integrity()` |
| `inference.py` | Predicción | `predict(image) -> Prediction` |
| `gradcam.py` | Explicabilidad | `explain(image) -> GradCAMResult` |
| `labels.py` | Taxonomía clínica | `human_name()`, `severity_of()`, `severity_badge()` |
| `report.py` | Reporte PDF | `build_report_pdf()` |

## 5. Cierre (15 seg)

> "El resultado: un pipeline modular donde cada pieza tiene una responsabilidad única, testeable y desacoplada. La UI no sabe nada de deep learning, y el modelo no sabe nada de la interfaz. Y lo más importante clínicamente: no es una caja negra — Grad-CAM te muestra **por qué** el modelo decide lo que decide."
