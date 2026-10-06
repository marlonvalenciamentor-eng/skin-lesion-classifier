# Guion de exposición — SkinLesionClassifier

Detección temprana de cáncer de piel con Vision Transformers (ViT) y Grad-CAM.

---

## Reparto de la exposición

Cada bloque lo presenta quien construyó esa parte del sistema.

| Bloque | Expositor | Contenido |
|---|---|---|
| 1. Contexto y planeación | Marlon | Problema, enfoque asistivo, Kanban y cronograma |
| 2–5. Producto, arquitectura y demo | Miguel | Fachada, ViT, Grad-CAM, UI y reporte PDF |
| 6. Proceso de ingeniería | Marlon | CI, Docker y decisión de servicio único |
| 7. Validación del modelo | Marlon | Métricas y limitaciones |
| 8. Cierre | Miguel | Mensaje final |

## 1. Contexto y planeación — Marlon (30 seg)

> "El cáncer de piel tiene una alta probabilidad de cura cuando se detecta a tiempo, y el melanoma es el más peligroso. Por eso nos propusimos un sistema **asistivo**, no un reemplazo del dermatólogo. Organizamos el trabajo en 4 fases con un tablero Kanban en GitHub Projects y un cronograma de 15 días con ruta crítica; cada funcionalidad entró al proyecto mediante pull request."

## 2. Apertura del producto — Miguel (30 seg)

> "SkinLesionClassifier es un sistema asistivo para la detección temprana de cáncer de piel. Subís una imagen dermatoscópica y, en segundos, te devuelve el diagnóstico probable, la severidad clínica, y un mapa de calor Grad-CAM que muestra **dónde** el modelo está mirando para decidir. Todo bajo Clean Architecture, con Python 3.13 y Streamlit."

## 3. Arquitectura en una frase — Miguel

> "El sistema sigue el patrón **Fachada**. La interfaz (Streamlit) **nunca** importa `torch` ni `transformers`: habla solo con `DermatologyDiagnosticFacade`, que es el único componente que toca el modelo. Eso se llama desacoplamiento UI ↔ modelo."

## 4. El recorrido paso a paso (el corazón de la demo) — Miguel

### Paso 1 — Abrir la app

Ejecutás `uv run streamlit run app.py`. Se renderiza `app.py`: el título institucional y dos campos de entrada.

### Paso 2 — Subir la imagen

Con `st.file_uploader` elegís un PNG/JPG. Con `st.text_input` escribís el nombre del paciente (opcional). Acá todavía **no se tocó el modelo**.

> Sugerencia para la demo: usar una imagen de **melanoma**, la clase clínicamente prioritaria. Evitar apoyarse en carcinoma basocelular, cuyo recall medido es de solo 36 % (ver sección 7).

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

## 5. Tabla resumen (para mostrar en diapositiva) — Miguel

| Módulo | Responsabilidad | Función clave |
|---|---|---|
| `app.py` | Presentación (Streamlit) | `get_facade()`, `resize_overlay_to_match()` |
| `facade.py` | Orquestación | `diagnose(image) -> DiagnosticResult` |
| `model_loader.py` | Carga + validación del modelo | `load_inference_service()`, `validate_model_integrity()` |
| `inference.py` | Predicción | `predict(image) -> Prediction` |
| `gradcam.py` | Explicabilidad | `explain(image) -> GradCAMResult` |
| `labels.py` | Taxonomía clínica | `human_name()`, `severity_of()`, `severity_badge()` |
| `report.py` | Reporte PDF | `build_report_pdf()` |

## 6. Proceso de ingeniería — Marlon (45 seg)

> "Probamos separar el sistema en una API con FastAPI y una interfaz que la consumía por HTTP, cada una en su propio contenedor Docker. Funcionó, pero comprobamos que el desacoplamiento real ya lo daba la fachada y que dos procesos solo sumaban complejidad. Por eso volvimos a un **servicio único**: una decisión basada en evidencia, no en moda."

> "Todo cambio pasa por integración continua antes de entrar a `main`: análisis estático con `ruff`, la suite de pruebas con `pytest` y la construcción de la imagen Docker. Además, cada commit pasa por una revisión automática de código (`gga`) que valida las reglas de arquitectura, por ejemplo que la UI nunca importe `torch`."

## 7. Validación del modelo — Marlon (1 min)

Fuente: [`REPORTE_VALIDACION_MODELO.md`](REPORTE_VALIDACION_MODELO.md).

> "Evaluamos el modelo sobre el split de prueba de HAM10000. La exactitud global es de **83,2 %**, pero ese número engaña: la clase mayoritaria son los nevus benignos. Lo clínicamente importante es la sensibilidad para melanoma: **87,5 %** (126 de 144), y nuestra meta era superar el 90 %, así que **no se cumple**."

> "Además encontramos que el split de prueba se solapa con el de entrenamiento, lo que hace optimistas los resultados. Y en carcinoma basocelular el modelo solo acierta el **36 %** de los casos."

> "Por eso lo presentamos como herramienta de **apoyo**. Los siguientes pasos serían reentrenar con datos sin solapamiento y ajustar el umbral de decisión para priorizar la sensibilidad en melanoma."

## 8. Cierre — Miguel (15 seg)

> "El resultado: un pipeline modular donde cada pieza tiene una responsabilidad única, testeable y desacoplada. La UI no sabe nada de deep learning, y el modelo no sabe nada de la interfaz. Y lo más importante clínicamente: no es una caja negra — Grad-CAM te muestra **por qué** el modelo decide lo que decide."

---

## Anexo — Flujo técnico de referencia (resumen)

Guía rápida del recorrido completo, de la interfaz al PDF. La **fachada orquesta**:
llama a `model_loader` → `inference` → `gradcam` **en orden** (secuencial, no en paralelo)
y junta el resultado.

```
1. app.py   →  file_uploader captura la imagen → Image.open().convert("RGB")
2. app.py   →  get_facade()  (cacheado: se construye UNA sola vez)
3. facade   →  load_inference_service()  (model_loader)
               · local-first: models/vit-skin-cancer si está completo
               · si no → descarga de Hugging Face Hub (transformers, SIN API)
               · valida las 7 clases HAM10000
4. facade   →  predict(image)  (inference)
               · RGB → 224×224 → tensor → forward pass → softmax → Top-1
5. facade   →  explain(image)  (gradcam)   ← secuencial, después de predict
               · hook en la última capa → gradientes → heatmap → overlay
6. facade   →  combina todo en DiagnosticResult
7. app.py   →  muestra 3 columnas (imagen | heatmap | diagnóstico)
8. report   →  build_report_pdf() genera el PDF descargable
```

> `load_inference_service()` **solo carga** el modelo y el procesador (una vez).
> La **predicción** la hace `predict()`, y la **explicación** `explain()` — son pasos distintos.
