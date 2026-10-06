"""
=============================================================================
APLICACIÓN: app.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Capa de Presentación (Streamlit)
RESPONSABILIDAD ÚNICA:
    Exponer una interfaz médica reactiva que consume la fachada orquestadora
    'DermatologyDiagnosticFacade' sin importar jamás 'torch' ni 'transformers'.
    Captura el nombre del paciente, muestra el diagnóstico Top-1 con severidad,
    la explicación Grad-CAM y la distribución de probabilidad de las 7 patologías,
    y permite exportar el resultado como reporte PDF (ver 'report.py').

EJECUCIÓN:
    uv run streamlit run app.py
=============================================================================
"""

import time

import streamlit as st
from PIL import Image

from skin_lesion_classifier.facade import DermatologyDiagnosticFacade
from skin_lesion_classifier.gradcam import GradCAMError
from skin_lesion_classifier.inference import InferenceError
from skin_lesion_classifier.labels import human_name, severity_badge, severity_of
from skin_lesion_classifier.model_loader import ModelLoadingError
from skin_lesion_classifier.report import build_report_pdf

# Ancho fijo de visualización para ambas imágenes (original y Grad-CAM), de modo
# que se muestren del mismo tamaño y sin ocupar todo el ancho de la columna.
DISPLAY_IMAGE_WIDTH = 360

# Lado máximo del overlay Grad-CAM al reescalarlo al aspect ratio de la imagen original.
_OVERLAY_MAX_LONG_SIDE = 512


@st.cache_resource(show_spinner=False)
def get_facade() -> DermatologyDiagnosticFacade:
    """Carga la fachada una única vez (el ViT se reutiliza entre reruns)."""
    return DermatologyDiagnosticFacade()


def resize_overlay_to_match(overlay: Image.Image, original: Image.Image) -> Image.Image:
    """Reescala el overlay Grad-CAM al aspect ratio de la imagen original.

    El overlay se produce a la resolución cuadrada del modelo (224x224); si la
    imagen original es rectangular, reescalarlo a su aspect ratio evita que la
    comparativa visual se vea distorsionada o de distinto tamaño.
    """
    original_width, original_height = original.size
    if original_width <= 0 or original_height <= 0:
        return overlay
    aspect_ratio = original_width / original_height
    if aspect_ratio >= 1.0:
        target_width = _OVERLAY_MAX_LONG_SIDE
        target_height = max(1, round(_OVERLAY_MAX_LONG_SIDE / aspect_ratio))
    else:
        target_height = _OVERLAY_MAX_LONG_SIDE
        target_width = max(1, round(_OVERLAY_MAX_LONG_SIDE * aspect_ratio))
    return overlay.convert("RGB").resize((target_width, target_height), Image.Resampling.LANCZOS)


st.set_page_config(
    page_title="SkinLesionClassifier",
    page_icon=":material/monitor_heart:",
    layout="wide",
)

st.title("SkinLesionClassifier", icon=":material/monitor_heart:")
st.caption(
    "Sistema asistivo para la detección temprana de cáncer de piel · "
    "Universidad Autónoma de Occidente · Especialización en Inteligencia Artificial"
)

# Entrada de imagen y datos del paciente.
col_upload, col_patient = st.columns(2)
with col_upload:
    uploaded = st.file_uploader("Subir imagen dermatoscópica", type=["png", "jpg", "jpeg"])
with col_patient:
    patient_name = st.text_input("Nombre del paciente", placeholder="Opcional")

image: Image.Image | None = None

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("No se pudo leer la imagen subida. Verificá que el archivo sea una imagen válida.")

if image is None:
    st.info("Subí una imagen dermatoscópica para comenzar.")
    st.stop()

# Diagnóstico clínico completo (predicción + Grad-CAM).
try:
    facade = get_facade()
except ModelLoadingError as err:
    st.error(f"No se pudo cargar el modelo de clasificación: {err}")
    st.stop()

try:
    started = time.perf_counter()
    with st.spinner("Analizando la lesión…"):
        result = facade.diagnose(image)
    elapsed_ms = (time.perf_counter() - started) * 1000
except (InferenceError, GradCAMError) as err:
    st.error(f"No se pudo completar el diagnóstico: {err}")
    st.stop()

# Vista en tres columnas: imagen | mapa de calor | diagnóstico.
severity = severity_of(result.label)
badge_color, badge_icon = severity_badge(severity)
label_name = human_name(result.label)
overlay = resize_overlay_to_match(result.superimposed_image, image)

col_image, col_heatmap, col_diagnosis = st.columns([1, 1, 1.25])

with col_image:
    st.subheader("Imagen original")
    st.image(image, width=DISPLAY_IMAGE_WIDTH)

with col_heatmap:
    st.subheader("Mapa de calor Grad-CAM")
    st.image(overlay, width=DISPLAY_IMAGE_WIDTH)
    st.caption("🔴 Mayor atención del modelo · 🔵 Menor atención")

with col_diagnosis:
    st.subheader(f"Diagnóstico: {label_name}")
    st.badge(severity.capitalize(), icon=badge_icon, color=badge_color)
    metric_conf, metric_lat = st.columns(2)
    metric_conf.metric("Confianza", f"{result.confidence:.1%}")
    metric_lat.metric("Latencia", f"{elapsed_ms:.0f} ms")

    st.markdown("**Probabilidad por patología**")
    ordered = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)
    probability_rows = [
        {"Patología": human_name(code), "Probabilidad (%)": round(probability * 100, 1)}
        for code, probability in ordered
    ]
    st.dataframe(
        probability_rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Probabilidad (%)": st.column_config.ProgressColumn(
                "Probabilidad",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            )
        },
    )

    pdf_bytes = build_report_pdf(
        patient_name=patient_name,
        image=image,
        overlay=overlay,
        label_name=label_name,
        severity=severity,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )
    safe_name = patient_name.strip().replace(" ", "_") or "paciente"
    st.download_button(
        "Descargar reporte (PDF)",
        data=pdf_bytes,
        file_name=f"reporte_{safe_name}.pdf",
        mime="application/pdf",
        icon=":material/download:",
    )

with st.expander("¿Cómo interpretar el mapa de calor?"):
    st.markdown(
        "Grad-CAM aplicado a un Vision Transformer es una aproximación: puede resaltar "
        "también piel circundante además de la lesión, no solo el área patológica exacta. "
        "Úsalo como apoyo visual complementario al diagnóstico, no como una delimitación "
        "precisa de los bordes de la lesión."
    )

st.warning(
    "Herramienta de apoyo a la decisión clínica (CDSS). No reemplaza el diagnóstico de un "
    "profesional médico: los resultados deben ser validados por un especialista.",
    icon=":material/warning:",
)
