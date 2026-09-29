"""
=============================================================================
APLICACIÓN: app.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Capa de Presentación (Streamlit)
RESPONSABILIDAD ÚNICA:
    Exponer una interfaz médica reactiva que consume la fachada orquestadora
    'DermatologyDiagnosticFacade' sin importar jamás 'torch' ni 'transformers'.
    Recibe una imagen dermatoscópica, muestra el diagnóstico Top-1 con severidad,
    la explicación Grad-CAM y la distribución de probabilidad de las 7 patologías.

EJECUCIÓN:
    uv run streamlit run app.py
=============================================================================
"""

import time
from pathlib import Path

import streamlit as st
from PIL import Image

from skin_lesion_classifier.facade import DermatologyDiagnosticFacade
from skin_lesion_classifier.gradcam import GradCAMError
from skin_lesion_classifier.inference import InferenceError
from skin_lesion_classifier.labels import human_name, severity_badge, severity_of
from skin_lesion_classifier.model_loader import ModelLoadingError

SAMPLES_DIR = Path(__file__).parent / "data" / "samples"


@st.cache_resource(show_spinner=False)
def get_facade() -> DermatologyDiagnosticFacade:
    """Carga la fachada una única vez (el ViT se reutiliza entre reruns)."""
    return DermatologyDiagnosticFacade()


def list_samples() -> list[Path]:
    """Devuelve las muestras dermatoscópicas disponibles en data/samples/."""
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(SAMPLES_DIR.glob("*.jpg"))


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

# Entrada de imagen: subir archivo o usar una muestra pre-cargada.
uploaded = st.file_uploader(
    "Subir imagen dermatoscópica",
    type=["png", "jpg", "jpeg"],
)

samples = list_samples()
sample_names = [sample.name for sample in samples]
selected_sample = st.selectbox(
    "O usar una muestra de prueba",
    options=[""] + sample_names,
    format_func=lambda name: name or "Seleccionar…",
)

image: Image.Image | None = None
source_name: str = ""

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        source_name = uploaded.name
    except Exception:
        st.error("No se pudo leer la imagen subida. Verificá que el archivo sea una imagen válida.")
elif selected_sample:
    try:
        image = Image.open(SAMPLES_DIR / selected_sample).convert("RGB")
        source_name = selected_sample
    except Exception:
        st.error("No se pudo cargar la muestra seleccionada.")

if image is None:
    st.info("Subí una imagen dermatoscópica o seleccioná una muestra de prueba para comenzar.")
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

# Encabezado del resultado con severidad y métricas de triaje.
label_name = human_name(result.label)
severity = severity_of(result.label)
badge_color, badge_icon = severity_badge(severity)

with st.container(border=True):
    st.subheader(f"Diagnóstico: {label_name}")
    st.badge(severity.capitalize(), icon=badge_icon, color=badge_color)
    metric_conf, metric_lat = st.columns(2)
    metric_conf.metric("Confianza", f"{result.confidence:.1%}")
    metric_lat.metric("Latencia", f"{elapsed_ms:.0f} ms")

# Comparativa visual: imagen original vs. mapa de calor Grad-CAM.
col_original, col_heatmap = st.columns(2)
with col_original:
    st.subheader("Imagen original")
    st.image(image, width="stretch")
with col_heatmap:
    st.subheader("Mapa de calor Grad-CAM")
    st.image(result.superimposed_image, width="stretch")

# Distribución de probabilidad para las 7 patologías HAM10000.
st.subheader("Probabilidad por patología")
ordered = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)
for code, probability in ordered:
    st.markdown(f"**{human_name(code)}** — {probability:.1%}")
    st.progress(min(probability, 1.0))

st.warning(
    "Herramienta de apoyo a la decisión clínica (CDSS). No reemplaza el diagnóstico "
    "de un profesional médico: los resultados deben ser validados por un especialista.",
    icon=":material/warning:",
)
