"""
=============================================================================
APLICACIÓN: app.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Capa de Presentación (Streamlit)
RESPONSABILIDAD ÚNICA:
    Exponer una interfaz médica reactiva que consume el servicio de inferencia
    por HTTP (`api_client.DiagnosisClient`). Nunca importa 'torch',
    'transformers' ni la fachada de orquestación: solo conoce los contratos
    Pydantic compartidos ('schemas.py') y la taxonomía clínica ('labels.py').
    Recibe una imagen dermatoscópica, muestra el diagnóstico Top-1 con
    severidad, la explicación Grad-CAM y la distribución de probabilidad de
    las 7 patologías HAM10000.

EJECUCIÓN:
    uv run streamlit run app.py
    (requiere el servicio de inferencia activo; ver API_URL más abajo)

NOTA: el cuerpo de la interfaz (a partir de `st.set_page_config`) vive bajo
`if __name__ == "__main__":`, exactamente como Streamlit ejecuta este script
(`__name__` es `"__main__"` tanto bajo `streamlit run` como bajo
`AppTest.from_file(...)`). Esto permite importar las funciones puras de este
módulo (p. ej. `_resolve_api_timeout_seconds`) en pruebas unitarias sin montar
una sesión de Streamlit ni intentar una conexión de red real.
=============================================================================
"""

import base64
import io
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from skin_lesion_classifier.api_client import ApiClientError, DiagnosisClient
from skin_lesion_classifier.labels import human_name, severity_badge

SAMPLES_DIR = Path(__file__).parent / "data" / "samples"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_API_TIMEOUT_SECONDS = 60.0
DISPLAY_IMAGE_WIDTH = 360


def _resolve_api_timeout_seconds() -> float:
    """
    Read `API_TIMEOUT_SECONDS` from the environment, falling back to the default
    when it is unset, not a number, or not strictly positive.
    """
    raw_value = os.environ.get("API_TIMEOUT_SECONDS")
    if raw_value is None:
        return DEFAULT_API_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_API_TIMEOUT_SECONDS
    if value <= 0:
        return DEFAULT_API_TIMEOUT_SECONDS
    return value


@st.cache_resource(show_spinner=False)
def get_client() -> DiagnosisClient:
    """Construye el cliente HTTP una única vez por sesión (se reutiliza entre reruns)."""
    api_url = os.environ.get("API_URL", DEFAULT_API_URL)
    return DiagnosisClient(base_url=api_url, timeout=_resolve_api_timeout_seconds())


def list_samples() -> list[Path]:
    """Devuelve las muestras dermatoscópicas disponibles en data/samples/."""
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(SAMPLES_DIR.glob("*.jpg"))


def decode_overlay(overlay_png_base64: str) -> Image.Image:
    """Decodifica el mapa de calor Grad-CAM (PNG en base64) devuelto por la API."""
    return Image.open(io.BytesIO(base64.b64decode(overlay_png_base64)))


if __name__ == "__main__":
    # --- Encabezado: debe renderizarse de inmediato, antes de cualquier llamada de red. ---
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

    client = get_client()

    # --- Barra lateral: entrada de imagen y estado del servicio de inferencia. ---
    with st.sidebar:
        st.header("Entrada")
        uploaded = st.file_uploader(
            "Imagen dermatoscópica",
            type=["png", "jpg", "jpeg"],
        )

        samples = list_samples()
        sample_names = [sample.name for sample in samples]
        selected_sample = st.selectbox(
            "O selecciona una muestra de prueba",
            options=[""] + sample_names,
            format_func=lambda name: name or "Seleccionar…",
        )

        st.divider()
        st.subheader("Estado del servicio")
        try:
            health = client.health()
        except ApiClientError:
            st.error(
                "Servicio de inferencia no disponible. Verifica que la API esté en ejecución.",
                icon=":material/error:",
            )
            health = None
        else:
            if health.model_loaded:
                st.success("En línea · modelo listo", icon=":material/check_circle:")
            else:
                st.warning("En línea · modelo cargando", icon=":material/hourglass_top:")

    # --- Resolución de la imagen de entrada (subida o muestra seleccionada). ---
    image: Image.Image | None = None
    image_bytes: bytes | None = None
    filename = ""
    content_type = "image/png"

    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            image_bytes = raw
            filename = uploaded.name
            content_type = uploaded.type or "image/png"
        except Exception:
            st.error("No se pudo leer la imagen subida. Verifica que el archivo sea válido.")
    elif selected_sample:
        try:
            raw = (SAMPLES_DIR / selected_sample).read_bytes()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            image_bytes = raw
            filename = selected_sample
            content_type = "image/jpeg"
        except Exception:
            st.error("No se pudo cargar la muestra seleccionada.")

    if image is None or image_bytes is None:
        st.info("Sube una imagen dermatoscópica o selecciona una muestra de prueba para comenzar.")
        st.stop()

    # --- Diagnóstico clínico completo, vía el servicio de inferencia HTTP. ---
    try:
        started = time.perf_counter()
        with st.spinner("Analizando la lesión…"):
            result = client.diagnose(image_bytes, filename, content_type)
        round_trip_ms = (time.perf_counter() - started) * 1000
    except ApiClientError as err:
        st.error(f"No se pudo completar el diagnóstico: {err}")
        st.stop()

    # --- Resultados primero: tarjeta con severidad, confianza y latencia. ---
    badge_color, badge_icon = severity_badge(result.severity)
    with st.container(border=True):
        st.subheader(f"Diagnóstico: {result.label_name}")
        st.badge(result.severity.capitalize(), icon=badge_icon, color=badge_color)
        metric_conf, metric_lat, metric_total = st.columns(3)
        metric_conf.metric("Confianza", f"{result.confidence:.1%}")
        metric_lat.metric("Latencia (servidor)", f"{result.latency_ms:.0f} ms")
        metric_total.metric("Latencia (total)", f"{round_trip_ms:.0f} ms")

    # --- Comparativa visual: imagen original vs. mapa de calor Grad-CAM. ---
    overlay = decode_overlay(result.overlay_png_base64)
    col_original, col_heatmap = st.columns(2)
    with col_original:
        st.subheader("Imagen original")
        st.image(image, width=DISPLAY_IMAGE_WIDTH)
    with col_heatmap:
        st.subheader("Mapa de calor Grad-CAM")
        st.image(overlay, width=DISPLAY_IMAGE_WIDTH)
        st.caption("🔴 Mayor atención del modelo · 🔵 Menor atención")

    with st.expander("¿Cómo interpretar el mapa de calor?"):
        st.markdown(
            "Grad-CAM aplicado a un Vision Transformer es una aproximación: puede resaltar "
            "también piel circundante además de la lesión, no solo el área patológica exacta. "
            "Úsalo como apoyo visual complementario al diagnóstico, no como una delimitación "
            "precisa de los bordes de la lesión."
        )

    # --- Distribución de probabilidad para las 7 patologías HAM10000. ---
    st.subheader("Probabilidad por patología")
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

    st.warning(
        "Herramienta de apoyo a la decisión clínica (CDSS). No reemplaza el diagnóstico de un "
        "profesional médico: los resultados deben ser validados por un especialista.",
        icon=":material/warning:",
    )
