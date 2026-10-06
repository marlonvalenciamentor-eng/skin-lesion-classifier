"""
=============================================================================
MÓDULO: report.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Generador de Reportes (Presentación)
RESPONSABILIDAD ÚNICA:
    Generar un reporte PDF (A4) con los datos del diagnóstico dermatológico:
    datos del paciente, diagnóstico, severidad, confianza, imagen original,
    mapa de calor Grad-CAM y distribución de probabilidades HAM10000.

    No depende de 'torch' ni 'transformers': recibe la imagen original y el
    overlay ya procesados por la fachada, y solo dibuja el documento.
=============================================================================
"""

import io
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from PIL import Image

from skin_lesion_classifier.labels import human_name


def build_report_pdf(
    patient_name: str,
    image: Image.Image,
    overlay: Image.Image,
    label_name: str,
    severity: str,
    confidence: float,
    probabilities: dict[str, float],
) -> bytes:
    """Genera un reporte PDF (A4) con los datos del diagnóstico dermatológico.

    Args:
        patient_name: Nombre del paciente (puede ser vacío).
        image: Imagen dermatoscópica original.
        overlay: Mapa de calor Grad-CAM superpuesto.
        label_name: Nombre legible de la patología Top-1.
        severity: Nivel de severidad clínica (maligno/precanceroso/benigno).
        confidence: Confianza del diagnóstico Top-1 en [0.0, 1.0].
        probabilities: Distribución de probabilidad de las 7 clases HAM10000.

    Returns:
        bytes: Contenido del PDF generado.
    """
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    figure = Figure(figsize=(8.27, 11.69))  # A4 portrait
    canvas = FigureCanvasPdf(figure)

    figure.suptitle("Reporte de diagnóstico dermatológico", fontsize=16, fontweight="bold")

    info_text = (
        f"Paciente: {patient_name.strip() or 'No registrado'}\n"
        f"Fecha: {now}\n"
        f"Diagnóstico: {label_name}\n"
        f"Severidad: {severity}\n"
        f"Confianza: {confidence:.1%}\n"
    )
    figure.text(0.5, 0.90, info_text, ha="center", va="top", fontsize=12)

    ax_image = figure.add_axes((0.05, 0.56, 0.42, 0.28))
    ax_image.imshow(np.asarray(image.convert("RGB")))
    ax_image.set_title("Imagen original", fontsize=11)
    ax_image.axis("off")

    ax_overlay = figure.add_axes((0.53, 0.56, 0.42, 0.28))
    ax_overlay.imshow(np.asarray(overlay.convert("RGB")))
    ax_overlay.set_title("Mapa de calor Grad-CAM", fontsize=11)
    ax_overlay.axis("off")

    probability_text = "Probabilidades por patología:\n" + "\n".join(
        f"  {human_name(code)}: {probability:.1%}" for code, probability in ordered
    )
    figure.text(0.08, 0.48, probability_text, ha="left", va="top", fontsize=10, family="monospace")

    buffer = io.BytesIO()
    # matplotlib no publica anotaciones de tipo para print_pdf.
    canvas.print_pdf(buffer)  # type: ignore[no-untyped-call]
    return buffer.getvalue()
