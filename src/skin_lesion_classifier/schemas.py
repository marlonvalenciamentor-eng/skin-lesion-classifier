"""
=============================================================================
MÓDULO: schemas.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Contrato de Datos (DTO) entre procesos
RESPONSABILIDAD ÚNICA:
    Definir los contratos Pydantic v2 que cruzan el límite HTTP entre el
    servicio de inferencia (FastAPI) y la interfaz web (Streamlit). El módulo
    es deliberadamente puro (sin `torch`, `numpy` ni `transformers`) para que
    ambos procesos —incluida una instalación ligera solo-UI— puedan importarlo
    y validar la misma forma de datos sin instalar el stack de Deep Learning.
=============================================================================
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Código corto HAM10000 (7 clases). Debe permanecer sincronizado con
# `skin_lesion_classifier.inference.HAM10000_LABELS` y `labels.HAM10000_CODES`.
LesionCode = Literal["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Niveles de severidad clínica. Debe permanecer sincronizado con
# `skin_lesion_classifier.labels.Severity`.
Severity = Literal["maligno", "precanceroso", "benigno"]

_EXPECTED_CODES: frozenset[str] = frozenset({"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"})

# Tolerancias de validación cruzada (redondeo de punto flotante en el servidor).
_PROBABILITY_SUM_TOLERANCE = 1e-2
_CONFIDENCE_TOLERANCE = 1e-2


class DiagnosisResponse(BaseModel):
    """
    Contrato inmutable de diagnóstico devuelto por `POST /predict`.

    Se autovalida para garantizar consistencia clínica del payload: las 7 clases
    HAM10000 deben estar presentes, las probabilidades deben sumar 1.0 (con
    tolerancia) y `label`/`confidence` deben coincidir con la clase de mayor
    probabilidad. Esto hace que un servidor con un bug de postprocesamiento
    falle rápido en el contrato en vez de propagar datos inconsistentes a la UI.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: LesionCode
    label_name: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[LesionCode, float]
    overlay_png_base64: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    model_id: str

    @model_validator(mode="after")
    def _check_probability_contract(self) -> Self:
        codes = set(self.probabilities)
        if codes != _EXPECTED_CODES:
            missing = _EXPECTED_CODES - codes
            extra = codes - _EXPECTED_CODES
            raise ValueError(
                "El diccionario de probabilidades debe cubrir exactamente las 7 clases "
                f"HAM10000. Faltantes: {missing or 'ninguna'}, inesperadas: {extra or 'ninguna'}."
            )

        for code, value in self.probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"La probabilidad de '{code}' debe estar en [0.0, 1.0], recibido {value}."
                )

        total = sum(self.probabilities.values())
        if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                f"Las probabilidades deben sumar 1.0 (tolerancia {_PROBABILITY_SUM_TOLERANCE}); "
                f"suma observada: {total}."
            )

        argmax_label = max(self.probabilities, key=lambda code: self.probabilities[code])
        if argmax_label != self.label:
            raise ValueError(
                f"'label' ('{self.label}') debe coincidir con la clase de mayor probabilidad "
                f"('{argmax_label}')."
            )

        if abs(self.probabilities[self.label] - self.confidence) > _CONFIDENCE_TOLERANCE:
            raise ValueError(
                "'confidence' debe coincidir con la probabilidad de 'label' "
                f"(tolerancia {_CONFIDENCE_TOLERANCE})."
            )

        return self


class HealthResponse(BaseModel):
    """Contrato inmutable devuelto por `GET /health`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok"]
    model_loaded: bool
    model_id: str


class ErrorResponse(BaseModel):
    """
    Envelope único de error para todas las respuestas no exitosas de la API.

    Se usa tanto en los manejadores de excepción de FastAPI como en el cliente
    HTTP (`api_client.py`), que lo parsea para construir mensajes en español
    sin filtrar tracebacks del servidor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detail: str
    error_type: str
