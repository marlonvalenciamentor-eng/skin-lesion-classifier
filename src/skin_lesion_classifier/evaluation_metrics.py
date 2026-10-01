"""
=============================================================================
MÓDULO: evaluation_metrics.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Funciones puras / SRP
RESPONSABILIDAD ÚNICA:
    Calcular métricas de clasificación multiclase (matriz de confusión,
    precisión, recall, F1, soporte y exactitud) sobre las 7 clases HAM10000 y
    comparar el recall de Melanoma contra la meta clínica del proyecto.
    Solo usa numpy: sin modelo, sin red y sin E/S.
=============================================================================
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from skin_lesion_classifier.labels import HAM10000_CODES

# Meta clínica: el recall de Melanoma debe superar estrictamente el 90 %.
MELANOMA_RECALL_TARGET: Final[float] = 0.90
MELANOMA_CODE: Final[str] = "mel"


class EvaluationMetricsError(ValueError):
    """Error de dominio para entradas inválidas al calcular métricas."""


@dataclass(frozen=True)
class ClassMetrics:
    """Métricas de una clase; las razones con denominador cero se definen como 0.0."""

    code: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationReport:
    """Reporte inmutable; la matriz tiene filas = real y columnas = predicho."""

    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: tuple[ClassMetrics, ...]
    accuracy: float
    macro_f1: float
    total: int


@dataclass(frozen=True)
class TargetCheck:
    """Comparación de un valor medido contra su meta."""

    name: str
    value: float
    target: float
    met: bool


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = HAM10000_CODES,
) -> npt.NDArray[np.int64]:
    """Matriz de confusión (filas = etiqueta real, columnas = predicha) en el orden de `labels`."""
    if len(y_true) != len(y_pred):
        raise EvaluationMetricsError(
            f"La longitud de y_true ({len(y_true)}) difiere de la de y_pred ({len(y_pred)})."
        )
    index = {label: position for position, label in enumerate(labels)}
    unknown = sorted({value for value in (*y_true, *y_pred) if value not in index})
    if unknown:
        raise EvaluationMetricsError(f"Etiqueta desconocida para la evaluación: {unknown}.")

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        matrix[index[true_label], index[predicted_label]] += 1
    return matrix


def build_report(y_true: Sequence[str], y_pred: Sequence[str]) -> ClassificationReport:
    """Construye el reporte completo sobre las clases HAM10000."""
    matrix = confusion_matrix(y_true, y_pred)
    true_positives = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    per_class: list[ClassMetrics] = []
    for position, code in enumerate(HAM10000_CODES):
        precision = _ratio(float(true_positives[position]), float(predicted[position]))
        recall = _ratio(float(true_positives[position]), float(support[position]))
        per_class.append(
            ClassMetrics(
                code=code,
                precision=precision,
                recall=recall,
                f1=_ratio(2 * precision * recall, precision + recall),
                support=int(support[position]),
            )
        )

    total = int(matrix.sum())
    return ClassificationReport(
        labels=HAM10000_CODES,
        confusion_matrix=tuple(tuple(int(cell) for cell in row) for row in matrix),
        per_class=tuple(per_class),
        accuracy=_ratio(float(true_positives.sum()), float(total)),
        macro_f1=sum(metrics.f1 for metrics in per_class) / len(per_class),
        total=total,
    )


def check_melanoma_recall(report: ClassificationReport) -> TargetCheck:
    """Compara el recall de Melanoma contra la meta (estrictamente mayor que 0.90)."""
    melanoma = next(metrics for metrics in report.per_class if metrics.code == MELANOMA_CODE)
    return TargetCheck(
        name="melanoma_recall",
        value=melanoma.recall,
        target=MELANOMA_RECALL_TARGET,
        met=melanoma.recall > MELANOMA_RECALL_TARGET,
    )
