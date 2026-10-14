"""
=============================================================================
MÓDULO: mlflow_tracking.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Adaptador de MLOps (tracking)
RESPONSABILIDAD ÚNICA:
    Registrar los resultados de la evaluación del modelo (parámetros, métricas
    y artefactos) en un servidor MLflow, sin acoplar la lógica de evaluación al
    SDK de tracking. El import de `mlflow` es diferido: la evaluación funciona
    igual si el tracking está desactivado.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Nombre por defecto del experimento en MLflow.
DEFAULT_EXPERIMENT: str = "skin-lesion-classifier"

# Arquitectura e hiperparámetros del entrenamiento ORIGINAL del modelo base,
# documentados en su Model Card (Anwarkh1/Skin_Cancer-Image_Classification).
# Este proyecto NO entrena: consume el modelo pre-entrenado, por lo que estos
# valores son los del entrenamiento upstream (no medidos por nosotros).
PRETRAINED_ARCHITECTURE: Final[str] = "google/vit-base-patch16-224-in21k"
PRETRAINED_LEARNING_RATE: Final[float] = 1e-4
PRETRAINED_BATCH_SIZE: Final[int] = 32


def _log_params(mlflow: Any, document: dict[str, Any]) -> None:
    model = document.get("model", {})
    dataset = document.get("dataset", {})
    limit = dataset.get("limit")
    mlflow.log_params(
        {
            "model_id": str(model.get("id", "")),
            "model_revision": str(model.get("revision", "")),
            "architecture": PRETRAINED_ARCHITECTURE,
            "learning_rate": str(PRETRAINED_LEARNING_RATE),
            "batch_size": str(PRETRAINED_BATCH_SIZE),
            "dataset_id": str(dataset.get("id", "")),
            "dataset_revision": str(dataset.get("revision", "")),
            "dataset_split": str(dataset.get("split", "")),
            "dataset_limit": str(limit) if limit else "all",
        }
    )


def _log_scenarios(mlflow: Any, document: dict[str, Any]) -> None:
    for scenario, data in document.get("scenarios", {}).items():
        mlflow.log_metric(f"{scenario}.accuracy", float(data.get("accuracy", 0.0)))
        mlflow.log_metric(f"{scenario}.macro_f1", float(data.get("macro_f1", 0.0)))
        mlflow.log_metric(f"{scenario}.samples", float(data.get("samples", 0)))
        for metrics in data.get("per_class", []):
            code = str(metrics.get("code", "?"))
            mlflow.log_metric(f"{scenario}.{code}.precision", float(metrics.get("precision", 0.0)))
            mlflow.log_metric(f"{scenario}.{code}.recall", float(metrics.get("recall", 0.0)))
            mlflow.log_metric(f"{scenario}.{code}.f1", float(metrics.get("f1", 0.0)))
        for key, value in data.get("latency_seconds", {}).items():
            if isinstance(value, int | float):
                mlflow.log_metric(f"{scenario}.latency.{key}", float(value))


def _log_targets(mlflow: Any, document: dict[str, Any]) -> None:
    targets = document.get("targets", {})
    recall = targets.get("melanoma_recall", {})
    if recall:
        mlflow.log_metric("target.melanoma_recall.value", float(recall.get("value", 0.0)))
        mlflow.log_metric("target.melanoma_recall.goal", float(recall.get("target", 0.0)))
        mlflow.set_tag("target.melanoma_recall.met", str(recall.get("met", False)))
    latency = targets.get("latency", {})
    if latency:
        mlflow.log_metric("target.latency.value", float(latency.get("value", 0.0)))
        mlflow.set_tag("target.latency.met", str(latency.get("met", False)))


def _log_artifacts(mlflow: Any, output_dir: Path) -> None:
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".png"}:
            mlflow.log_artifact(str(path))


def log_evaluation(
    document: dict[str, Any],
    output_dir: Path,
    tracking_uri: str,
    experiment: str = DEFAULT_EXPERIMENT,
) -> str:
    """Registra una corrida de evaluación en MLflow y devuelve su `run_id`.

    Args:
        document: documento de resultados producido por `run_evaluation`.
        output_dir: directorio con `metrics.json`, `confusion_matrix.png` y overlays.
        tracking_uri: URI de MLflow (ej. `file:./mlruns` o `http://host:5000`).
        experiment: nombre del experimento en MLflow.

    Returns:
        str: identificador de la corrida (`run_id`).
    """
    import mlflow  # import diferido: solo cuando el tracking está activado

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as active_run:
        _log_params(mlflow, document)
        _log_scenarios(mlflow, document)
        _log_targets(mlflow, document)
        _log_artifacts(mlflow, output_dir)
        run_id: str = str(active_run.info.run_id)

    logger.info("Corrida registrada en MLflow: run_id=%s (tracking: %s)", run_id, tracking_uri)
    return run_id
