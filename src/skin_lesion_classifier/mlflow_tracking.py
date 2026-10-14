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
import subprocess
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
# Curvas de validación del entrenamiento original (Model Card, épocas 1..5).
PRETRAINED_VAL_LOSS: Final[tuple[float, ...]] = (0.4994, 0.3237, 0.1790, 0.1498, 0.1000)
PRETRAINED_VAL_ACCURACY: Final[tuple[float, ...]] = (0.8355, 0.8973, 0.9530, 0.9555, 0.9695)

# Metadatos del proyecto, registrados como tags de la corrida.
PROJECT_NAME: Final[str] = "SkinLesionClassifier"
PROJECT_INSTITUTION: Final[str] = "Universidad Autónoma de Occidente (UAO)"
PROJECT_COURSE: Final[str] = "Desarrollo de Soluciones de IA II — Módulo 2"
PROJECT_TEAM: Final[str] = "Miguel Ángel Ortiz Roldán, Marlon Valencia Velosa"


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


def _log_summary_metrics(mlflow: Any, document: dict[str, Any]) -> None:
    """Métricas de nivel superior (escenario original) para comparar corridas entre sí."""
    original = document.get("scenarios", {}).get("original", {})
    if not original:
        return
    mlflow.log_metric("accuracy", float(original.get("accuracy", 0.0)))
    mlflow.log_metric("f1_score", float(original.get("macro_f1", 0.0)))
    for key, value in original.get("latency_seconds", {}).items():
        if isinstance(value, int | float):
            mlflow.log_metric(f"latency_{key}", float(value))


def _log_pretrained_training(mlflow: Any) -> None:
    """Curvas de validación del entrenamiento upstream (Model Card), por época."""
    for epoch, loss in enumerate(PRETRAINED_VAL_LOSS, start=1):
        mlflow.log_metric("pretrained.val_loss", loss, step=epoch)
    for epoch, accuracy in enumerate(PRETRAINED_VAL_ACCURACY, start=1):
        mlflow.log_metric("pretrained.val_accuracy", accuracy, step=epoch)


def _git_commit_hash() -> str | None:
    """Hash corto del commit actual, o `None` si no hay repositorio git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _log_project_metadata(mlflow: Any) -> None:
    """Registra los metadatos del proyecto como tags de la corrida."""
    tags = {
        "project.name": PROJECT_NAME,
        "project.institution": PROJECT_INSTITUTION,
        "project.course": PROJECT_COURSE,
        "project.team": PROJECT_TEAM,
    }
    commit = _git_commit_hash()
    if commit is not None:
        tags["git.commit"] = commit
    mlflow.set_tags(tags)


def _log_artifacts(mlflow: Any, output_dir: Path) -> None:
    """Sube los artefactos de evaluación: metrics.json, matriz de confusión y overlays Grad-CAM."""
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".png"}:
            mlflow.log_artifact(str(path))


def _log_model_weights(mlflow: Any, model_dir: Path) -> None:
    """Registra los pesos del modelo local como artefactos (versionado del modelo)."""
    if not model_dir.is_dir():
        logger.warning("Directorio de modelo '%s' inexistente; no se registran pesos.", model_dir)
        return
    for path in sorted(model_dir.glob("*")):
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path="model")


def log_evaluation(
    document: dict[str, Any],
    output_dir: Path,
    tracking_uri: str,
    model_dir: Path | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
) -> str:
    """Registra una corrida de evaluación en MLflow y devuelve su `run_id`.

    Args:
        document: documento de resultados producido por `run_evaluation`.
        output_dir: directorio con `metrics.json`, `confusion_matrix.png` y overlays.
        tracking_uri: URI de MLflow (ej. `sqlite:///mlflow.db` o `http://host:5000`).
        model_dir: directorio local con los pesos del modelo (opcional; se suben como artefactos).
        experiment: nombre del experimento en MLflow.

    Returns:
        str: identificador de la corrida (`run_id`).
    """
    import mlflow  # import diferido: solo cuando el tracking está activado

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as active_run:
        _log_project_metadata(mlflow)
        _log_params(mlflow, document)
        _log_summary_metrics(mlflow, document)
        _log_scenarios(mlflow, document)
        _log_targets(mlflow, document)
        _log_pretrained_training(mlflow)
        _log_artifacts(mlflow, output_dir)
        if model_dir is not None:
            _log_model_weights(mlflow, model_dir)
        run_id: str = str(active_run.info.run_id)

    logger.info("Corrida registrada en MLflow: run_id=%s (tracking: %s)", run_id, tracking_uri)
    return run_id
