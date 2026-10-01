"""
=============================================================================
MÓDULO: evaluation.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Inyección de Dependencias / SRP
RESPONSABILIDAD ÚNICA:
    Orquestar la evaluación del modelo (CRISP-DM, fase de Evaluación) sobre el
    split de prueba de HAM10000: métricas por clase, sensibilidad de Melanoma,
    latencia por imagen, robustez (baja luminosidad y ruido gaussiano) y el
    subconjunto sin solapamiento con `train`. Genera `metrics.json`, la matriz
    de confusión en PNG y overlays Grad-CAM de muestra.

USO:
    uv run python -m skin_lesion_classifier.evaluation --output-dir reports/evaluation

NOTA: el split de prueba se solapa con `train` (mismo `image_id`), por lo que los
resultados del escenario `original` son un techo optimista, no una medida de
generalización.
=============================================================================
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from skin_lesion_classifier.evaluation_dataset import (
    DATASET_ID,
    DATASET_REVISION,
    TEST_PARQUET_FILENAME,
    LabeledImage,
    download_test_split,
    fetch_split_ids,
    iter_labeled_images,
)
from skin_lesion_classifier.evaluation_metrics import (
    ClassificationReport,
    TargetCheck,
    build_report,
    check_melanoma_recall,
)
from skin_lesion_classifier.facade import (
    DermatologyDiagnosticFacade,
    DiagnosticResult,
    PredictionProvider,
)
from skin_lesion_classifier.gradcam import GradCAMError
from skin_lesion_classifier.inference import InferenceError
from skin_lesion_classifier.model_loader import MODEL_ID, load_inference_service
from skin_lesion_classifier.perturbations import add_gaussian_noise, darken

logger = logging.getLogger(__name__)

MODEL_REVISION: Final[str] = "e37ebda4a662db26d9221c78f6c72b1cb8736ce0"
LATENCY_TARGET_SECONDS: Final[float] = 3.0
DEFAULT_OUTPUT_DIR: Final[Path] = Path("reports/evaluation")
DEFAULT_GRADCAM_SAMPLES: Final[int] = 6
METRICS_FILENAME: Final[str] = "metrics.json"

STATUS_PARTIAL: Final[str] = "partial"
STATUS_COMPLETE: Final[str] = "complete"
GRADCAM_OK: Final[str] = "ok"
GRADCAM_SKIPPED: Final[str] = "skipped"
GRADCAM_FAILED: Final[str] = "failed"

SCENARIO_ORIGINAL: Final[str] = "original"
SCENARIO_LOW_BRIGHTNESS: Final[str] = "low_brightness"
SCENARIO_GAUSSIAN_NOISE: Final[str] = "gaussian_noise"
SCENARIO_UNSEEN: Final[str] = "unseen_in_train"

DARKEN_FACTOR: Final[float] = 0.4
NOISE_SIGMA: Final[float] = 25.0
NOISE_SEED: Final[int] = 0

Transform = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class LatencyStats:
    """Resumen de latencia de inferencia por imagen, en segundos."""

    mean: float
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class EvaluationRun:
    """Resultado inmutable de evaluar un proveedor sobre un conjunto de muestras."""

    y_true: tuple[str, ...]
    y_pred: tuple[str, ...]
    image_ids: tuple[str, ...]
    latencies: tuple[float, ...]
    report: ClassificationReport
    latency: LatencyStats
    latency_check: TargetCheck


class Diagnoser(Protocol):
    """Contrato de la fachada de diagnóstico usada para los overlays Grad-CAM."""

    def diagnose(self, image: Image.Image, target_class: str | None = None) -> DiagnosticResult: ...


@dataclass(frozen=True)
class EvaluationDependencies:
    """Dependencias inyectables del CLI (modelo, datos, red y reloj)."""

    provider: PredictionProvider
    diagnoser: Diagnoser
    samples_factory: Callable[[int | None], Iterable[LabeledImage]]
    fetch_train_ids: Callable[[], frozenset[str]]
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    clock: Callable[[], float] = time.perf_counter


def _summarise_latency(latencies: Sequence[float]) -> LatencyStats:
    if not latencies:
        return LatencyStats(mean=0.0, p50=0.0, p95=0.0, max=0.0)
    values = np.asarray(latencies, dtype=np.float64)
    return LatencyStats(
        mean=float(values.mean()),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        max=float(values.max()),
    )


def _assemble_run(
    y_true: tuple[str, ...],
    y_pred: tuple[str, ...],
    image_ids: tuple[str, ...],
    latencies: tuple[float, ...],
) -> EvaluationRun:
    latency = _summarise_latency(latencies)
    return EvaluationRun(
        y_true=y_true,
        y_pred=y_pred,
        image_ids=image_ids,
        latencies=latencies,
        report=build_report(y_true, y_pred),
        latency=latency,
        latency_check=TargetCheck(
            name="latency_p95_seconds",
            value=latency.p95,
            target=LATENCY_TARGET_SECONDS,
            met=latency.p95 < LATENCY_TARGET_SECONDS,
        ),
    )


def evaluate(
    provider: PredictionProvider,
    samples: Iterable[LabeledImage],
    transform: Transform | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationRun:
    """Predice cada muestra (tras `transform`) midiendo solo la latencia de `predict`."""
    y_true: list[str] = []
    y_pred: list[str] = []
    image_ids: list[str] = []
    latencies: list[float] = []
    for sample in samples:
        image = transform(sample.image) if transform is not None else sample.image
        started = clock()
        prediction = provider.predict(image)
        latencies.append(clock() - started)
        y_true.append(sample.code)
        y_pred.append(prediction.label)
        image_ids.append(sample.image_id)
    return _assemble_run(tuple(y_true), tuple(y_pred), tuple(image_ids), tuple(latencies))


def select_subset(run: EvaluationRun, image_ids: frozenset[str]) -> EvaluationRun:
    """Reconstruye una corrida con solo las muestras cuyo `image_id` está en `image_ids`."""
    kept = [position for position, image_id in enumerate(run.image_ids) if image_id in image_ids]
    return _assemble_run(
        tuple(run.y_true[position] for position in kept),
        tuple(run.y_pred[position] for position in kept),
        tuple(run.image_ids[position] for position in kept),
        tuple(run.latencies[position] for position in kept),
    )


def _scenario_to_dict(run: EvaluationRun) -> dict[str, Any]:
    return {
        "samples": run.report.total,
        "accuracy": run.report.accuracy,
        "macro_f1": run.report.macro_f1,
        "labels": list(run.report.labels),
        "confusion_matrix": [list(row) for row in run.report.confusion_matrix],
        "per_class": [asdict(metrics) for metrics in run.report.per_class],
        "latency_seconds": asdict(run.latency),
        "targets": {
            "melanoma_recall": asdict(check_melanoma_recall(run.report)),
            "latency": asdict(run.latency_check),
        },
    }


def save_confusion_matrix_png(report: ClassificationReport, path: Path) -> None:
    """Guarda la matriz de confusión normalizada por fila, anotada con conteos."""
    counts = np.asarray(report.confusion_matrix, dtype=np.float64)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)

    figure = Figure(figsize=(7.5, 6.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes, label="Proporción por clase real")
    ticks = range(len(report.labels))
    axes.set_xticks(list(ticks), labels=list(report.labels))
    axes.set_yticks(list(ticks), labels=list(report.labels))
    axes.set_xlabel("Clase predicha")
    axes.set_ylabel("Clase real")
    axes.set_title("Matriz de confusión (escenario original, normalizada por fila)")
    for row in ticks:
        for column in ticks:
            axes.text(
                column,
                row,
                f"{int(counts[row, column])}\n{normalised[row, column]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalised[row, column] > 0.5 else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def save_gradcam_overlays(
    diagnoser: Diagnoser,
    samples: Iterable[LabeledImage],
    count: int,
    output_dir: Path,
) -> list[Path]:
    """Guarda overlays Grad-CAM de `count` muestras: mitad melanoma, mitad otras clases."""
    melanoma_quota = (count + 1) // 2
    other_quota = count - melanoma_quota
    melanoma_found = 0
    other_found = 0
    saved: list[Path] = []
    for sample in samples:
        if melanoma_found >= melanoma_quota and other_found >= other_quota:
            break
        if sample.code == "mel":
            if melanoma_found >= melanoma_quota:
                continue
            melanoma_found += 1
        else:
            if other_found >= other_quota:
                continue
            other_found += 1
        result = diagnoser.diagnose(sample.image)
        path = output_dir / f"gradcam_{sample.image_id}_true-{sample.code}_pred-{result.label}.png"
        result.superimposed_image.save(path)
        saved.append(path)
    return saved


def _targets_block(original: EvaluationRun) -> dict[str, Any]:
    return {
        "melanoma_recall": asdict(check_melanoma_recall(original.report)),
        "latency": asdict(original.latency_check),
    }


def _write_metrics_atomically(document: dict[str, Any], output_dir: Path) -> None:
    """Escribe `metrics.json` en un temporal del mismo directorio y lo reemplaza atómicamente."""
    target = output_dir / METRICS_FILENAME
    handle, temp_name = tempfile.mkstemp(dir=output_dir, prefix=".metrics-", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with open(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, ensure_ascii=False))
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _generate_gradcam_overlays(
    dependencies: EvaluationDependencies, gradcam_samples: int, output_dir: Path
) -> dict[str, Any]:
    """Genera los overlays Grad-CAM; al ser salida opcional, un error de dominio no aborta."""
    if gradcam_samples <= 0:
        return {"status": GRADCAM_SKIPPED}
    logger.info("Generando overlays Grad-CAM de muestra...")
    try:
        saved = save_gradcam_overlays(
            dependencies.diagnoser, dependencies.samples_factory(None), gradcam_samples, output_dir
        )
    except (GradCAMError, InferenceError) as err:
        logger.warning("No se pudieron generar los overlays Grad-CAM: %s", err)
        return {"status": GRADCAM_FAILED, "error": str(err)}
    return {"status": GRADCAM_OK, "samples": len(saved)}


def run_evaluation(
    dependencies: EvaluationDependencies,
    output_dir: Path,
    limit: int | None,
    gradcam_samples: int,
    skip_overlap: bool,
) -> dict[str, Any]:
    """Ejecuta los escenarios, persiste resultados parciales y devuelve el documento final.

    Los `image_id` de `train` se leen antes de evaluar (falla rápido sin red) y `metrics.json`
    se reescribe de forma atómica tras cada escenario con `status: partial` hasta el final.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = dependencies.provider
    clock = dependencies.clock

    train_ids: frozenset[str] = frozenset()
    if not skip_overlap:
        logger.info("Leyendo los image_id de 'train' para el subconjunto sin solapamiento...")
        train_ids = dependencies.fetch_train_ids()

    scenarios: dict[str, EvaluationRun] = {}
    document: dict[str, Any] = {
        "generated_at": dependencies.now().isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "test",
            "file": TEST_PARQUET_FILENAME,
            "limit": limit,
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "perturbations": {
            SCENARIO_LOW_BRIGHTNESS: {"factor": DARKEN_FACTOR},
            SCENARIO_GAUSSIAN_NOISE: {"sigma": NOISE_SIGMA, "seed": NOISE_SEED},
        },
        "overlap": {"skipped": skip_overlap},
        "scenarios": {},
        "status": STATUS_PARTIAL,
        "completed_scenarios": [],
    }

    def record(name: str, run: EvaluationRun) -> None:
        scenarios[name] = run
        document["scenarios"][name] = _scenario_to_dict(run)
        document["completed_scenarios"].append(name)
        _write_metrics_atomically(document, output_dir)

    transforms: dict[str, Transform | None] = {
        SCENARIO_ORIGINAL: None,
        SCENARIO_LOW_BRIGHTNESS: lambda image: darken(image, DARKEN_FACTOR),
        SCENARIO_GAUSSIAN_NOISE: lambda image: add_gaussian_noise(image, NOISE_SIGMA, NOISE_SEED),
    }
    for name, transform in transforms.items():
        logger.info("Evaluando escenario '%s'...", name)
        run = evaluate(
            provider, dependencies.samples_factory(limit), transform=transform, clock=clock
        )
        if name == SCENARIO_ORIGINAL:
            # `targets` solo existe una vez completado `original`; antes no hay archivo.
            document["targets"] = _targets_block(run)
            save_confusion_matrix_png(run.report, output_dir / "confusion_matrix.png")
        record(name, run)

    original = scenarios[SCENARIO_ORIGINAL]
    if not skip_overlap:
        unseen_ids = frozenset(original.image_ids) - train_ids
        document["overlap"].update(
            test_samples=original.report.total,
            in_train=original.report.total - len(unseen_ids),
            not_in_train=len(unseen_ids),
        )
        record(SCENARIO_UNSEEN, select_subset(original, unseen_ids))

    document["gradcam"] = _generate_gradcam_overlays(dependencies, gradcam_samples, output_dir)
    document["status"] = STATUS_COMPLETE
    _write_metrics_atomically(document, output_dir)
    return document


def build_default_dependencies() -> EvaluationDependencies:
    """Carga el modelo fijado y el parquet del caché de HF (requiere red la primera vez)."""
    service = load_inference_service(revision=MODEL_REVISION, local_dir=None)
    parquet_path = download_test_split()
    return EvaluationDependencies(
        provider=service,
        diagnoser=DermatologyDiagnosticFacade(inference_service=service),
        samples_factory=lambda limit: iter_labeled_images(parquet_path, limit=limit),
        fetch_train_ids=lambda: fetch_split_ids("train"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m skin_lesion_classifier.evaluation",
        description="Evalúa el modelo ViT sobre el split de prueba de HAM10000.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Máximo de imágenes a evaluar.")
    parser.add_argument("--gradcam-samples", type=int, default=DEFAULT_GRADCAM_SAMPLES)
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="No leer los image_id de train (omite el subconjunto sin solapamiento).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    dependencies: EvaluationDependencies | None = None,
) -> int:
    """Punto de entrada del CLI; devuelve el código de salida del proceso."""
    args = _build_parser().parse_args(argv)
    resolved = dependencies if dependencies is not None else build_default_dependencies()
    document = run_evaluation(
        resolved,
        output_dir=args.output_dir,
        limit=args.limit,
        gradcam_samples=args.gradcam_samples,
        skip_overlap=args.skip_overlap,
    )
    recall = document["targets"]["melanoma_recall"]
    latency = document["targets"]["latency"]
    logger.info(
        "Recall de melanoma: %.3f (meta > %.2f, cumplida: %s). "
        "Latencia p95: %.3f s (meta < %.1f s, cumplida: %s). Resultados en '%s'.",
        recall["value"],
        recall["target"],
        recall["met"],
        latency["value"],
        latency["target"],
        latency["met"],
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
