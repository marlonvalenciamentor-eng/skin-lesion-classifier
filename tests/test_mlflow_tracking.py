from pathlib import Path
from typing import Any

import mlflow

from skin_lesion_classifier.mlflow_tracking import log_evaluation


def _document() -> dict[str, Any]:
    return {
        "model": {"id": "test/model", "revision": "abc123"},
        "dataset": {"id": "test/ds", "revision": "def456", "split": "test", "limit": 50},
        "scenarios": {
            "original": {
                "samples": 50,
                "accuracy": 0.83,
                "macro_f1": 0.69,
                "labels": ["mel", "nv"],
                "per_class": [
                    {"code": "mel", "precision": 0.49, "recall": 0.88, "f1": 0.63, "support": 20},
                    {"code": "nv", "precision": 0.95, "recall": 0.94, "f1": 0.94, "support": 30},
                ],
                "latency_seconds": {"p50": 0.05, "p95": 0.06},
                "targets": {},
            }
        },
        "targets": {
            "melanoma_recall": {
                "name": "melanoma_recall",
                "value": 0.88,
                "target": 0.90,
                "met": False,
            },
            "latency": {"name": "latency", "value": 0.06, "target": 3.0, "met": True},
        },
    }


def _output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    return output_dir


def test_log_evaluation_logs_params_metrics_and_targets(tmp_path: Path) -> None:
    # Arrange
    output_dir = _output_dir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    # Act
    run_id = log_evaluation(_document(), output_dir, tracking_uri)

    # Assert
    assert run_id
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.params["model_id"] == "test/model"
    assert run.data.params["architecture"] == "google/vit-base-patch16-224-in21k"
    assert run.data.params["learning_rate"] == "0.0001"
    assert run.data.params["batch_size"] == "32"
    assert run.data.metrics["original.accuracy"] == 0.83
    assert run.data.metrics["original.mel.recall"] == 0.88
    assert run.data.tags["target.melanoma_recall.met"] == "False"


def test_log_evaluation_records_artifacts(tmp_path: Path) -> None:
    # Arrange
    output_dir = _output_dir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    # Act
    run_id = log_evaluation(_document(), output_dir, tracking_uri)

    # Assert
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    assert any(artifact.path == "metrics.json" for artifact in client.list_artifacts(run_id))
