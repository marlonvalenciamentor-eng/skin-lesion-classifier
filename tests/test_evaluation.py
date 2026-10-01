import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from skin_lesion_classifier.evaluation import (
    EvaluationDependencies,
    EvaluationRun,
    evaluate,
    main,
    select_subset,
)
from skin_lesion_classifier.evaluation_dataset import DATASET_REVISION, LabeledImage
from skin_lesion_classifier.facade import DiagnosticResult
from skin_lesion_classifier.inference import Prediction
from skin_lesion_classifier.labels import HAM10000_CODES


def make_prediction(label: str) -> Prediction:
    probabilities = {code: 0.0 for code in HAM10000_CODES}
    probabilities[label] = 1.0
    return Prediction(label=label, confidence=1.0, probabilities=probabilities)


def make_sample(image_id: str, code: str) -> LabeledImage:
    # The width encodes the true class so fakes can "predict" without a model.
    width = 10 + HAM10000_CODES.index(code)
    return LabeledImage(
        image_id=image_id,
        lesion_id=f"HAM_{image_id}",
        code=code,
        image=Image.new("RGB", (width, 8), (120, 90, 70)),
    )


class QueueProvider:
    """Returns queued labels in call order."""

    def __init__(self, labels: list[str]) -> None:
        self._labels: Iterator[str] = iter(labels)
        self.received_sizes: list[tuple[int, int]] = []

    def predict(self, image: Image.Image) -> Prediction:
        self.received_sizes.append(image.size)
        return make_prediction(next(self._labels))


class WidthProvider:
    """Predicts the class encoded in the image width (perfect model)."""

    def predict(self, image: Image.Image) -> Prediction:
        return make_prediction(HAM10000_CODES[image.size[0] - 10])


class WidthDiagnoser:
    def diagnose(self, image: Image.Image, target_class: str | None = None) -> DiagnosticResult:
        label = HAM10000_CODES[image.size[0] - 10]
        return DiagnosticResult(
            label=label,
            confidence=1.0,
            probabilities=make_prediction(label).probabilities,
            heatmap=np.zeros((4, 4), dtype=np.float32),
            superimposed_image=Image.new("RGB", (16, 16), (1, 2, 3)),
            target_class=label,
        )


class FakeClock:
    def __init__(self, step: float) -> None:
        self._step = step
        self._now = 0.0
        self._calls = 0

    def __call__(self) -> float:
        # Calls alternate start/end, so each measured interval lasts exactly `step`.
        self._calls += 1
        if self._calls % 2 == 0:
            self._now += self._step
        return self._now


def test_evaluate_collects_labels_predictions_ids_and_report() -> None:
    samples = [make_sample("a", "mel"), make_sample("b", "nv"), make_sample("c", "mel")]
    provider = QueueProvider(["mel", "nv", "nv"])

    run = evaluate(provider, samples, clock=FakeClock(0.1))

    assert isinstance(run, EvaluationRun)
    assert run.y_true == ("mel", "nv", "mel")
    assert run.y_pred == ("mel", "nv", "nv")
    assert run.image_ids == ("a", "b", "c")
    assert run.report.total == 3
    assert run.report.accuracy == pytest.approx(2 / 3)


def test_evaluate_measures_latency_per_image_and_summarises_it() -> None:
    samples = [make_sample(str(n), "nv") for n in range(4)]
    clock = FakeClock(0.5)

    run = evaluate(WidthProvider(), samples, clock=clock)

    assert run.latencies == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert run.latency.mean == pytest.approx(0.5)
    assert run.latency.p50 == pytest.approx(0.5)
    assert run.latency.p95 == pytest.approx(0.5)
    assert run.latency.max == pytest.approx(0.5)
    assert run.latency_check.met is True
    assert run.latency_check.target == 3.0


def test_evaluate_flags_latency_target_not_met_when_p95_reaches_three_seconds() -> None:
    samples = [make_sample(str(n), "nv") for n in range(3)]

    run = evaluate(WidthProvider(), samples, clock=FakeClock(3.0))

    assert run.latency_check.met is False


def test_evaluate_applies_transform_before_predicting() -> None:
    samples = [make_sample("a", "nv")]
    provider = QueueProvider(["nv"])

    evaluate(provider, samples, transform=lambda image: image.resize((50, 40)))

    assert provider.received_sizes == [(50, 40)]


def test_evaluate_with_no_samples_returns_empty_zeroed_run() -> None:
    run = evaluate(WidthProvider(), [])

    assert run.report.total == 0
    assert run.latency.mean == 0.0
    assert run.latency_check.met is True


def test_select_subset_keeps_only_requested_ids_and_rebuilds_report() -> None:
    samples = [make_sample("a", "mel"), make_sample("b", "nv"), make_sample("c", "mel")]
    run = evaluate(QueueProvider(["mel", "nv", "nv"]), samples, clock=FakeClock(1.0))

    subset = select_subset(run, frozenset({"a", "c"}))

    assert subset.image_ids == ("a", "c")
    assert subset.y_true == ("mel", "mel")
    assert subset.y_pred == ("mel", "nv")
    assert subset.latencies == pytest.approx((1.0, 1.0))
    assert subset.report.total == 2


SAMPLES = [
    make_sample("img1", "mel"),
    make_sample("img2", "mel"),
    make_sample("img3", "nv"),
    make_sample("img4", "nv"),
    make_sample("img5", "bcc"),
]


def make_dependencies(
    train_ids: frozenset[str] = frozenset({"img1", "img3"}),
    fetch_calls: list[int] | None = None,
) -> EvaluationDependencies:
    def samples_factory(limit: int | None) -> Iterable[LabeledImage]:
        return SAMPLES if limit is None else SAMPLES[:limit]

    def fetch_train_ids() -> frozenset[str]:
        if fetch_calls is not None:
            fetch_calls.append(1)
        return train_ids

    return EvaluationDependencies(
        provider=WidthProvider(),
        diagnoser=WidthDiagnoser(),
        samples_factory=samples_factory,
        fetch_train_ids=fetch_train_ids,
        now=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )


def test_main_writes_metrics_confusion_matrix_and_gradcam_overlays(tmp_path: Path) -> None:
    out = tmp_path / "out"

    exit_code = main(["--output-dir", str(out), "--gradcam-samples", "2"], make_dependencies())

    assert exit_code == 0
    assert (out / "metrics.json").is_file()
    assert (out / "confusion_matrix.png").stat().st_size > 0
    overlays = sorted(path.name for path in out.glob("gradcam_*.png"))
    assert len(overlays) == 2
    assert any("true-mel" in name for name in overlays)
    assert any("true-mel" not in name for name in overlays)


def test_main_metrics_json_contains_scenarios_targets_and_provenance(tmp_path: Path) -> None:
    out = tmp_path / "out"

    main(["--output-dir", str(out), "--gradcam-samples", "0"], make_dependencies())

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["scenarios"]) == {
        "original",
        "low_brightness",
        "gaussian_noise",
        "unseen_in_train",
    }
    assert metrics["scenarios"]["original"]["samples"] == 5
    assert metrics["scenarios"]["original"]["accuracy"] == 1.0
    assert metrics["scenarios"]["unseen_in_train"]["samples"] == 3
    assert metrics["targets"]["melanoma_recall"]["met"] is True
    assert metrics["targets"]["latency"]["target"] == 3.0
    assert metrics["dataset"]["revision"] == DATASET_REVISION
    assert metrics["model"]["revision"] == "e37ebda4a662db26d9221c78f6c72b1cb8736ce0"
    assert metrics["overlap"] == {
        "skipped": False,
        "test_samples": 5,
        "in_train": 2,
        "not_in_train": 3,
    }
    assert metrics["generated_at"] == "2026-10-01T12:00:00+00:00"


def test_main_skip_overlap_omits_unseen_subset_and_never_fetches_train_ids(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    fetch_calls: list[int] = []

    main(
        ["--output-dir", str(out), "--gradcam-samples", "0", "--skip-overlap"],
        make_dependencies(fetch_calls=fetch_calls),
    )

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "unseen_in_train" not in metrics["scenarios"]
    assert metrics["overlap"]["skipped"] is True
    assert fetch_calls == []


def test_main_limit_restricts_the_number_of_evaluated_samples(tmp_path: Path) -> None:
    out = tmp_path / "out"

    main(
        ["--output-dir", str(out), "--limit", "2", "--gradcam-samples", "0", "--skip-overlap"],
        make_dependencies(),
    )

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["scenarios"]["original"]["samples"] == 2
