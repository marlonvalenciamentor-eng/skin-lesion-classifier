import numpy as np
import pytest

from skin_lesion_classifier.evaluation_metrics import (
    ClassificationReport,
    EvaluationMetricsError,
    TargetCheck,
    build_report,
    check_melanoma_recall,
    confusion_matrix,
)
from skin_lesion_classifier.labels import HAM10000_CODES


def test_confusion_matrix_counts_rows_as_true_and_columns_as_predicted() -> None:
    y_true = ["mel", "mel", "nv", "nv", "nv"]
    y_pred = ["mel", "nv", "nv", "nv", "mel"]

    matrix = confusion_matrix(y_true, y_pred)

    mel, nv = HAM10000_CODES.index("mel"), HAM10000_CODES.index("nv")
    assert matrix.shape == (7, 7)
    assert matrix[mel, mel] == 1
    assert matrix[mel, nv] == 1
    assert matrix[nv, nv] == 2
    assert matrix[nv, mel] == 1
    assert matrix.sum() == 5


def test_confusion_matrix_rejects_unknown_labels() -> None:
    with pytest.raises(EvaluationMetricsError, match="desconocida"):
        confusion_matrix(["mel", "xyz"], ["mel", "nv"])


def test_confusion_matrix_rejects_length_mismatch() -> None:
    with pytest.raises(EvaluationMetricsError, match="longitud"):
        confusion_matrix(["mel", "nv"], ["mel"])


def test_build_report_computes_per_class_precision_recall_f1_and_support() -> None:
    y_true = ["mel", "mel", "mel", "mel", "nv", "nv"]
    y_pred = ["mel", "mel", "mel", "nv", "nv", "mel"]

    report = build_report(y_true, y_pred)

    mel = next(m for m in report.per_class if m.code == "mel")
    assert mel.support == 4
    assert mel.recall == pytest.approx(0.75)
    assert mel.precision == pytest.approx(0.75)
    assert mel.f1 == pytest.approx(0.75)
    assert report.total == 6
    assert report.accuracy == pytest.approx(4 / 6)


def test_build_report_uses_zero_for_classes_without_samples_or_predictions() -> None:
    report = build_report(["mel", "mel"], ["mel", "mel"])

    vasc = next(m for m in report.per_class if m.code == "vasc")
    assert (vasc.precision, vasc.recall, vasc.f1, vasc.support) == (0.0, 0.0, 0.0, 0)


def test_build_report_orders_classes_as_ham10000_codes_and_macro_f1_averages_them() -> None:
    report = build_report(["mel", "nv"], ["mel", "nv"])

    assert tuple(m.code for m in report.per_class) == HAM10000_CODES
    assert report.labels == HAM10000_CODES
    assert report.macro_f1 == pytest.approx(2 / 7)


def test_build_report_on_empty_input_returns_zeroed_report() -> None:
    report = build_report([], [])

    assert report.total == 0
    assert report.accuracy == 0.0
    assert report.macro_f1 == 0.0


def test_report_confusion_matrix_is_immutable_tuple_of_tuples() -> None:
    report = build_report(["mel"], ["mel"])

    assert isinstance(report, ClassificationReport)
    assert isinstance(report.confusion_matrix, tuple)
    assert np.array(report.confusion_matrix).shape == (7, 7)


@pytest.mark.parametrize(
    ("recall", "met"),
    [(0.91, True), (0.90, False), (0.5, False)],
)
def test_check_melanoma_recall_requires_strictly_greater_than_target(
    recall: float, met: bool
) -> None:
    total = 100
    hits = round(recall * total)
    y_true = ["mel"] * total
    y_pred = ["mel"] * hits + ["nv"] * (total - hits)
    report = build_report(y_true, y_pred)

    check = check_melanoma_recall(report)

    assert isinstance(check, TargetCheck)
    assert check.value == pytest.approx(recall)
    assert check.target == 0.90
    assert check.met is met
