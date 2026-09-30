from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from skin_lesion_classifier.schemas import DiagnosisResponse, ErrorResponse, HealthResponse

MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"


def _valid_probabilities() -> dict[str, float]:
    # Sums to exactly 1.0; "mel" is the argmax and matches `confidence` below.
    return {
        "akiec": 0.02,
        "bcc": 0.02,
        "bkl": 0.02,
        "df": 0.02,
        "mel": 0.90,
        "nv": 0.01,
        "vasc": 0.01,
    }


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": "mel",
        "label_name": "Melanoma",
        "severity": "maligno",
        "confidence": 0.90,
        "probabilities": _valid_probabilities(),
        "overlay_png_base64": "aGVsbG8=",
        "latency_ms": 123.4,
        "model_id": MODEL_ID,
    }
    payload.update(overrides)
    return payload


def test_valid_payload_passes() -> None:
    # Arrange
    payload = _valid_payload()

    # Act
    response = DiagnosisResponse(**payload)

    # Assert
    assert response.label == "mel"
    assert response.confidence == pytest.approx(0.90)
    assert response.probabilities["mel"] == pytest.approx(0.90)


def test_confidence_above_one_is_rejected() -> None:
    # Arrange
    payload = _valid_payload(confidence=1.5)

    # Act / Assert
    with pytest.raises(ValidationError):
        DiagnosisResponse(**payload)


def test_unknown_label_is_rejected() -> None:
    # Arrange
    payload = _valid_payload(label="xx")

    # Act / Assert
    with pytest.raises(ValidationError):
        DiagnosisResponse(**payload)


def test_missing_class_in_probabilities_is_rejected() -> None:
    # Arrange
    probabilities = _valid_probabilities()
    del probabilities["vasc"]
    payload = _valid_payload(probabilities=probabilities)

    # Act / Assert
    with pytest.raises(ValidationError, match="7 clases"):
        DiagnosisResponse(**payload)


def test_extra_field_is_rejected() -> None:
    # Arrange
    payload = _valid_payload(unexpected_field="not allowed")

    # Act / Assert
    with pytest.raises(ValidationError):
        DiagnosisResponse(**payload)


def test_probabilities_not_summing_to_one_is_rejected() -> None:
    # Arrange
    probabilities = _valid_probabilities()
    probabilities["mel"] = 0.50  # total now well below 1.0
    payload = _valid_payload(probabilities=probabilities)

    # Act / Assert
    with pytest.raises(ValidationError, match="sumar 1.0"):
        DiagnosisResponse(**payload)


def test_label_not_matching_argmax_is_rejected() -> None:
    # Arrange: "mel" is still the highest probability, but `label` claims "nv".
    payload = _valid_payload(label="nv", confidence=0.01)

    # Act / Assert
    with pytest.raises(ValidationError, match="mayor probabilidad"):
        DiagnosisResponse(**payload)


def test_confidence_not_matching_label_probability_is_rejected() -> None:
    # Arrange: label is the correct argmax, but confidence disagrees with its probability.
    payload = _valid_payload(confidence=0.10)

    # Act / Assert
    with pytest.raises(ValidationError, match="coincidir con la probabilidad"):
        DiagnosisResponse(**payload)


def test_response_is_frozen() -> None:
    # Arrange
    response = DiagnosisResponse(**_valid_payload())

    # Act / Assert
    with pytest.raises(ValidationError):
        response.confidence = 0.5


def test_health_response_accepts_ok_status() -> None:
    # Arrange / Act
    health = HealthResponse(status="ok", model_loaded=True, model_id=MODEL_ID)

    # Assert
    assert health.status == "ok"
    assert health.model_loaded is True


def test_health_response_rejects_extra_field() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        HealthResponse(status="ok", model_loaded=True, model_id=MODEL_ID, extra="nope")  # type: ignore[call-arg]


def test_error_response_round_trip() -> None:
    # Arrange / Act
    error = ErrorResponse(detail="Modelo no disponible", error_type="ModelLoadingError")

    # Assert
    assert error.detail == "Modelo no disponible"
    assert error.error_type == "ModelLoadingError"


def test_probabilities_mapping_type_is_plain_dict() -> None:
    # Arrange
    response = DiagnosisResponse(**_valid_payload())

    # Assert: consumers can treat it as a read-only mapping regardless of pydantic internals.
    assert isinstance(response.probabilities, Mapping)
