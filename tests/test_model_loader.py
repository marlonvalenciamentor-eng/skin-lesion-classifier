import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from skin_lesion_classifier.inference import HAM10000_LABELS
from skin_lesion_classifier.model_loader import (
    EXPECTED_MODEL_LABELS,
    MODEL_ID,
    ModelLoadingError,
    load_inference_service,
    validate_model_integrity,
)


def test_model_id_is_the_pretrained_ham10000_vit() -> None:
    assert MODEL_ID == "Anwarkh1/Skin_Cancer-Image_Classification"


def test_validate_model_integrity_raises_on_invalid_num_labels() -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 3  # HAM10000 requires exactly 7 classes

    with pytest.raises(ModelLoadingError, match="Fallo de integridad arquitectónica"):
        validate_model_integrity(fake_model)


def test_validate_model_integrity_raises_on_mismatched_labels() -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    # 7 classes but with arbitrary non-HAM10000 labels
    fake_model.config.id2label = {i: f"clase_arbitraria_{i}" for i in range(7)}

    with pytest.raises(ModelLoadingError, match="Fallo de integridad clínica"):
        validate_model_integrity(fake_model)


def test_validate_model_integrity_passes_on_valid_ham10000_labels() -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    fake_model.config.id2label = {idx: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)}

    # Must pass without raising any exception
    validate_model_integrity(fake_model)


@patch("skin_lesion_classifier.model_loader.ViTForImageClassification.from_pretrained")
def test_load_inference_service_raises_on_nonexistent_model(
    mock_from_pretrained: MagicMock,
) -> None:
    mock_from_pretrained.side_effect = OSError("Model repo not found")

    with pytest.raises(ModelLoadingError, match="No fue posible cargar el modelo"):
        load_inference_service("nonexistent/invalid-model-path-12345")


@pytest.mark.integration
def test_loaded_service_classifies_an_image_on_cpu_under_3_seconds() -> None:
    service = load_inference_service()
    image = Image.new("RGB", (600, 450), color=(180, 120, 100))
    service.predict(image)  # warm-up: first call pays one-off allocation costs

    started = time.perf_counter()
    prediction = service.predict(image)
    elapsed = time.perf_counter() - started

    assert prediction.label in HAM10000_LABELS
    assert len(prediction.probabilities) == 7
    assert elapsed < 3.0, f"CPU inference took {elapsed:.2f}s"
