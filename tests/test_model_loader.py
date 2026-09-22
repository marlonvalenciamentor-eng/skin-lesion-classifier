import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from skin_lesion_classifier.inference import HAM10000_LABELS, InferenceService
from skin_lesion_classifier.model_loader import (
    EXPECTED_MODEL_LABELS,
    MODEL_ID,
    MODEL_REQUIRED_FILES,
    PROCESSOR_REQUIRED_FILES,
    ModelLoadingError,
    load_inference_service,
    resolve_model_source,
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
    fake_model.config.label2id = {label: idx for idx, label in enumerate(EXPECTED_MODEL_LABELS)}

    # Must pass without raising any exception
    validate_model_integrity(fake_model)


def test_validate_model_integrity_raises_on_invalid_id2label_indices() -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    # Indices are 10..16 instead of 0..6
    fake_model.config.id2label = {
        idx + 10: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)
    }

    with pytest.raises(ModelLoadingError, match="Fallo de integridad arquitectónica"):
        validate_model_integrity(fake_model)


def test_validate_model_integrity_raises_on_missing_label2id() -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    fake_model.config.id2label = {idx: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    fake_model.config.label2id = None

    with pytest.raises(ModelLoadingError, match="debe incluir el diccionario 'label2id'"):
        validate_model_integrity(fake_model)


def test_validate_model_integrity_raises_on_label2id_mismatched_keys() -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    fake_model.config.id2label = {idx: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    fake_model.config.label2id = {f"otra_clave_{i}": i for i in range(7)}

    with pytest.raises(ModelLoadingError, match="Las claves de 'label2id' no coinciden"):
        validate_model_integrity(fake_model)


def test_validate_model_integrity_raises_on_label2id_inconsistency() -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    fake_model.config.id2label = {idx: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    # Inconsistent label2id mapping
    labels_list = list(EXPECTED_MODEL_LABELS)
    fake_model.config.label2id = {labels_list[i]: (i + 1) % 7 for i in range(7)}

    with pytest.raises(ModelLoadingError, match="Inconsistencia biyectiva"):
        validate_model_integrity(fake_model)


@patch("skin_lesion_classifier.model_loader.ViTImageProcessor.from_pretrained")
@patch("skin_lesion_classifier.model_loader.ViTForImageClassification.from_pretrained")
def test_load_inference_service_success_mocked(
    mock_model_from_pretrained: MagicMock,
    mock_proc_from_pretrained: MagicMock,
) -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    fake_model.config.id2label = {idx: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    fake_model.config.label2id = {label: idx for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    mock_model_from_pretrained.return_value = fake_model
    mock_proc_from_pretrained.return_value = MagicMock()

    service = load_inference_service("fake-repo/fake-model", local_dir=None)
    assert isinstance(service, InferenceService)
    assert service._processor is not None


@patch("skin_lesion_classifier.model_loader.ViTForImageClassification.from_pretrained")
def test_load_inference_service_raises_on_nonexistent_model(
    mock_from_pretrained: MagicMock,
) -> None:
    mock_from_pretrained.side_effect = OSError("Model repo not found")

    with pytest.raises(ModelLoadingError, match="No fue posible cargar el modelo"):
        load_inference_service("nonexistent/invalid-model-path-12345", local_dir=None)


def test_resolve_model_source_prefers_local_when_complete(tmp_path: Path) -> None:
    for name in MODEL_REQUIRED_FILES:
        (tmp_path / name).write_text("{}")

    source = resolve_model_source(tmp_path, "remote/model", MODEL_REQUIRED_FILES)

    assert source == str(tmp_path)


def test_resolve_model_source_falls_back_when_incomplete(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")  # falta model.safetensors

    source = resolve_model_source(tmp_path, "remote/model", MODEL_REQUIRED_FILES)

    assert source == "remote/model"


def test_resolve_model_source_falls_back_when_local_dir_is_none() -> None:
    source = resolve_model_source(None, "remote/model", MODEL_REQUIRED_FILES)

    assert source == "remote/model"


@patch("skin_lesion_classifier.model_loader.ViTImageProcessor.from_pretrained")
@patch("skin_lesion_classifier.model_loader.ViTForImageClassification.from_pretrained")
def test_load_inference_service_uses_local_dir_when_complete(
    mock_model_from_pretrained: MagicMock,
    mock_proc_from_pretrained: MagicMock,
    tmp_path: Path,
) -> None:
    for name in (*MODEL_REQUIRED_FILES, *PROCESSOR_REQUIRED_FILES):
        (tmp_path / name).write_text("{}")
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    fake_model.config.id2label = {idx: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    fake_model.config.label2id = {label: idx for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    mock_model_from_pretrained.return_value = fake_model
    mock_proc_from_pretrained.return_value = MagicMock()

    load_inference_service("remote/model", local_dir=tmp_path)

    mock_model_from_pretrained.assert_called_once_with(str(tmp_path))
    mock_proc_from_pretrained.assert_called_once_with(str(tmp_path))


@patch("skin_lesion_classifier.model_loader.ViTImageProcessor.from_pretrained")
@patch("skin_lesion_classifier.model_loader.ViTForImageClassification.from_pretrained")
def test_load_inference_service_downloads_when_local_dir_missing(
    mock_model_from_pretrained: MagicMock,
    mock_proc_from_pretrained: MagicMock,
    tmp_path: Path,
) -> None:
    fake_model = MagicMock()
    fake_model.config.num_labels = 7
    fake_model.config.id2label = {idx: label for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    fake_model.config.label2id = {label: idx for idx, label in enumerate(EXPECTED_MODEL_LABELS)}
    mock_model_from_pretrained.return_value = fake_model
    mock_proc_from_pretrained.return_value = MagicMock()

    load_inference_service("remote/model", "remote/processor", local_dir=tmp_path / "missing")

    mock_model_from_pretrained.assert_called_once_with("remote/model")
    mock_proc_from_pretrained.assert_called_once_with("remote/processor")


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
