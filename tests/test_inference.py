import torch
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from skin_lesion_classifier.inference import HAM10000_LABELS, InferenceService

# Same id2label as Anwarkh1/Skin_Cancer-Image_Classification, so the mapping is exercised for real.
MODEL_ID2LABEL = {
    0: "benign_keratosis-like_lesions",
    1: "basal_cell_carcinoma",
    2: "actinic_keratoses",
    3: "vascular_lesions",
    4: "melanocytic_Nevi",
    5: "melanoma",
    6: "dermatofibroma",
}


def make_service(seed: int = 0) -> InferenceService:
    torch.manual_seed(seed)
    config = ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=224,
        num_labels=7,
        id2label=MODEL_ID2LABEL,
        label2id={v: k for k, v in MODEL_ID2LABEL.items()},
    )
    return InferenceService(model=ViTForImageClassification(config), processor=ViTImageProcessor())


def test_predict_returns_top1_label_from_ham10000_codes() -> None:
    service = make_service()
    image = Image.new("RGB", (300, 200), color=(120, 80, 60))

    prediction = service.predict(image)

    assert prediction.label in HAM10000_LABELS


def test_predict_exposes_probability_for_each_of_the_7_classes() -> None:
    service = make_service()
    image = Image.new("RGB", (224, 224), color=(200, 150, 130))

    prediction = service.predict(image)

    assert set(prediction.probabilities) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-5


def test_predict_label_and_confidence_match_the_highest_probability() -> None:
    service = make_service(seed=7)
    image = Image.new("RGB", (224, 224), color=(50, 50, 50))

    prediction = service.predict(image)

    best_label = max(prediction.probabilities, key=prediction.probabilities.get)
    assert prediction.label == best_label
    assert prediction.confidence == prediction.probabilities[best_label]


def test_predict_raises_on_invalid_input_type() -> None:
    import pytest

    from skin_lesion_classifier.inference import InferenceError

    service = make_service()
    with pytest.raises(InferenceError, match="Entrada inválida"):
        service.predict("not_an_image_path_or_object")  # type: ignore[arg-type]


def test_predict_raises_on_zero_dimension_image() -> None:
    import pytest

    from skin_lesion_classifier.inference import InferenceError

    service = make_service()
    empty_image = Image.new("RGB", (0, 0))
    with pytest.raises(InferenceError, match="dimensiones nulas"):
        service.predict(empty_image)


def test_predict_raises_inference_error_on_corrupted_model_output() -> None:
    from unittest.mock import MagicMock

    import pytest

    from skin_lesion_classifier.inference import InferenceError

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    # Corrupt logits with empty tensor that fails indexing
    fake_model.return_value.logits = torch.empty((0,))

    fake_processor = MagicMock()
    fake_processor.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}

    service = InferenceService(model=fake_model, processor=fake_processor)
    image = Image.new("RGB", (224, 224))

    with pytest.raises(InferenceError, match="Error al postprocesar"):
        service.predict(image)
