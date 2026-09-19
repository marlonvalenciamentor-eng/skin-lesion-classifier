import time

import pytest
from PIL import Image

from skin_lesion_classifier.inference import HAM10000_LABELS
from skin_lesion_classifier.model_loader import MODEL_ID, load_inference_service

pytestmark = pytest.mark.integration


def test_model_id_is_the_pretrained_ham10000_vit() -> None:
    assert MODEL_ID == "Anwarkh1/Skin_Cancer-Image_Classification"


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
