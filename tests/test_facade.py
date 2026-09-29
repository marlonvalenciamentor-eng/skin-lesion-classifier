from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from skin_lesion_classifier.facade import DermatologyDiagnosticFacade, DiagnosticResult
from skin_lesion_classifier.gradcam import GradCAMError, GradCAMResult, ViTGradCAM
from skin_lesion_classifier.inference import InferenceError, InferenceService, Prediction

# Distribution of the 7 HAM10000 classes that sums to 1.0
SEVEN_CLASS_PROBABILITIES = {
    "akiec": 0.02,
    "bcc": 0.03,
    "bkl": 0.02,
    "df": 0.01,
    "mel": 0.90,
    "nv": 0.01,
    "vasc": 0.01,
}


def make_prediction(label: str = "mel", confidence: float = 0.9) -> Prediction:
    probabilities = dict(SEVEN_CLASS_PROBABILITIES)
    probabilities[label] = confidence
    return Prediction(label=label, confidence=confidence, probabilities=probabilities)


def make_explanation(target_class: str = "mel") -> GradCAMResult:
    heatmap = np.zeros((224, 224), dtype=np.float32)
    return GradCAMResult(
        heatmap=heatmap,
        superimposed_image=Image.new("RGB", (224, 224), (10, 20, 30)),
        target_class=target_class,
    )


def make_image() -> Image.Image:
    return Image.new("RGB", (300, 200), color=(120, 80, 60))


class FakePredictionProvider:
    """Predictions provider that never touches torch or transformers."""

    def __init__(self, prediction: Prediction | None = None) -> None:
        self._prediction = prediction or make_prediction()
        self.predict_calls = 0

    def predict(self, image: Image.Image) -> Prediction:
        del image
        self.predict_calls += 1
        return self._prediction


class FakeExplainer:
    """Grad-CAM provider that records the requested target class."""

    def __init__(self, explanation: GradCAMResult | None = None) -> None:
        self._explanation = explanation or make_explanation()
        self.explain_calls = 0
        self.last_target_class: str | None = None

    def explain(self, image: Image.Image, target_class: str | None = None) -> GradCAMResult:
        del image
        self.explain_calls += 1
        self.last_target_class = target_class
        return self._explanation


def test_diagnose_combines_prediction_and_explanation_into_diagnostic_result() -> None:
    # Arrange
    prediction = make_prediction(label="bcc", confidence=0.85)
    explanation = make_explanation(target_class="bcc")
    facade = DermatologyDiagnosticFacade(
        inference_service=FakePredictionProvider(prediction),
        explainer=FakeExplainer(explanation),
    )
    image = make_image()

    # Act
    result = facade.diagnose(image)

    # Assert
    assert isinstance(result, DiagnosticResult)
    assert result.label == "bcc"
    assert result.confidence == 0.85
    assert result.probabilities == prediction.probabilities
    assert result.heatmap.shape == (224, 224)
    assert result.superimposed_image.mode == "RGB"
    assert result.target_class == "bcc"


def test_diagnose_explains_predicted_top1_class_by_default() -> None:
    # Arrange
    explainer = FakeExplainer()
    facade = DermatologyDiagnosticFacade(
        inference_service=FakePredictionProvider(make_prediction(label="mel")),
        explainer=explainer,
    )

    # Act
    result = facade.diagnose(make_image())

    # Assert
    assert explainer.last_target_class == "melanoma"
    assert result.target_class == "mel"


def test_diagnose_maps_each_ham10000_code_to_model_label() -> None:
    # Arrange
    expected_mapping = {
        "mel": "melanoma",
        "nv": "melanocytic_Nevi",
        "bcc": "basal_cell_carcinoma",
        "akiec": "actinic_keratoses",
        "bkl": "benign_keratosis-like_lesions",
        "df": "dermatofibroma",
        "vasc": "vascular_lesions",
    }

    for code, model_label in expected_mapping.items():
        explainer = FakeExplainer()
        facade = DermatologyDiagnosticFacade(
            inference_service=FakePredictionProvider(make_prediction(label=code)),
            explainer=explainer,
        )

        # Act
        result = facade.diagnose(make_image())

        # Assert
        assert explainer.last_target_class == model_label
        assert result.target_class == code


def test_diagnose_forwards_explicit_target_class() -> None:
    # Arrange
    explainer = FakeExplainer()
    facade = DermatologyDiagnosticFacade(
        inference_service=FakePredictionProvider(make_prediction(label="mel")),
        explainer=explainer,
    )

    # Act
    result = facade.diagnose(make_image(), target_class="nv")

    # Assert
    assert explainer.last_target_class == "melanocytic_Nevi"
    assert result.target_class == "nv"


def test_diagnose_propagates_inference_error() -> None:
    # Arrange
    class FailingPredictionProvider(FakePredictionProvider):
        def predict(self, image: Image.Image) -> Prediction:
            del image
            raise InferenceError("Fallo de inferencia simulado")

    facade = DermatologyDiagnosticFacade(
        inference_service=FailingPredictionProvider(),
        explainer=FakeExplainer(),
    )

    # Act / Assert
    with pytest.raises(InferenceError, match="Fallo de inferencia simulado"):
        facade.diagnose(make_image())


def test_diagnose_propagates_gradcam_error() -> None:
    # Arrange
    class FailingExplainer(FakeExplainer):
        def explain(self, image: Image.Image, target_class: str | None = None) -> GradCAMResult:
            del image, target_class
            raise GradCAMError("Fallo de explicación simulado")

    facade = DermatologyDiagnosticFacade(
        inference_service=FakePredictionProvider(),
        explainer=FailingExplainer(),
    )

    # Act / Assert
    with pytest.raises(GradCAMError, match="Fallo de explicación simulado"):
        facade.diagnose(make_image())


def test_constructor_loads_service_and_builds_explainer_from_its_components() -> None:
    # Arrange
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_processor = MagicMock()
    fake_service = InferenceService(model=fake_model, processor=fake_processor)

    with patch(
        "skin_lesion_classifier.facade.load_inference_service",
        return_value=fake_service,
    ) as load_mock:
        # Act
        facade = DermatologyDiagnosticFacade()

        # Assert
        load_mock.assert_called_once_with()
        assert facade._inference_service is fake_service
        explainer = facade._explainer
        assert isinstance(explainer, ViTGradCAM)
        assert explainer._model is fake_model
        assert explainer._processor is fake_processor


def test_constructor_requires_explainer_for_non_inference_service() -> None:
    # Arrange / Act / Assert
    with pytest.raises(TypeError, match="explainer"):
        DermatologyDiagnosticFacade(inference_service=FakePredictionProvider())


def test_diagnostic_result_is_immutable() -> None:
    # Arrange
    result = DiagnosticResult(
        label="mel",
        confidence=0.9,
        probabilities=SEVEN_CLASS_PROBABILITIES,
        heatmap=np.zeros((224, 224), dtype=np.float32),
        superimposed_image=Image.new("RGB", (224, 224)),
        target_class="mel",
    )

    # Act / Assert
    with pytest.raises(FrozenInstanceError):
        result.label = "nv"  # type: ignore[misc]
