from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from skin_lesion_classifier.gradcam import GradCAMError, GradCAMResult, ViTGradCAM


class SyntheticModel(torch.nn.Module):
    def __init__(self, labels: list[str], tokens: int = 197) -> None:
        super().__init__()
        self._tokens = tokens
        self.vit: Any = torch.nn.Module()
        self.vit.encoder = torch.nn.Module()
        block = torch.nn.Module()
        block.layernorm_before = torch.nn.Identity()
        self.vit.encoder.layer = torch.nn.ModuleList([block])
        self.config = MagicMock(id2label={i: label for i, label in enumerate(labels)})
        self.classifier = torch.nn.Linear(tokens, len(labels), bias=False)
        with torch.no_grad():
            self.classifier.weight.fill_(0.01)

    def forward(self, pixel_values: torch.Tensor) -> Any:
        batch, _, height, width = pixel_values.shape
        del height, width
        tokens = self._tokens
        sequence = pixel_values.mean(dim=1).reshape(batch, -1)
        sequence = torch.nn.functional.adaptive_avg_pool1d(sequence.unsqueeze(1), tokens).squeeze(1)
        sequence = sequence.unsqueeze(-1).expand(-1, -1, 4)
        encoded = self.vit.encoder.layer[-1].layernorm_before(sequence)
        logits = self.classifier(encoded[:, :, 0])
        return MagicMock(logits=logits)


class SyntheticProcessor:
    def __call__(self, images: Image.Image, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del return_tensors
        width, height = images.size
        return {"pixel_values": torch.ones((1, 3, height, width), requires_grad=True)}


def make_explainer(size: tuple[int, int] = (224, 224)) -> tuple[ViTGradCAM, Image.Image]:
    tokens = (size[0] // 16) * (size[1] // 16) + 1
    model = SyntheticModel(["mel", "nv"], tokens=tokens)
    return ViTGradCAM(model, SyntheticProcessor()), Image.new("RGB", size, (100, 80, 60))


def test_explain_returns_normalized_heatmap_and_rgb_overlay() -> None:
    # Arrange
    explainer, image = make_explainer()

    # Act
    result = explainer.explain(image)

    # Assert
    assert isinstance(result, GradCAMResult)
    assert result.heatmap.shape == (224, 224)
    assert 0.0 <= float(result.heatmap.min()) <= float(result.heatmap.max()) <= 1.0
    assert result.superimposed_image.mode == "RGB"
    assert result.superimposed_image.size == (224, 224)


def test_explain_supports_explicit_class_and_top1_class() -> None:
    # Arrange
    explainer, image = make_explainer()

    # Act
    explicit = explainer.explain(image, target_class="nv")
    top1 = explainer.explain(image)

    # Assert
    assert explicit.target_class == "nv"
    assert top1.target_class in {"mel", "nv"}


def test_explain_resolves_dynamic_patch_grid() -> None:
    # Arrange
    explainer, image = make_explainer((256, 256))

    # Act
    result = explainer.explain(image)

    # Assert
    assert result.heatmap.shape == (256, 256)
    assert result.superimposed_image.size == (256, 256)


def test_explain_removes_hook_after_success() -> None:
    # Arrange
    explainer, image = make_explainer()
    layer = explainer._model.vit.encoder.layer[-1].layernorm_before
    assert len(layer._forward_hooks) == 0

    # Act
    explainer.explain(image)

    # Assert
    assert len(layer._forward_hooks) == 0


def test_explain_removes_hook_after_failure() -> None:
    # Arrange
    model = SyntheticModel(["mel", "nv"])
    processor = MagicMock(side_effect=RuntimeError("processor failure"))
    explainer = ViTGradCAM(model, processor)
    layer = explainer._model.vit.encoder.layer[-1].layernorm_before

    # Act / Assert
    with pytest.raises(GradCAMError, match="generar el mapa"):
        explainer.explain(Image.new("RGB", (224, 224)))
    assert len(layer._forward_hooks) == 0


def test_explain_rejects_unknown_class_and_invalid_tokens() -> None:
    # Arrange
    explainer, image = make_explainer()

    # Act / Assert
    with pytest.raises(GradCAMError, match="Clase objetivo desconocida"):
        explainer.explain(image, target_class="unknown")

    invalid_model = SyntheticModel(["mel", "nv"], tokens=200)
    invalid_explainer = ViTGradCAM(invalid_model, SyntheticProcessor())
    with pytest.raises(GradCAMError, match="grilla espacial"):
        invalid_explainer.explain(image)
