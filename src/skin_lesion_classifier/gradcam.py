"""
=============================================================================
MÓDULO: gradcam.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Responsabilidad Única (SRP)
RESPONSABILIDAD ÚNICA:
    Generar explicaciones visuales Grad-CAM para un clasificador Vision
    Transformer (ViT) mediante un modelo y un procesador inyectados, sin
    acoplar la explicabilidad a la interfaz gráfica ni a la carga del modelo.

FLUJO GRAD-CAM:
    Registrar un hook en la capa objetivo del encoder ViT, capturar activaciones
    y gradientes para la clase seleccionada, excluir el token [CLS], ponderar
    los parches espaciales, formar el mapa de calor, interpolarlo a la
    resolución de entrada y producir una superposición RGB sobre la imagen.
=============================================================================
"""

from dataclasses import dataclass
from math import isqrt
from typing import Any, Protocol

import matplotlib
import numpy as np
import torch
from PIL import Image


class GradCAMError(RuntimeError):
    """Error de dominio para operaciones inválidas de Grad-CAM."""


@dataclass(frozen=True)
class GradCAMResult:
    """Immutable output of a Grad-CAM explanation."""

    heatmap: np.ndarray
    superimposed_image: Image.Image
    target_class: str


class _ModelProtocol(Protocol):
    config: Any

    def eval(self) -> Any: ...

    def __call__(self, **kwargs: Any) -> Any: ...


class ViTGradCAM:
    """Generate a token-based Grad-CAM map without owning model dependencies."""

    def __init__(self, model: _ModelProtocol, processor: Any) -> None:
        self._model = model.eval()
        self._processor = processor

    def explain(
        self,
        image: Image.Image,
        target_class: str | None = None,
    ) -> GradCAMResult:
        """Explain an explicit class or the model's top-1 prediction."""
        if not isinstance(image, Image.Image) or image.width == 0 or image.height == 0:
            raise GradCAMError("Entrada inválida: se esperaba una imagen PIL no vacía.")

        handles: list[Any] = []
        activation: list[torch.Tensor] = []
        try:
            target_layer = self._target_layer()

            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                if not isinstance(tensor, torch.Tensor):
                    raise GradCAMError("La capa objetivo no produjo un tensor válido.")
                tensor.retain_grad()
                activation.append(tensor)

            handles.append(target_layer.register_forward_hook(capture))
            processed = self._processor(images=image.convert("RGB"), return_tensors="pt")
            pixel_values = self._pixel_values(processed)
            input_height, input_width = self._input_size(pixel_values)

            with torch.enable_grad():
                outputs = self._model(**processed)
                logits = outputs.logits
                class_index, resolved_class = self._resolve_target(logits, target_class)
                score = logits[0, class_index]
                score.backward()

            if len(activation) != 1:
                raise GradCAMError("No se capturaron activaciones de la capa objetivo.")
            return self._build_result(
                activation[0], input_height, input_width, image, resolved_class
            )
        except GradCAMError:
            raise
        except Exception as error:
            raise GradCAMError(f"Error al generar el mapa Grad-CAM: {error}") from error
        finally:
            for handle in handles:
                handle.remove()

    def _target_layer(self) -> Any:
        try:
            return self._model.vit.encoder.layer[-1].layernorm_before
        except (AttributeError, IndexError, TypeError) as error:
            raise GradCAMError("No se encontró una capa ViT válida para Grad-CAM.") from error

    @staticmethod
    def _pixel_values(processed: Any) -> torch.Tensor:
        try:
            values = processed["pixel_values"]
        except (KeyError, TypeError) as error:
            raise GradCAMError("El procesador no devolvió pixel_values.") from error
        if not isinstance(values, torch.Tensor) or values.ndim != 4 or values.shape[0] != 1:
            raise GradCAMError("pixel_values debe tener forma [1, canales, alto, ancho].")
        return values

    @staticmethod
    def _input_size(pixel_values: torch.Tensor) -> tuple[int, int]:
        height, width = pixel_values.shape[-2:]
        if height <= 0 or width <= 0:
            raise GradCAMError("La resolución de entrada debe ser positiva.")
        return int(height), int(width)

    def _resolve_target(self, logits: torch.Tensor, requested: str | None) -> tuple[int, str]:
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] == 0:
            raise GradCAMError("Los logits deben tener forma [1, clases].")
        labels = self._labels(int(logits.shape[1]))
        if requested is None:
            index = int(torch.argmax(logits[0]).item())
            return index, labels[index]
        if requested in labels:
            return labels.index(requested), requested
        raise GradCAMError(f"Clase objetivo desconocida: {requested}.")

    def _labels(self, count: int) -> list[str]:
        try:
            raw_labels = self._model.config.id2label
            labels = [str(raw_labels[index]) for index in range(count)]
        except (AttributeError, KeyError, TypeError, IndexError) as error:
            raise GradCAMError("El modelo no expone un mapeo id2label válido.") from error
        if len(set(labels)) != count:
            raise GradCAMError("El mapeo id2label contiene clases duplicadas.")
        return labels

    def _build_result(
        self,
        activation: torch.Tensor,
        input_height: int,
        input_width: int,
        image: Image.Image,
        target_class: str,
    ) -> GradCAMResult:
        if activation.ndim != 3 or activation.shape[0] != 1:
            raise GradCAMError("Las activaciones deben tener forma [1, tokens, canales].")
        if activation.grad is None:
            raise GradCAMError("No se obtuvieron gradientes de la activación objetivo.")

        tokens = int(activation.shape[1]) - 1
        grid = self._grid(tokens, input_height, input_width)
        spatial_activation = activation[:, 1:, :]
        spatial_gradient = activation.grad[:, 1:, :]
        weights = spatial_gradient.mean(dim=1, keepdim=True)
        values = torch.relu((spatial_activation * weights).sum(dim=-1))
        values = values.reshape(1, 1, grid[0], grid[1])
        values = torch.nn.functional.interpolate(
            values, size=(input_height, input_width), mode="bilinear", align_corners=False
        )[0, 0]
        minimum, maximum = values.min(), values.max()
        if float((maximum - minimum).detach()) > 0:
            values = (values - minimum) / (maximum - minimum)
        else:
            values = torch.zeros_like(values)
        heatmap = values.detach().cpu().numpy().astype(np.float32, copy=False)
        overlay = self._overlay(image, heatmap)
        return GradCAMResult(heatmap=heatmap, superimposed_image=overlay, target_class=target_class)

    @staticmethod
    def _grid(tokens: int, height: int, width: int) -> tuple[int, int]:
        if tokens <= 0:
            raise GradCAMError("La salida del encoder no contiene tokens espaciales.")
        root = isqrt(tokens)
        if root * root == tokens:
            return root, root
        if tokens == 196:
            return 14, 14
        for rows in range(isqrt(tokens), 0, -1):
            if tokens % rows == 0:
                columns = tokens // rows
                if abs((columns / rows) - (width / height)) < 0.5:
                    return rows, columns
        raise GradCAMError("No se pudo inferir una grilla espacial válida de los tokens.")

    @staticmethod
    def _overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
        rgb = np.asarray(image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0])))
        jet = matplotlib.colormaps["jet"](heatmap)[..., :3]
        blended = (0.55 * rgb.astype(np.float32) / 255.0 + 0.45 * jet) * 255.0
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
