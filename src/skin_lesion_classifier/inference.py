"""
=============================================================================
MÓDULO: inference.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Inyección de Dependencias / SRP
RESPONSABILIDAD ÚNICA:
    Transformar imágenes dermatoscópicas en predicciones clínicas estructuradas
    y probabilísticas para las 7 clases diagnósticas del benchmark HAM10000,
    garantizando que la interfaz gráfica permanezca completamente desacoplada
    de PyTorch y Hugging Face.

CLASES DEL BENCHMARK HAM10000:
    1. mel   -> Melanoma (Maligno)
    2. nv    -> Nevus melanocítico benigno (Lunar común)
    3. bcc   -> Carcinoma basocelular (Maligno)
    4. akiec -> Queratosis actínica y carcinoma intraepitelial
    5. bkl   -> Queratosis benigna (tipo seborreica)
    6. df    -> Dermatofibroma
    7. vasc  -> Lesión vascular

OBJETIVO POR CLASE / FUNCIÓN:
    - Prediction (dataclass inmutable):
      Estructura de datos que encapsula el diagnóstico Top-1, su nivel de confianza
      (0.0 a 1.0) y la distribución completa de probabilidades de las 7 patologías.

    - InferenceService:
      Servicio desacoplado que recibe el modelo y procesador inyectados,
      valida la imagen de entrada y ejecuta la inferencia en modo evaluación.
=============================================================================
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Mapeo canónico: de las etiquetas largas de Hugging Face a códigos estándar HAM10000
MODEL_LABEL_TO_CODE = {
    "actinic_keratoses": "akiec",
    "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl",
    "dermatofibroma": "df",
    "melanocytic_Nevi": "nv",
    "melanoma": "mel",
    "vascular_lesions": "vasc",
}

# Conjunto inmutable con los 7 códigos válidos del benchmark HAM10000
HAM10000_LABELS = frozenset(MODEL_LABEL_TO_CODE.values())


class InferenceError(RuntimeError):
    """Excepción lanzada cuando ocurre un error durante el preprocesamiento o inferencia."""

    pass


@dataclass(frozen=True)
class Prediction:
    """
    Objeto de transferencia de datos (DTO) inmutable con el resultado del diagnóstico.

    Attributes:
        label (str): Código corto de la patología con mayor probabilidad (ej. 'mel', 'nv').
        confidence (float): Nivel de certidumbre del diagnóstico Top-1 en rango [0.0, 1.0].
        probabilities (dict[str, float]): Distribución de probabilidad para las 7 clases.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]


@runtime_checkable
class ImageProcessorProtocol(Protocol):
    """Protocolo que define el contrato del extractor de características de imagen."""

    def __call__(self, images: Any, return_tensors: str = "pt") -> Any: ...


@runtime_checkable
class ClassificationModelProtocol(Protocol):
    """Protocolo que define el contrato del modelo de clasificación por visión."""

    def eval(self) -> Any: ...
    def __call__(self, **kwargs: Any) -> Any: ...

    config: Any


class InferenceService:
    """
    Servicio desacoplado de inferencia para clasificación dermatológica.

    Utiliza inyección de dependencias para operar independientemente del origen del modelo
    (pesos reales de Hugging Face o mocks sintéticos para pruebas unitarias rápidas).
    """

    def __init__(
        self,
        model: ClassificationModelProtocol,
        processor: ImageProcessorProtocol,
    ) -> None:
        """
        Inicializa el servicio configurando el modelo en modo de evaluación.

        Args:
            model: Modelo Vision Transformer que cumple con ClassificationModelProtocol.
            processor: Procesador de imágenes que cumple con ImageProcessorProtocol.
        """
        self._model = model.eval()
        self._processor = processor

    def predict(self, image: Image.Image) -> Prediction:
        """
        Ejecuta la inferencia diagnóstica sobre una imagen dermatoscópica.

        Flujo secuencial:
            1. Validación defensiva de la imagen de entrada.
            2. Conversión a espacio de color RGB y extracción de tensores (1, 3, 224, 224).
            3. Paso hacia adelante (forward pass) en modo 'inference_mode' (sin gradientes).
            4. Normalización Softmax para obtener la distribución de probabilidad (0 a 1).
            5. Mapeo a las 7 etiquetas clínicas y selección de la hipótesis diagnóstica Top-1.

        Args:
            image (Image.Image): Imagen dermatoscópica en formato PIL.

        Returns:
            Prediction: Diagnóstico estructurado con clase, confianza y probabilidades.

        Raises:
            InferenceError: Si la imagen es inválida, está corrupta o falla el cálculo tensorial.
        """
        # 1. Validación defensiva del tipo de dato de entrada
        if not isinstance(image, Image.Image):
            raise InferenceError(
                f"Entrada inválida: Se esperaba una imagen PIL.Image, recibido {type(image)}."
            )

        if image.size[0] == 0 or image.size[1] == 0:
            raise InferenceError("Entrada inválida: La imagen tiene dimensiones nulas (0x0).")

        # 2. Preprocesamiento de la imagen a tensores PyTorch
        try:
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        except Exception as err:
            logger.error(f"Fallo durante el preprocesamiento de imagen: {err}")
            raise InferenceError(f"Error al transformar la imagen en tensor: {err}") from err

        # 3. Inferencia de alta eficiencia sin cálculo de gradientes
        try:
            with torch.inference_mode():
                outputs = self._model(**inputs)
                logits = outputs.logits
        except Exception as err:
            logger.error(f"Fallo durante el forward pass del modelo: {err}")
            raise InferenceError(f"Error en la ejecución del modelo ViT: {err}") from err

        # 4. Postprocesamiento seguro (Softmax, mapeo a HAM10000 y Top-1)
        try:
            scores = torch.softmax(logits, dim=-1)[0].tolist()

            probabilities = {
                MODEL_LABEL_TO_CODE[self._model.config.id2label[index]]: score
                for index, score in enumerate(scores)
            }

            top_label = max(probabilities, key=lambda k: probabilities[k])
        except Exception as err:
            logger.error(f"Fallo durante el postprocesamiento de predicciones: {err}")
            raise InferenceError(
                f"Error al postprocesar los logits del modelo en probabilidades HAM10000: {err}"
            ) from err

        return Prediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )
