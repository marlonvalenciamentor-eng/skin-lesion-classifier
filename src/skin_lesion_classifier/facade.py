"""
=============================================================================
MÓDULO: facade.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Patrón Fachada / SRP
RESPONSABILIDAD ÚNICA:
    Orquestar el pipeline completo de diagnóstico dermatológico —carga del
    servicio de inferencia, predicción clínica y explicabilidad Grad-CAM—
    exponiendo un único punto de entrada que la capa de presentación puede
    consumir sin importar jamás PyTorch ni Hugging Face.

FLUJO:
    load_inference_service() -> InferenceService.predict() -> ViTGradCAM.explain()
    -> DiagnosticResult (DTO inmutable sin tipos de bajo nivel)
=============================================================================
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from skin_lesion_classifier.gradcam import GradCAMResult, ViTGradCAM
from skin_lesion_classifier.inference import InferenceService, Prediction
from skin_lesion_classifier.model_loader import load_inference_service


class PredictionProvider(Protocol):
    """Contrato de un componente capaz de producir una predicción clínica."""

    def predict(self, image: Image.Image) -> Prediction: ...


class ExplainerProvider(Protocol):
    """Contrato de un componente capaz de generar una explicación Grad-CAM."""

    def explain(self, image: Image.Image, target_class: str | None = None) -> GradCAMResult: ...


@dataclass(frozen=True)
class DiagnosticResult:
    """
    Objeto de transferencia de datos (DTO) inmutable con el diagnóstico completo.

    Combina la predicción clínica y la explicación visual sin exponer tipos
    de bajo nivel (tensores o componentes de Hugging Face), de modo que la
    capa de presentación permanece totalmente desacoplada del stack de Deep Learning.

    Attributes:
        label (str): Código corto HAM10000 de la patología con mayor probabilidad.
        confidence (float): Confianza del diagnóstico Top-1 en rango [0.0, 1.0].
        probabilities (dict[str, float]): Distribución de probabilidad de las 7 clases.
        heatmap (np.ndarray): Mapa de calor Grad-CAM normalizado en [0.0, 1.0].
        superimposed_image (Image.Image): Superposición RGB del mapa sobre la lesión.
        target_class (str): Clase explicada por Grad-CAM.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]
    heatmap: np.ndarray
    superimposed_image: Image.Image
    target_class: str


class DermatologyDiagnosticFacade:
    """
    Fachada orquestadora del sistema de diagnóstico dermatológico.

    Centraliza la carga del modelo, la inferencia clínica y la explicabilidad
    Grad-CAM en un único método, aplicando inyección de dependencias para que
    las pruebas unitarias puedan sustituir los componentes reales por dobles
    sintéticos sin conectividad ni pesos.
    """

    def __init__(
        self,
        inference_service: PredictionProvider | None = None,
        explainer: ExplainerProvider | None = None,
    ) -> None:
        """
        Inicializa la fachada cargando los componentes reales o aceptando dobles inyectados.

        Si no se provee servicio de inferencia, se carga con la política offline-first
        (`load_inference_service`). Si no se provee explainer, se construye un
        `ViTGradCAM` reutilizando el modelo y procesador del servicio cargado.

        Args:
            inference_service: Proveedor de predicciones (por defecto, el servicio real).
            explainer: Proveedor de explicaciones Grad-CAM (por defecto, ViTGradCAM real).

        Raises:
            TypeError: Si se inyecta un servicio que no expone `model`/`processor`
                       y no se provee un explainer explícito.
        """
        if inference_service is None:
            inference_service = load_inference_service()
        self._inference_service = inference_service

        if explainer is None:
            if not isinstance(inference_service, InferenceService):
                raise TypeError(
                    "Se requiere un 'explainer' explícito cuando el servicio inyectado "
                    "no es una instancia de 'InferenceService'."
                )
            explainer = ViTGradCAM(
                model=inference_service.model,
                processor=inference_service.processor,
            )
        self._explainer = explainer

    def diagnose(
        self,
        image: Image.Image,
        target_class: str | None = None,
    ) -> DiagnosticResult:
        """
        Ejecuta el diagnóstico completo sobre una imagen dermatoscópica.

        Orquesta la predicción clínica y la explicación Grad-CAM. Por defecto,
        Grad-CAM explica la clase Top-1 predicha para garantizar consistencia
        clínica entre el diagnóstico y la explicación visual.

        Args:
            image: Imagen dermatoscópica en formato PIL.
            target_class: Clase HAM10000 a explicar (por defecto, la clase predicha).

        Returns:
            DiagnosticResult: Diagnóstico estructurado con predicción y explicación.

        Raises:
            InferenceError: Si la imagen es inválida o falla la inferencia.
            GradCAMError: Si falla la generación del mapa de explicabilidad.
        """
        prediction = self._inference_service.predict(image)
        explanation = self._explainer.explain(
            image,
            target_class=target_class or prediction.label,
        )
        return DiagnosticResult(
            label=prediction.label,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            heatmap=explanation.heatmap,
            superimposed_image=explanation.superimposed_image,
            target_class=explanation.target_class,
        )
