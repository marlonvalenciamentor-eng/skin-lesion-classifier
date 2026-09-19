"""
=============================================================================
MÓDULO: model_loader.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Resiliencia y Manejo Defensivo de Excepciones
RESPONSABILIDAD ÚNICA:
    Cargar, validar la integridad y ensamblar el Vision Transformer (ViT)
    y su procesador de imágenes, gestionando fallos de red, corrupción de archivos
    o discrepancias en la arquitectura del modelo.

COMPONENTES:
    1. MODEL_ID: 'Anwarkh1/Skin_Cancer-Image_Classification' (Modelo ajustado a HAM10000).
    2. BASE_PROCESSOR_ID: 'google/vit-base-patch16-224-in21k' (Extractor base de características).

OBJETIVO POR FUNCIÓN:
    - validate_model_integrity(model: ViTForImageClassification) -> None
      Certifica que el modelo cargado contiene exactamente las 7 neuronas de salida
      correspondientes a las clases clínicas del benchmark HAM10000.

    - load_inference_service(model_id: str) -> InferenceService
      Carga los pesos del modelo, inicializa el procesador de tensores y devuelve
      una instancia de 'InferenceService' mediante inyección de dependencias.
=============================================================================
"""

import logging

from transformers import ViTForImageClassification, ViTImageProcessor

from skin_lesion_classifier.inference import InferenceService

logger = logging.getLogger(__name__)

# Identificador oficial del modelo en Hugging Face Hub (fine-tuned para 7 clases dermatológicas)
MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"

# El repositorio del modelo no incluye configuración propia del extractor de imágenes;
# por tanto, se utiliza el procesador del checkpoint base oficial de Google (224x224 píxeles).
BASE_PROCESSOR_ID = "google/vit-base-patch16-224-in21k"

# Número canónico de clases patológicas exigidas por el benchmark HAM10000
EXPECTED_LABELS_COUNT = 7


class ModelLoadingError(RuntimeError):
    """Excepción de dominio lanzada cuando el modelo no puede cargarse o no supera la validación."""

    pass


def validate_model_integrity(model: ViTForImageClassification) -> None:
    """
    Certifica que el modelo cargado cumple estrictamente con la especificación clínica.

    Args:
        model (ViTForImageClassification): Instancia del modelo a validar.

    Raises:
        ModelLoadingError: Si el número de clases de salida difiere de 7.
    """
    num_labels = getattr(model.config, "num_labels", None)
    if num_labels != EXPECTED_LABELS_COUNT:
        raise ModelLoadingError(
            f"Fallo de integridad arquitectónica: Se esperaban {EXPECTED_LABELS_COUNT} "
            f"clases patológicas, pero el modelo tiene {num_labels}."
        )


def load_inference_service(model_id: str = MODEL_ID) -> InferenceService:
    """
    Carga de forma segura el modelo ViT y su procesador con manejo defensivo de excepciones.

    Aplica el principio de Inyección de Dependencias: el modelo y el procesador
    se configuran aquí y se inyectan en 'InferenceService', permitiendo que la interfaz
    gráfica (Streamlit) o las pruebas unitarias consuman el servicio sin importar
    directamente dependencias de bajo nivel como PyTorch o Transformers.

    Args:
        model_id (str, opcional): Ruta local o ID de Hugging Face del modelo.
                                  Por defecto utiliza MODEL_ID.

    Returns:
        InferenceService: Instancia configurada y lista para ejecutar diagnósticos.

    Raises:
        ModelLoadingError: Si no se encuentra el modelo, falta conexión o los pesos están corruptos.
    """
    # 1. Intentar cargar los pesos de la red y su configuración
    try:
        model = ViTForImageClassification.from_pretrained(model_id)
    except (OSError, ValueError) as err:
        logger.error(f"Error al cargar el modelo desde '{model_id}': {err}")
        raise ModelLoadingError(
            f"No fue posible cargar el modelo desde '{model_id}'. "
            "Verifique la ruta, la integridad del archivo 'model.safetensors' o su conexión a red."
        ) from err
    except Exception as err:
        logger.error(f"Error inesperado al instanciar el modelo: {err}")
        raise ModelLoadingError(f"Error crítico al inicializar el modelo: {err}") from err

    # 2. Validar integridad de la arquitectura clínica
    validate_model_integrity(model)

    # 3. Intentar cargar el procesador de imágenes (normalización y redimensionado)
    try:
        processor = ViTImageProcessor.from_pretrained(BASE_PROCESSOR_ID)
    except Exception as err:
        logger.error(f"Error al cargar el procesador '{BASE_PROCESSOR_ID}': {err}")
        raise ModelLoadingError(
            f"No se pudo inicializar el procesador de imágenes '{BASE_PROCESSOR_ID}'. "
            "Verifique la conectividad con Hugging Face o la caché local."
        ) from err

    # 4. Retornar el servicio con inyección de dependencias
    return InferenceService(model=model, processor=processor)
