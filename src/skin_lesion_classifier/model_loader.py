"""
=============================================================================
MÓDULO: model_loader.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Resiliencia y Manejo Defensivo de Excepciones
RESPONSABILIDAD ÚNICA:
    Cargar, validar la integridad arquitectónica y clínica, y ensamblar el
    Vision Transformer (ViT) y su procesador de imágenes, gestionando fallos
    de red, corrupción de archivos o discrepancias en la taxonomía HAM10000.

COMPONENTES:
    1. MODEL_ID: 'Anwarkh1/Skin_Cancer-Image_Classification' (Modelo ajustado a HAM10000).
    2. BASE_PROCESSOR_ID: 'google/vit-base-patch16-224-in21k' (Extractor base de características).
    3. DEFAULT_LOCAL_MODEL_DIR: 'models/vit-skin-cancer' (Copia local preferida, offline-first).

OBJETIVO POR FUNCIÓN:
    - validate_model_integrity(model: ViTForImageClassification) -> None
      Certifica que el modelo cargado contiene exactamente las 7 neuronas de salida
      y que sus etiquetas de clasificación corresponden rigurosamente con las
      7 patologías dermatológicas del benchmark HAM10000.

    - resolve_model_source(local_dir, remote_id, required_files) -> str
      Decide de dónde cargar un artefacto: el directorio local si está completo,
      o el identificador remoto de Hugging Face como respaldo.

    - load_inference_service(model_id, processor_id, local_dir) -> InferenceService
      Carga los pesos y el procesador aplicando política offline-first (local primero,
      descarga de Hugging Face solo si el directorio local no existe o está incompleto)
      y devuelve una instancia de 'InferenceService' mediante inyección de dependencias.
=============================================================================
"""

import logging
from pathlib import Path

from transformers import ViTForImageClassification, ViTImageProcessor

from skin_lesion_classifier.inference import MODEL_LABEL_TO_CODE, InferenceService

logger = logging.getLogger(__name__)

# Identificador oficial del modelo en Hugging Face Hub (fine-tuned para 7 clases dermatológicas)
MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"

# El repositorio del modelo no incluye configuración propia del extractor de imágenes;
# por tanto, se utiliza el procesador del checkpoint base oficial de Google (224x224 píxeles).
BASE_PROCESSOR_ID = "google/vit-base-patch16-224-in21k"

# Directorio local preferido para el modelo (relativo a la raíz del proyecto).
# Si contiene los archivos requeridos, se carga desde disco y no se toca la red.
DEFAULT_LOCAL_MODEL_DIR = Path("models/vit-skin-cancer")

# Archivos mínimos que certifican que el directorio local está completo.
MODEL_REQUIRED_FILES = ("config.json", "model.safetensors")
PROCESSOR_REQUIRED_FILES = ("preprocessor_config.json",)

# Conjunto canónico de clases patológicas exigidas por el benchmark HAM10000
EXPECTED_LABELS_COUNT = 7
EXPECTED_MODEL_LABELS = frozenset(MODEL_LABEL_TO_CODE.keys())


def resolve_model_source(
    local_dir: Path | None,
    remote_id: str,
    required_files: tuple[str, ...],
) -> str:
    """
    Resuelve de dónde cargar un artefacto: primero el disco local, luego Hugging Face.

    Aplica una política *offline-first*: si el directorio local existe y contiene
    todos los archivos requeridos, se usa esa ruta. En caso contrario, se devuelve
    el identificador remoto para que Hugging Face lo descargue o lo tome de su caché.

    Args:
        local_dir (Path | None): Directorio local candidato. Si es None se omite.
        remote_id (str): Identificador de Hugging Face usado como respaldo.
        required_files (tuple[str, ...]): Archivos que deben existir para considerar
                                          el directorio local como válido.

    Returns:
        str: Ruta local (como texto) o el identificador remoto.
    """
    if local_dir is not None and local_dir.is_dir():
        missing = [name for name in required_files if not (local_dir / name).is_file()]
        if not missing:
            logger.info(f"Usando artefacto local desde '{local_dir}'.")
            return str(local_dir)
        logger.warning(
            f"Directorio local '{local_dir}' incompleto (faltan {missing}); "
            f"se usará el origen remoto '{remote_id}'."
        )
    return remote_id


class ModelLoadingError(RuntimeError):
    """Excepción de dominio lanzada cuando el modelo no puede cargarse o no supera la validación."""

    pass


def validate_model_integrity(model: ViTForImageClassification) -> None:
    """
    Certifica que el modelo cargado cumple estrictamente con la especificación clínica.

    Valida tanto la cantidad de neuronas de salida como la correspondencia exacta
    de las etiquetas diagnósticas con la taxonomía oficial de HAM10000.

    Args:
        model (ViTForImageClassification): Instancia del modelo a validar.

    Raises:
        ModelLoadingError: Si el número de clases difiere de 7 o si las etiquetas
                           diagnósticas no coinciden con las patologías esperadas.
    """
    num_labels = getattr(model.config, "num_labels", None)
    if num_labels != EXPECTED_LABELS_COUNT:
        raise ModelLoadingError(
            f"Fallo de integridad arquitectónica: Se esperaban {EXPECTED_LABELS_COUNT} "
            f"clases patológicas, pero el modelo tiene {num_labels}."
        )

    id2label = getattr(model.config, "id2label", {})
    # Validar que los identificadores numéricos cubran exactamente 0..6
    expected_ids = set(range(EXPECTED_LABELS_COUNT))
    actual_ids = {int(k) for k in id2label.keys()} if id2label else set()
    if actual_ids != expected_ids:
        raise ModelLoadingError(
            f"Fallo de integridad arquitectónica: Los índices de id2label {actual_ids} "
            f"no cubren la secuencia esperada 0..{EXPECTED_LABELS_COUNT - 1}."
        )

    model_labels = set(id2label.values())
    if model_labels != EXPECTED_MODEL_LABELS:
        missing = EXPECTED_MODEL_LABELS - model_labels
        unexpected = model_labels - EXPECTED_MODEL_LABELS
        raise ModelLoadingError(
            f"Fallo de integridad clínica: Las etiquetas del modelo no coinciden con HAM10000. "
            f"Faltantes: {missing or 'Ninguna'}, Inesperadas: {unexpected or 'Ninguna'}."
        )

    # Validar presencia obligatoria y correspondencia biyectiva estricta de label2id
    label2id = getattr(model.config, "label2id", None)
    if not isinstance(label2id, dict):
        raise ModelLoadingError(
            "Fallo de integridad clínica: La configuración del modelo "
            "debe incluir el diccionario 'label2id'."
        )

    if set(label2id.keys()) != EXPECTED_MODEL_LABELS:
        missing_keys = EXPECTED_MODEL_LABELS - set(label2id.keys())
        extra_keys = set(label2id.keys()) - EXPECTED_MODEL_LABELS
        raise ModelLoadingError(
            f"Fallo de integridad clínica: Las claves de 'label2id' no coinciden con HAM10000. "
            f"Faltantes: {missing_keys or 'Ninguna'}, Inesperadas: {extra_keys or 'Ninguna'}."
        )

    for idx_raw, label in id2label.items():
        idx = int(idx_raw)
        if label not in label2id or int(label2id[label]) != idx:
            raise ModelLoadingError(
                f"Fallo de integridad clínica: Inconsistencia biyectiva entre "
                f"id2label[{idx}]='{label}' y label2id."
            )


def load_inference_service(
    model_id: str = MODEL_ID,
    processor_id: str = BASE_PROCESSOR_ID,
    local_dir: Path | None = DEFAULT_LOCAL_MODEL_DIR,
) -> InferenceService:
    """
    Carga de forma segura el modelo ViT y su procesador con manejo defensivo de excepciones.

    Política *offline-first*: si el directorio local contiene los archivos requeridos,
    el modelo y el procesador se cargan desde disco sin tocar la red. Si el directorio
    no existe o está incompleto, se usa el identificador remoto y Hugging Face realiza
    la descarga (o la toma de su caché).

    Aplica el principio de Inyección de Dependencias: el modelo y el procesador
    se configuran aquí y se inyectan en 'InferenceService', permitiendo que la interfaz
    gráfica (Streamlit) o las pruebas unitarias consuman el servicio sin importar
    directamente dependencias de bajo nivel como PyTorch o Transformers.

    Args:
        model_id (str, opcional): ID de Hugging Face del modelo, usado como respaldo.
        processor_id (str, opcional): ID de Hugging Face del procesador, usado como respaldo.
        local_dir (Path | None, opcional): Directorio local preferido. Si es None se
                                           omite y se carga siempre desde el origen remoto.

    Returns:
        InferenceService: Instancia configurada y lista para ejecutar diagnósticos.

    Raises:
        ModelLoadingError: Si no se encuentra el modelo, falta conexión o los pesos están corruptos.
    """
    # 0. Resolver el origen de cada artefacto (local primero, remoto como respaldo)
    model_source = resolve_model_source(local_dir, model_id, MODEL_REQUIRED_FILES)
    processor_source = resolve_model_source(local_dir, processor_id, PROCESSOR_REQUIRED_FILES)

    # 1. Intentar cargar los pesos de la red y su configuración
    try:
        model = ViTForImageClassification.from_pretrained(model_source)
    except (OSError, ValueError) as err:
        logger.error(f"Error al cargar el modelo desde '{model_source}': {err}")
        raise ModelLoadingError(
            f"No fue posible cargar el modelo desde '{model_source}'. "
            "Verifique la ruta, la integridad del archivo 'model.safetensors' o su conexión a red."
        ) from err
    except Exception as err:
        logger.error(f"Error inesperado al instanciar el modelo: {err}")
        raise ModelLoadingError(f"Error crítico al inicializar el modelo: {err}") from err

    # 2. Validar integridad de la arquitectura y taxonomía clínica
    validate_model_integrity(model)

    # 3. Intentar cargar el procesador de imágenes (normalización y redimensionado)
    try:
        processor = ViTImageProcessor.from_pretrained(processor_source)
    except Exception as err:
        logger.error(f"Error al cargar el procesador '{processor_source}': {err}")
        raise ModelLoadingError(
            f"No se pudo inicializar el procesador de imágenes '{processor_source}'. "
            "Verifique la conectividad con Hugging Face o la caché local."
        ) from err

    # 4. Retornar el servicio con inyección de dependencias
    return InferenceService(model=model, processor=processor)
