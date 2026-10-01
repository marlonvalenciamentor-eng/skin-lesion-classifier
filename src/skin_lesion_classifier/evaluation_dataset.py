"""
=============================================================================
MÓDULO: evaluation_dataset.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Inyección de Dependencias / SRP
RESPONSABILIDAD ÚNICA:
    Acceder al dataset HAM10000 (`marmal88/skin_cancer`, revisión fijada) para
    la evaluación: descargar el split de prueba al caché de Hugging Face,
    recorrerlo por lotes (sin cargar los ~354 MB en memoria) devolviendo
    imágenes RGB con su código corto, y leer solo la columna `image_id` de otros
    splits para calcular el solapamiento. Descargador y sistema de archivos son
    inyectables para que los tests nunca toquen la red.
=============================================================================
"""

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image, UnidentifiedImageError

from skin_lesion_classifier.inference import MODEL_LABEL_TO_CODE

DATASET_ID: Final[str] = "marmal88/skin_cancer"
DATASET_REVISION: Final[str] = "bdd59e10860746202dfdc452e0ac3a9eaa25268d"
TEST_PARQUET_FILENAME: Final[str] = "data/test-00000-of-00001-61e7cf54bf274ae2.parquet"
DEFAULT_BATCH_SIZE: Final[int] = 32


class DatasetLoadingError(RuntimeError):
    """Error de dominio cuando el dataset de evaluación no puede obtenerse o leerse."""


@dataclass(frozen=True)
class LabeledImage:
    """Muestra de evaluación: imagen RGB con su identificador y su código corto HAM10000."""

    image_id: str
    lesion_id: str
    code: str
    image: Image.Image


class Downloader(Protocol):
    """Contrato compatible con `huggingface_hub.hf_hub_download`."""

    def __call__(self, *, repo_id: str, filename: str, repo_type: str, revision: str) -> str: ...


class SplitFileSystem(Protocol):
    """Subconjunto del contrato de `HfFileSystem` usado para leer IDs de otros splits."""

    def ls(self, path: str, detail: bool = ...) -> list[Any]: ...

    def open(self, path: str, mode: str = ...) -> Any: ...


def download_test_split(downloader: Downloader = hf_hub_download) -> Path:
    """Descarga (o toma del caché de HF) el parquet del split de prueba en la revisión fijada."""
    try:
        local_path = downloader(
            repo_id=DATASET_ID,
            filename=TEST_PARQUET_FILENAME,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    except Exception as err:
        raise DatasetLoadingError(
            f"No fue posible descargar el split de prueba de '{DATASET_ID}': {err}"
        ) from err
    return Path(local_path)


def _decode_image(raw: bytes, image_id: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as err:
        raise DatasetLoadingError(f"No se pudo decodificar la imagen '{image_id}': {err}") from err


def iter_labeled_images(
    parquet_path: Path,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[LabeledImage]:
    """Recorre el parquet por lotes y emite muestras RGB con código corto (máximo `limit`)."""
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except (OSError, ValueError) as err:
        raise DatasetLoadingError(
            f"No se pudo abrir el archivo parquet '{parquet_path}': {err}"
        ) from err

    emitted = 0
    for batch in parquet_file.iter_batches(
        batch_size=batch_size, columns=["image", "image_id", "lesion_id", "dx"]
    ):
        for row in batch.to_pylist():
            if limit is not None and emitted >= limit:
                return
            image_id = str(row["image_id"])
            code = MODEL_LABEL_TO_CODE.get(row["dx"])
            if code is None:
                raise DatasetLoadingError(
                    f"Etiqueta dx desconocida '{row['dx']}' en la imagen '{image_id}'."
                )
            yield LabeledImage(
                image_id=image_id,
                lesion_id=str(row["lesion_id"]),
                code=code,
                image=_decode_image(row["image"]["bytes"], image_id),
            )
            emitted += 1


def fetch_split_ids(
    split_prefix: str,
    filesystem: SplitFileSystem | None = None,
) -> frozenset[str]:
    """Une los `image_id` de los parquet del split indicado leyendo solo esa columna."""
    fs: SplitFileSystem = filesystem if filesystem is not None else HfFileSystem()
    data_dir = f"datasets/{DATASET_ID}@{DATASET_REVISION}/data"
    try:
        names = [
            name
            for name in fs.ls(data_dir, detail=False)
            if name.rsplit("/", 1)[-1].startswith(f"{split_prefix}-") and name.endswith(".parquet")
        ]
        if not names:
            raise DatasetLoadingError(
                f"No se encontraron archivos del split '{split_prefix}' en '{DATASET_ID}'."
            )
        ids: set[str] = set()
        for name in sorted(names):
            with fs.open(name, "rb") as handle:
                table = pq.read_table(handle, columns=["image_id"])
            ids.update(str(value) for value in table.column("image_id").to_pylist())
    except DatasetLoadingError:
        raise
    except Exception as err:
        raise DatasetLoadingError(
            f"No fue posible leer los IDs del split '{split_prefix}': {err}"
        ) from err
    return frozenset(ids)
