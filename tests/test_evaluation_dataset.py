import io
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from PIL import Image

from skin_lesion_classifier.evaluation_dataset import (
    DATASET_ID,
    DATASET_REVISION,
    TEST_PARQUET_FILENAME,
    DatasetLoadingError,
    LabeledImage,
    download_test_split,
    fetch_split_ids,
    iter_labeled_images,
)

DATA_DIR = f"datasets/{DATASET_ID}@{DATASET_REVISION}/data"


def encode_image(color: tuple[int, int, int], size: tuple[int, int] = (8, 6)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def build_parquet_bytes(rows: list[tuple[str, str, str, bytes]], row_group_size: int = 2) -> bytes:
    table = pa.table(
        {
            "image": [{"bytes": row[3], "path": f"{row[0]}.png"} for row in rows],
            "image_id": [row[0] for row in rows],
            "lesion_id": [row[1] for row in rows],
            "dx": [row[2] for row in rows],
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, row_group_size=row_group_size)
    return buffer.getvalue()


def write_parquet(path: Path, rows: list[tuple[str, str, str, bytes]]) -> Path:
    path.write_bytes(build_parquet_bytes(rows))
    return path


SAMPLE_ROWS = [
    ("ISIC_1", "HAM_1", "melanoma", encode_image((200, 10, 10))),
    ("ISIC_2", "HAM_2", "melanocytic_Nevi", encode_image((10, 200, 10))),
    ("ISIC_3", "HAM_3", "vascular_lesions", encode_image((10, 10, 200))),
    ("ISIC_4", "HAM_4", "basal_cell_carcinoma", encode_image((90, 90, 90))),
    ("ISIC_5", "HAM_5", "dermatofibroma", encode_image((30, 60, 90))),
]


class FakeFileSystem:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files
        self.opened: list[str] = []

    def ls(self, path: str, detail: bool = False) -> list[Any]:
        prefix = path.rstrip("/") + "/"
        return [name for name in self._files if name.startswith(prefix)]

    def open(self, path: str, mode: str = "rb") -> io.BytesIO:
        self.opened.append(path)
        return io.BytesIO(self._files[path])


def test_iter_labeled_images_maps_dx_to_short_codes_and_decodes_rgb(tmp_path: Path) -> None:
    parquet_path = write_parquet(tmp_path / "test.parquet", SAMPLE_ROWS)

    samples = list(iter_labeled_images(parquet_path))

    assert [sample.code for sample in samples] == ["mel", "nv", "vasc", "bcc", "df"]
    assert [sample.image_id for sample in samples] == [f"ISIC_{n}" for n in range(1, 6)]
    assert samples[0].lesion_id == "HAM_1"
    assert isinstance(samples[0], LabeledImage)
    assert samples[0].image.mode == "RGB"
    assert samples[0].image.size == (8, 6)
    assert samples[0].image.getpixel((0, 0)) == (200, 10, 10)


def test_iter_labeled_images_honours_limit_and_is_lazy(tmp_path: Path) -> None:
    parquet_path = write_parquet(tmp_path / "test.parquet", SAMPLE_ROWS)

    iterator = iter_labeled_images(parquet_path, limit=3, batch_size=2)
    first = next(iterator)
    rest = list(iterator)

    assert first.image_id == "ISIC_1"
    assert [sample.image_id for sample in rest] == ["ISIC_2", "ISIC_3"]


def test_iter_labeled_images_rejects_unknown_dx(tmp_path: Path) -> None:
    parquet_path = write_parquet(
        tmp_path / "test.parquet", [("ISIC_9", "HAM_9", "alien", encode_image((1, 2, 3)))]
    )

    with pytest.raises(DatasetLoadingError, match="alien"):
        list(iter_labeled_images(parquet_path))


def test_iter_labeled_images_rejects_undecodable_bytes(tmp_path: Path) -> None:
    parquet_path = write_parquet(
        tmp_path / "test.parquet", [("ISIC_9", "HAM_9", "melanoma", b"not an image")]
    )

    with pytest.raises(DatasetLoadingError, match="ISIC_9"):
        list(iter_labeled_images(parquet_path))


def test_iter_labeled_images_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadingError, match="parquet"):
        list(iter_labeled_images(tmp_path / "missing.parquet"))


def test_download_test_split_requests_pinned_dataset_file(tmp_path: Path) -> None:
    calls: list[dict[str, str]] = []

    def fake_downloader(*, repo_id: str, filename: str, repo_type: str, revision: str) -> str:
        calls.append(
            {"repo_id": repo_id, "filename": filename, "repo_type": repo_type, "revision": revision}
        )
        return str(tmp_path / "cached.parquet")

    path = download_test_split(downloader=fake_downloader)

    assert path == tmp_path / "cached.parquet"
    assert calls == [
        {
            "repo_id": DATASET_ID,
            "filename": TEST_PARQUET_FILENAME,
            "repo_type": "dataset",
            "revision": DATASET_REVISION,
        }
    ]


def test_download_test_split_wraps_downloader_failures() -> None:
    def failing_downloader(*, repo_id: str, filename: str, repo_type: str, revision: str) -> str:
        raise OSError("sin red")

    with pytest.raises(DatasetLoadingError, match="sin red"):
        download_test_split(downloader=failing_downloader)


def test_fetch_split_ids_unions_ids_of_every_matching_file() -> None:
    files = {
        f"{DATA_DIR}/train-00000-of-00002-aa.parquet": build_parquet_bytes(SAMPLE_ROWS[:2]),
        f"{DATA_DIR}/train-00001-of-00002-bb.parquet": build_parquet_bytes(SAMPLE_ROWS[2:4]),
        f"{DATA_DIR}/test-00000-of-00001-cc.parquet": build_parquet_bytes(SAMPLE_ROWS[4:]),
    }
    fake_fs = FakeFileSystem(files)

    ids = fetch_split_ids("train", filesystem=fake_fs)

    assert ids == frozenset({"ISIC_1", "ISIC_2", "ISIC_3", "ISIC_4"})
    assert len(fake_fs.opened) == 2


def test_fetch_split_ids_fails_when_no_file_matches() -> None:
    fake_fs = FakeFileSystem({})

    with pytest.raises(DatasetLoadingError, match="train"):
        fetch_split_ids("train", filesystem=fake_fs)
