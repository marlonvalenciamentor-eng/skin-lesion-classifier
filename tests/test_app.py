import runpy
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _load_app_module_namespace() -> dict[str, Any]:
    """
    Execute app.py with `__name__ != "__main__"` so only its function/constant
    definitions run, never the Streamlit UI body (which is guarded behind
    `if __name__ == "__main__":`). This lets pure helpers be unit-tested directly
    without a running Streamlit session or a live API.
    """
    return runpy.run_path(APP_PATH, run_name="skin_lesion_classifier_app_under_test")


def test_app_renders_header_and_service_status_without_api_running() -> None:
    # Arrange
    # Act: no image selected and no API listening on the default localhost:8000 URL --
    # this must render a friendly "service unavailable" state, never an exception.
    app = AppTest.from_file(APP_PATH).run(timeout=30)

    # Assert
    assert not app.exception
    assert any(title.value == "SkinLesionClassifier" for title in app.title)
    assert len(app.file_uploader) == 1
    assert len(app.selectbox) == 1
    assert len(app.error) >= 1
    # The message must be one clear sentence -- not the domain error text glued onto
    # a wrapper sentence that already says the same thing twice.
    messages = [error.value for error in app.error]
    assert any(
        message == "Servicio de inferencia no disponible. Verifica que la API esté en ejecución."
        for message in messages
    )
    assert not any(message.count("no disponible") > 1 for message in messages)


def test_resolve_api_timeout_seconds_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("API_TIMEOUT_SECONDS", "12.5")

    # Act
    namespace = _load_app_module_namespace()

    # Assert
    assert namespace["_resolve_api_timeout_seconds"]() == pytest.approx(12.5)


def test_resolve_api_timeout_seconds_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv("API_TIMEOUT_SECONDS", raising=False)

    # Act
    namespace = _load_app_module_namespace()

    # Assert
    assert namespace["_resolve_api_timeout_seconds"]() == namespace["DEFAULT_API_TIMEOUT_SECONDS"]


def test_resolve_api_timeout_seconds_falls_back_on_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("API_TIMEOUT_SECONDS", "not-a-number")

    # Act
    namespace = _load_app_module_namespace()

    # Assert
    assert namespace["_resolve_api_timeout_seconds"]() == namespace["DEFAULT_API_TIMEOUT_SECONDS"]


def test_resolve_api_timeout_seconds_falls_back_on_non_positive_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("API_TIMEOUT_SECONDS", "-5")

    # Act
    namespace = _load_app_module_namespace()

    # Assert
    assert namespace["_resolve_api_timeout_seconds"]() == namespace["DEFAULT_API_TIMEOUT_SECONDS"]


def test_app_py_does_not_import_torch_transformers_or_facade() -> None:
    # Arrange
    source = Path(APP_PATH).read_text(encoding="utf-8")

    # Assert: static source check, robust regardless of what streamlit/AppTest itself
    # happens to import while running the script. Only import statements are checked --
    # the module header docstring is allowed to mention these names in prose.
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            assert "torch" not in stripped
            assert "transformers" not in stripped
            assert "facade" not in stripped
