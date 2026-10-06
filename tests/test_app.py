from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_renders_header_and_inputs_without_loading_model() -> None:
    # Arrange
    # Act: sin imagen cargada, el script se detiene antes de cargar el ViT.
    app = AppTest.from_file(APP_PATH).run(timeout=30)

    # Assert
    assert not app.exception
    assert any(title.value == "SkinLesionClassifier" for title in app.title)
    assert len(app.file_uploader) == 1
