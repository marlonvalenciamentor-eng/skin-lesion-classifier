import asyncio
import base64
import io
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from skin_lesion_classifier.api import (
    MAX_UPLOAD_BYTES,
    UPLOAD_SIZE_MARGIN_BYTES,
    DiagnosisProvider,
    create_app,
)
from skin_lesion_classifier.facade import DiagnosticResult
from skin_lesion_classifier.gradcam import GradCAMError
from skin_lesion_classifier.inference import InferenceError
from skin_lesion_classifier.model_loader import MODEL_ID, ModelLoadingError
from skin_lesion_classifier.schemas import ErrorResponse

SEVEN_CLASS_PROBABILITIES = {
    "akiec": 0.02,
    "bcc": 0.02,
    "bkl": 0.02,
    "df": 0.02,
    "mel": 0.90,
    "nv": 0.01,
    "vasc": 0.01,
}


class FakeFacade:
    """Test double standing in for DermatologyDiagnosticFacade.diagnose()."""

    def __init__(
        self,
        result: DiagnosticResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result or _make_result()
        self._error = error

    def diagnose(self, image: Image.Image, target_class: str | None = None) -> DiagnosticResult:
        del image, target_class
        if self._error is not None:
            raise self._error
        return self._result


def _make_result(
    label: str = "mel",
    confidence: float = 0.90,
    overlay_size: tuple[int, int] = (224, 224),
) -> DiagnosticResult:
    probabilities = dict(SEVEN_CLASS_PROBABILITIES)
    probabilities[label] = confidence
    return DiagnosticResult(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        heatmap=np.zeros((224, 224), dtype=np.float32),
        superimposed_image=Image.new("RGB", overlay_size, (128, 64, 32)),
        target_class=label,
    )


def _png_bytes(size: tuple[int, int] = (300, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _client(facade: DiagnosisProvider) -> TestClient:
    app = create_app(facade_factory=lambda: facade)
    return TestClient(app)


def test_health_reports_ok_and_model_loaded() -> None:
    # Arrange
    with _client(FakeFacade()) as client:
        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_id"] == MODEL_ID


def test_predict_with_valid_png_returns_diagnosis_response() -> None:
    # Arrange
    with _client(FakeFacade(_make_result(label="mel", confidence=0.90))) as client:
        # Act
        response = client.post(
            "/predict", files={"file": ("lesion.png", _png_bytes(), "image/png")}
        )

        # Assert
        assert response.status_code == 200
        body = response.json()
        assert body["label"] == "mel"
        assert body["label_name"] == "Melanoma"
        assert body["severity"] == "maligno"
        assert body["confidence"] == pytest.approx(0.90)
        assert set(body["probabilities"]) == set(SEVEN_CLASS_PROBABILITIES)
        assert body["overlay_png_base64"]
        assert body["latency_ms"] >= 0
        assert body["model_id"] == MODEL_ID


def test_predict_rejects_unsupported_content_type() -> None:
    # Arrange
    with _client(FakeFacade()) as client:
        # Act
        response = client.post(
            "/predict", files={"file": ("lesion.txt", _png_bytes(), "text/plain")}
        )

        # Assert
        assert response.status_code == 415
        body = response.json()
        assert "detail" in body and "error_type" in body


def test_predict_rejects_oversized_upload() -> None:
    # Arrange
    oversized = b"0" * (MAX_UPLOAD_BYTES + 1)
    with _client(FakeFacade()) as client:
        # Act
        response = client.post("/predict", files={"file": ("big.png", oversized, "image/png")})

        # Assert
        assert response.status_code == 413
        assert "error_type" in response.json()


def test_predict_rejects_corrupt_image_bytes() -> None:
    # Arrange
    with _client(FakeFacade()) as client:
        # Act
        response = client.post(
            "/predict", files={"file": ("broken.png", b"not a real png", "image/png")}
        )

        # Assert
        assert response.status_code == 422
        body = ErrorResponse.model_validate(response.json())
        assert body.error_type == "UndecodableImageError"
        # The client never sees raw PIL internals (object addresses, exception repr).
        assert "_io.BytesIO" not in body.detail
        assert "0x" not in body.detail


def test_predict_rejects_declared_content_length_far_beyond_margin() -> None:
    # Arrange: a request body whose declared Content-Length is well beyond even the
    # multipart-overhead margin -- the ASGI middleware must reject it before Starlette
    # ever buffers/parses the multipart body.
    huge = b"0" * (MAX_UPLOAD_BYTES + UPLOAD_SIZE_MARGIN_BYTES + 1)
    with _client(FakeFacade()) as client:
        # Act
        response = client.post("/predict", files={"file": ("huge.png", huge, "image/png")})

        # Assert
        assert response.status_code == 413
        body = ErrorResponse.model_validate(response.json())
        assert body.error_type == "PayloadTooLargeError"


async def _collect_asgi_response(
    app: Callable[..., Awaitable[None]],
    scope: dict[str, Any],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    message_iter = iter(messages)

    async def receive() -> dict[str, Any]:
        try:
            return next(message_iter)
        except StopIteration:
            return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def test_predict_rejects_chunked_body_without_content_length_once_over_limit() -> None:
    # Arrange: no Content-Length header (as with chunked transfer-encoding); the body
    # streams in over several chunks that add up to more than the allowed limit.
    app = create_app(facade_factory=lambda: FakeFacade())
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [(b"content-type", b"multipart/form-data; boundary=x")],
    }
    chunk = b"0" * (64 * 1024)
    chunk_count = (MAX_UPLOAD_BYTES + UPLOAD_SIZE_MARGIN_BYTES) // len(chunk) + 2
    messages: list[dict[str, Any]] = [
        {"type": "http.request", "body": chunk, "more_body": True} for _ in range(chunk_count)
    ]
    messages.append({"type": "http.request", "body": b"", "more_body": False})

    # Act
    sent = asyncio.run(_collect_asgi_response(app, scope, messages))

    # Assert
    start_message = next(message for message in sent if message["type"] == "http.response.start")
    assert start_message["status"] == 413
    body_message = next(message for message in sent if message["type"] == "http.response.body")
    error = ErrorResponse.model_validate_json(body_message["body"])
    assert error.error_type == "PayloadTooLargeError"


def test_predict_maps_decompression_bomb_to_422(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: force Image.open() to raise as if the payload were a decompression bomb.
    def _raise_bomb(*_args: object, **_kwargs: object) -> Image.Image:
        raise Image.DecompressionBombError(
            "Imagen sospechosamente grande (bomba de descompresión)."
        )

    monkeypatch.setattr("skin_lesion_classifier.api.Image.open", _raise_bomb)

    with _client(FakeFacade()) as client:
        # Act
        response = client.post(
            "/predict", files={"file": ("lesion.png", _png_bytes(), "image/png")}
        )

        # Assert
        assert response.status_code == 422
        body = ErrorResponse.model_validate(response.json())
        assert body.error_type == "UndecodableImageError"


def test_predict_maps_inference_error_to_422() -> None:
    # Arrange
    with _client(FakeFacade(error=InferenceError("Fallo de inferencia simulado"))) as client:
        # Act
        response = client.post(
            "/predict", files={"file": ("lesion.png", _png_bytes(), "image/png")}
        )

        # Assert
        assert response.status_code == 422
        body = ErrorResponse.model_validate(response.json())
        assert body.error_type == "InferenceError"
        # The client sees a fixed, user-facing message -- not the raw domain exception
        # text, which may embed low-level details (tensor errors, object reprs, etc.).
        assert "Fallo de inferencia simulado" not in body.detail
        assert body.detail


def test_predict_maps_gradcam_error_to_500() -> None:
    # Arrange
    with _client(FakeFacade(error=GradCAMError("Fallo de explicación simulado"))) as client:
        # Act
        response = client.post(
            "/predict", files={"file": ("lesion.png", _png_bytes(), "image/png")}
        )

        # Assert
        assert response.status_code == 500
        body = ErrorResponse.model_validate(response.json())
        assert body.error_type == "GradCAMError"
        assert "Fallo de explicación simulado" not in body.detail
        assert body.detail


def test_predict_missing_file_returns_error_envelope() -> None:
    # Arrange
    with _client(FakeFacade()) as client:
        # Act: no `files=` at all -- FastAPI's own request validation should reject it.
        response = client.post("/predict")

        # Assert
        assert response.status_code == 422
        body = ErrorResponse.model_validate(response.json())
        assert body.error_type == "RequestValidationError"
        assert body.detail


def test_unknown_route_returns_error_envelope() -> None:
    # Arrange
    with _client(FakeFacade()) as client:
        # Act
        response = client.get("/does-not-exist")

        # Assert
        assert response.status_code == 404
        body = ErrorResponse.model_validate(response.json())
        assert body.detail


def test_predict_returns_503_when_model_failed_to_load() -> None:
    # Arrange
    def failing_factory() -> DiagnosisProvider:
        raise ModelLoadingError("No se pudo cargar el modelo simulado")

    app = create_app(facade_factory=failing_factory)
    with TestClient(app) as client:
        # Act
        health = client.get("/health")
        predict = client.post("/predict", files={"file": ("lesion.png", _png_bytes(), "image/png")})

        # Assert
        assert health.json()["model_loaded"] is False
        assert predict.status_code == 503
        assert predict.json()["error_type"] == "ModelLoadingError"


def test_warm_up_calls_diagnose_once_with_a_small_synthetic_image() -> None:
    # Arrange
    from skin_lesion_classifier.api import _warm_up

    calls: list[Image.Image] = []

    class RecordingFacade:
        def diagnose(self, image: Image.Image, target_class: str | None = None) -> DiagnosticResult:
            del target_class
            calls.append(image)
            return _make_result()

    # Act
    _warm_up(RecordingFacade())

    # Assert
    assert len(calls) == 1
    assert calls[0].mode == "RGB"
    assert max(calls[0].size) <= 224


def test_warm_up_swallows_failures_without_raising() -> None:
    # Arrange
    from skin_lesion_classifier.api import _warm_up

    class FailingFacade:
        def diagnose(self, image: Image.Image, target_class: str | None = None) -> DiagnosticResult:
            del image, target_class
            raise InferenceError("Fallo simulado de calentamiento")

    # Act / Assert: must not raise.
    _warm_up(FailingFacade())


def test_default_facade_factory_warms_up_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    from skin_lesion_classifier import api as api_module

    fake_service = object()
    monkeypatch.setattr(api_module, "load_inference_service", lambda: fake_service)

    created: list[object] = []

    class FakeDermatologyFacade:
        def __init__(self, inference_service: object) -> None:
            created.append(self)
            self.inference_service = inference_service

    monkeypatch.setattr(api_module, "DermatologyDiagnosticFacade", FakeDermatologyFacade)

    warm_up_calls: list[object] = []
    monkeypatch.setattr(api_module, "_warm_up", lambda facade: warm_up_calls.append(facade))

    # Act
    result = api_module._default_facade_factory()

    # Assert
    assert created == [result]
    assert warm_up_calls == [result]


def test_predict_serializes_concurrent_diagnose_calls_on_shared_model() -> None:
    # Arrange: the shared model (ViTGradCAM registers a forward hook + calls
    # backward() on ONE module) must never process two requests at the same time,
    # even though /predict offloads each call to a worker thread.
    tracker_lock = threading.Lock()
    tracker = {"current": 0, "max": 0}

    class OverlapDetectingFacade:
        def diagnose(self, image: Image.Image, target_class: str | None = None) -> DiagnosticResult:
            del image, target_class
            with tracker_lock:
                tracker["current"] += 1
                tracker["max"] = max(tracker["max"], tracker["current"])
            time.sleep(0.05)  # long enough for real overlap to occur if unsynchronized
            with tracker_lock:
                tracker["current"] -= 1
            return _make_result()

    with _client(OverlapDetectingFacade()) as client:

        def call() -> httpx.Response:
            return cast(
                httpx.Response,
                client.post("/predict", files={"file": ("lesion.png", _png_bytes(), "image/png")}),
            )

        # Act
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(call) for _ in range(6)]
            responses = [future.result() for future in futures]

    # Assert
    assert all(response.status_code == 200 for response in responses)
    assert tracker["max"] == 1


def test_predict_overlay_keeps_original_aspect_ratio() -> None:
    # Arrange: a wide 300x200 (3:2) original, but the Grad-CAM overlay is a square 224x224
    # (typical ViT crop) -- the API must resize it back to the original aspect ratio.
    result = _make_result(overlay_size=(224, 224))
    with _client(FakeFacade(result)) as client:
        # Act
        response = client.post(
            "/predict", files={"file": ("lesion.png", _png_bytes((300, 200)), "image/png")}
        )

        # Assert
        body = response.json()
        overlay_bytes = base64.b64decode(body["overlay_png_base64"])
        overlay = Image.open(io.BytesIO(overlay_bytes))
        assert overlay.format == "PNG"
        original_ratio = 300 / 200
        overlay_ratio = overlay.width / overlay.height
        assert overlay_ratio == pytest.approx(original_ratio, rel=0.05)
        assert max(overlay.size) <= 512
