"""
=============================================================================
MÓDULO: api.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Puerto HTTP (FastAPI) sobre la Fachada
RESPONSABILIDAD ÚNICA:
    Exponer el pipeline de diagnóstico (`DermatologyDiagnosticFacade`) como un
    servicio HTTP con contratos Pydantic fuertes (`schemas.py`), de modo que la
    interfaz de usuario pueda ejecutarse en un proceso separado sin importar
    jamás PyTorch ni Transformers. El modelo se carga una única vez al inicio
    (lifespan) para que /health refleje el estado real de disponibilidad.

FLUJO:
    UploadFile -> validación (tipo/tamaño/decodificación) -> facade.diagnose()
    -> DiagnosisResponse (overlay Grad-CAM reescalado al aspect ratio original)
=============================================================================
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Protocol, cast

from fastapi import FastAPI, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from skin_lesion_classifier.facade import DermatologyDiagnosticFacade, DiagnosticResult
from skin_lesion_classifier.gradcam import GradCAMError
from skin_lesion_classifier.inference import InferenceError
from skin_lesion_classifier.labels import human_name, severity_of
from skin_lesion_classifier.model_loader import (
    MODEL_ID,
    ModelLoadingError,
    load_inference_service,
)
from skin_lesion_classifier.schemas import (
    DiagnosisResponse,
    ErrorResponse,
    HealthResponse,
    LesionCode,
)

logger = logging.getLogger(__name__)

# Multipart upload limits and allowed image formats for POST /predict.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_CONTENT_TYPES = frozenset({"image/png", "image/jpeg"})

# The Grad-CAM overlay is produced at the model's square crop resolution
# (typically 224x224); the API resizes it back to the original image's aspect
# ratio, capping the long side, so the UI never shows a distorted heatmap.
_OVERLAY_MAX_LONG_SIDE = 512

# Multipart framing (boundaries, per-part headers, filename) adds a modest amount of
# overhead on top of the raw file bytes. This margin lets the ASGI-level circuit
# breaker (UploadSizeLimitMiddleware) accept a legitimately-sized upload while still
# rejecting a request whose *declared or streamed* body is far larger than any real
# image upload -- well before Starlette would buffer/parse it.
UPLOAD_SIZE_MARGIN_BYTES = 64 * 1024  # 64 KB
_MAX_REQUEST_BODY_BYTES = MAX_UPLOAD_BYTES + UPLOAD_SIZE_MARGIN_BYTES


class DiagnosisProvider(Protocol):
    """Contract of a component able to run the full diagnostic pipeline."""

    def diagnose(self, image: Image.Image, target_class: str | None = None) -> DiagnosticResult: ...


FacadeFactory = Callable[[], DiagnosisProvider]


class ModelUnavailableError(RuntimeError):
    """Raised when /predict is called while the model failed to load or is not ready."""


class UnsupportedMediaTypeError(RuntimeError):
    """Raised when the uploaded file's content type is not PNG or JPEG."""


class PayloadTooLargeError(RuntimeError):
    """Raised when the uploaded file exceeds MAX_UPLOAD_BYTES."""


class UndecodableImageError(RuntimeError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


# Small synthetic RGB image used only to warm up the model right after loading.
_WARMUP_IMAGE_SIZE = (224, 224)


def _warm_up(facade: DiagnosisProvider) -> None:
    """
    Run one throwaway diagnosis right after loading the model.

    PyTorch lazily initializes kernels/threads on the first real forward+backward
    pass (Grad-CAM needs both), so the first real request pays a one-time cost the
    model wasn't warmed up (observed ~4.5s vs ~1.1s on later requests). A failed
    warm-up is logged and swallowed: it must never prevent the service from
    starting, since /health and later requests will still surface a real problem.
    """
    try:
        warmup_image = Image.new("RGB", _WARMUP_IMAGE_SIZE, color=(127, 127, 127))
        facade.diagnose(warmup_image)
    except Exception as err:  # noqa: BLE001 -- warm-up must never crash startup
        logger.warning(f"El calentamiento inicial del modelo falló (se ignora): {err}")


def _default_facade_factory() -> DiagnosisProvider:
    facade = DermatologyDiagnosticFacade(inference_service=load_inference_service())
    _warm_up(facade)
    return facade


def _diagnose_synchronized(
    facade: DiagnosisProvider,
    image: Image.Image,
    lock: threading.Lock,
) -> DiagnosticResult:
    """
    Run `facade.diagnose()` while holding `lock`.

    `ViTGradCAM.explain()` registers a forward hook on a module shared by every
    request and calls `.backward()` on it; two concurrent calls on the same
    model instance could cross-capture each other's activations/gradients. Each
    call already runs in its own worker thread (via `run_in_threadpool`), so a
    plain `threading.Lock` here serializes actual model access without blocking
    the asyncio event loop.
    """
    with lock:
        return facade.diagnose(image)


def _error_response(status_code: int, detail: str, error_type: str) -> JSONResponse:
    payload = ErrorResponse(detail=detail, error_type=error_type)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _parse_content_length(raw_headers: Iterable[tuple[bytes, bytes]]) -> int | None:
    for key, value in raw_headers:
        if key.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class UploadSizeLimitMiddleware:
    """
    Pure-ASGI middleware guarding `POST /predict` against oversized request bodies.

    Starlette's multipart parser fully buffers/spools the request body before the
    route handler (and its post-read `MAX_UPLOAD_BYTES` check) ever runs, so a huge
    upload would already be parsed into memory or a temp file by the time that
    in-route check could reject it. This middleware rejects the request earlier:
    immediately when `Content-Length` declares a body over the limit, or as soon as
    streamed bytes exceed the limit when `Content-Length` is absent (chunked
    transfer encoding) -- in both cases, before Starlette ever sees the body.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/predict"
        ):
            await self._app(scope, receive, send)
            return

        content_length = _parse_content_length(scope.get("headers") or [])
        if content_length is not None:
            if content_length > _MAX_REQUEST_BODY_BYTES:
                await self._reject(scope, receive, send)
                return
            await self._app(scope, receive, send)
            return

        # No Content-Length (chunked): buffer the body ourselves, capped at the
        # limit, then replay it to the downstream app if it stayed within bounds.
        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)
                break
            total += len(message.get("body", b""))
            buffered.append(message)
            if total > _MAX_REQUEST_BODY_BYTES:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        buffered_iter = iter(buffered)

        async def replay_receive() -> Message:
            try:
                return next(buffered_iter)
            except StopIteration:
                return await receive()

        await self._app(scope, replay_receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = _error_response(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            "PayloadTooLargeError",
        )
        await response(scope, receive, send)


def _encode_overlay_png_base64(overlay: Image.Image, original_size: tuple[int, int]) -> str:
    """Resize the Grad-CAM overlay to the original aspect ratio and PNG+base64 encode it."""
    original_width, original_height = original_size
    if original_width <= 0 or original_height <= 0:
        target_size = overlay.size
    else:
        aspect_ratio = original_width / original_height
        if aspect_ratio >= 1.0:
            target_width = _OVERLAY_MAX_LONG_SIDE
            target_height = max(1, round(_OVERLAY_MAX_LONG_SIDE / aspect_ratio))
        else:
            target_height = _OVERLAY_MAX_LONG_SIDE
            target_width = max(1, round(_OVERLAY_MAX_LONG_SIDE * aspect_ratio))
        target_size = (target_width, target_height)

    resized = overlay.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_app(facade_factory: FacadeFactory | None = None) -> FastAPI:
    """
    Build the FastAPI application, optionally injecting a facade factory.

    Production code omits `facade_factory` and gets the real
    `DermatologyDiagnosticFacade` built inside the lifespan (model loaded once
    at startup). Tests inject a factory returning a fake facade so that unit
    tests never load the real model or touch the network.
    """
    factory = facade_factory or _default_facade_factory
    state: dict[str, DiagnosisProvider | None] = {"facade": None}
    diagnose_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            state["facade"] = factory()
        except ModelLoadingError as err:
            logger.error(f"No se pudo cargar el modelo de clasificación al iniciar: {err}")
            state["facade"] = None
        yield
        state["facade"] = None

    app = FastAPI(title="SkinLesionClassifier Inference API", lifespan=lifespan)
    app.add_middleware(UploadSizeLimitMiddleware)

    @app.exception_handler(InferenceError)
    async def _handle_inference_error(_request: Request, exc: InferenceError) -> JSONResponse:
        # The domain exception's own message may embed low-level details (tensor
        # errors, library internals); log it server-side but never relay it as-is.
        logger.warning(f"InferenceError en /predict: {exc}")
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "No se pudo procesar la imagen para el diagnóstico. Intente con otra imagen.",
            "InferenceError",
        )

    @app.exception_handler(GradCAMError)
    async def _handle_gradcam_error(_request: Request, exc: GradCAMError) -> JSONResponse:
        logger.warning(f"GradCAMError en /predict: {exc}")
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "No se pudo generar la explicación visual del diagnóstico.",
            "GradCAMError",
        )

    @app.exception_handler(ModelUnavailableError)
    async def _handle_model_unavailable(
        _request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        return _error_response(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc), "ModelLoadingError")

    @app.exception_handler(UnsupportedMediaTypeError)
    async def _handle_unsupported_media_type(
        _request: Request, exc: UnsupportedMediaTypeError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc), "UnsupportedMediaTypeError"
        )

    @app.exception_handler(PayloadTooLargeError)
    async def _handle_payload_too_large(
        _request: Request, exc: PayloadTooLargeError
    ) -> JSONResponse:
        return _error_response(status.HTTP_413_CONTENT_TOO_LARGE, str(exc), "PayloadTooLargeError")

    @app.exception_handler(UndecodableImageError)
    async def _handle_undecodable_image(
        _request: Request, exc: UndecodableImageError
    ) -> JSONResponse:
        logger.warning(f"UndecodableImageError en /predict: {exc}")
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "No se pudo procesar la imagen enviada. Verifique que sea un archivo PNG o JPEG "
            "válido.",
            "UndecodableImageError",
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(f"RequestValidationError en la API: {exc}")
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "La solicitud no cumple el formato esperado. Verifique los campos enviados.",
            "RequestValidationError",
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) and exc.detail else "Error HTTP."
        return _error_response(exc.status_code, detail, "HTTPException")

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Error interno no controlado en la API de inferencia: {exc}")
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Error interno inesperado en el servicio de inferencia.",
            "InternalServerError",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        facade = state["facade"]
        return HealthResponse(status="ok", model_loaded=facade is not None, model_id=MODEL_ID)

    @app.post("/predict", response_model=DiagnosisResponse)
    async def predict(file: UploadFile) -> DiagnosisResponse:
        facade = state["facade"]
        if facade is None:
            raise ModelUnavailableError(
                "El modelo de clasificación no está disponible. Intente nuevamente en unos "
                "segundos o revise los registros del servicio."
            )

        if file.content_type not in _ALLOWED_CONTENT_TYPES:
            raise UnsupportedMediaTypeError(
                f"Tipo de contenido no soportado: '{file.content_type}'. Use una imagen PNG o JPEG."
            )

        raw_bytes = await file.read()
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise PayloadTooLargeError(
                f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )

        try:
            opened = Image.open(io.BytesIO(raw_bytes))
            opened.load()
            image: Image.Image = opened.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as err:
            raise UndecodableImageError(f"No se pudo decodificar la imagen: {err}") from err

        # Model inference is CPU-bound and synchronous; offload it to a worker thread so
        # it never blocks the event loop from serving other requests (e.g. /health).
        # The lock serializes actual model access across concurrent requests.
        started = time.perf_counter()
        result = await run_in_threadpool(_diagnose_synchronized, facade, image, diagnose_lock)
        elapsed_ms = (time.perf_counter() - started) * 1000

        overlay_b64 = _encode_overlay_png_base64(result.superimposed_image, image.size)

        return DiagnosisResponse(
            label=cast(LesionCode, result.label),
            label_name=human_name(result.label),
            severity=severity_of(result.label),
            confidence=result.confidence,
            probabilities=cast(dict[LesionCode, float], result.probabilities),
            overlay_png_base64=overlay_b64,
            latency_ms=elapsed_ms,
            model_id=MODEL_ID,
        )

    return app


app = create_app()
