"""
=============================================================================
MÓDULO: api_client.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Adaptador HTTP (cliente tipado)
RESPONSABILIDAD ÚNICA:
    Consumir el servicio de inferencia (`api.py`) por HTTP, validando cada
    respuesta contra los mismos contratos Pydantic (`schemas.py`) que el
    servidor. Este módulo es deliberadamente ligero (solo `httpx` + `pydantic`)
    para que la interfaz Streamlit lo importe sin instalar jamás PyTorch ni
    Transformers, y traduce fallos de red, HTTP o de contrato a un único tipo
    de excepción de dominio con mensajes en español.
=============================================================================
"""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from skin_lesion_classifier.schemas import DiagnosisResponse, ErrorResponse, HealthResponse

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ApiClientError(RuntimeError):
    """Excepción de dominio para fallos de comunicación con la API de inferencia."""


class DiagnosisClient:
    """
    Cliente HTTP tipado para el servicio de inferencia dermatológica.

    Valida cada respuesta contra los modelos Pydantic de `schemas.py`, de modo
    que el contrato entre la UI y la API se hace cumplir en ambos extremos del
    límite de proceso, no solo en el servidor.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Inicializa el cliente HTTP.

        Args:
            base_url: URL base del servicio de inferencia (ej. 'http://localhost:8000').
            timeout: Tiempo máximo de espera por solicitud, en segundos.
            transport: Transporte HTTP inyectable (ej. `httpx.MockTransport` en pruebas).
        """
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        """Cierra la conexión HTTP subyacente."""
        self._client.close()

    def __enter__(self) -> DiagnosisClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def health(self) -> HealthResponse:
        """
        Consulta el estado del servicio de inferencia.

        Returns:
            HealthResponse: Estado y disponibilidad del modelo.

        Raises:
            ApiClientError: Si el servicio no responde o el contrato es inválido.
        """
        response = self._request("GET", "/health")
        return self._parse(response, HealthResponse)

    def diagnose(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> DiagnosisResponse:
        """
        Envía una imagen dermatoscópica al servicio de inferencia y valida el diagnóstico.

        Args:
            image_bytes: Contenido binario de la imagen (PNG o JPEG).
            filename: Nombre de archivo original, reenviado como metadato multipart.
            content_type: Tipo MIME de la imagen (ej. 'image/png').

        Returns:
            DiagnosisResponse: Diagnóstico validado según el contrato compartido.

        Raises:
            ApiClientError: Si el servicio no está disponible, responde con un error
                             HTTP, o la respuesta no cumple el contrato esperado.
        """
        files = {"file": (filename, image_bytes, content_type)}
        response = self._request("POST", "/predict", files=files)
        return self._parse(response, DiagnosisResponse)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as err:
            raise ApiClientError(
                "El servicio de inferencia no respondió a tiempo. Intente nuevamente."
            ) from err
        except httpx.ConnectError as err:
            raise ApiClientError(
                "El servicio de inferencia no está disponible. "
                "Verifique que esté en ejecución y accesible."
            ) from err
        except httpx.HTTPError as err:
            raise ApiClientError(
                f"Error de comunicación con el servicio de inferencia: {err}"
            ) from err

    def _parse(self, response: httpx.Response, model: type[_ModelT]) -> _ModelT:
        if response.status_code >= 400:
            raise ApiClientError(self._error_detail(response))
        try:
            return model.model_validate_json(response.content)
        except ValidationError as err:
            raise ApiClientError(
                f"La respuesta del servicio de inferencia no cumple el contrato esperado: {err}"
            ) from err

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            error = ErrorResponse.model_validate_json(response.content)
        except (ValidationError, ValueError):
            return f"El servicio de inferencia respondió con error HTTP {response.status_code}."
        return error.detail
