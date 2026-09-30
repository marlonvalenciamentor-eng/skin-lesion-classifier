import subprocess
import sys

import httpx
import pytest

from skin_lesion_classifier.api_client import ApiClientError, DiagnosisClient

VALID_PROBABILITIES = {
    "akiec": 0.02,
    "bcc": 0.02,
    "bkl": 0.02,
    "df": 0.02,
    "mel": 0.90,
    "nv": 0.01,
    "vasc": 0.01,
}

VALID_DIAGNOSIS_BODY = {
    "label": "mel",
    "label_name": "Melanoma",
    "severity": "maligno",
    "confidence": 0.90,
    "probabilities": VALID_PROBABILITIES,
    "overlay_png_base64": "aGVsbG8=",
    "latency_ms": 42.0,
    "model_id": "Anwarkh1/Skin_Cancer-Image_Classification",
}

VALID_HEALTH_BODY = {
    "status": "ok",
    "model_loaded": True,
    "model_id": "Anwarkh1/Skin_Cancer-Image_Classification",
}


def _client(handler: object) -> DiagnosisClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return DiagnosisClient(base_url="http://testserver", timeout=5.0, transport=transport)


def test_health_returns_parsed_health_response() -> None:
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json=VALID_HEALTH_BODY)

    client = _client(handler)

    # Act
    health = client.health()

    # Assert
    assert health.status == "ok"
    assert health.model_loaded is True


def test_diagnose_returns_parsed_diagnosis_response() -> None:
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(200, json=VALID_DIAGNOSIS_BODY)

    client = _client(handler)

    # Act
    diagnosis = client.diagnose(b"fake-png-bytes", "lesion.png", "image/png")

    # Assert
    assert diagnosis.label == "mel"
    assert diagnosis.confidence == pytest.approx(0.90)


def test_diagnose_raises_api_client_error_with_server_detail_on_4xx() -> None:
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            422,
            json={
                "detail": "No se pudo decodificar la imagen.",
                "error_type": "UndecodableImageError",
            },
        )

    client = _client(handler)

    # Act / Assert
    with pytest.raises(ApiClientError, match="No se pudo decodificar la imagen"):
        client.diagnose(b"garbage", "broken.png", "image/png")


def test_diagnose_raises_api_client_error_on_malformed_payload() -> None:
    # Arrange: 200 OK but the body violates the DiagnosisResponse contract (missing field).
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"label": "mel"})

    client = _client(handler)

    # Act / Assert
    with pytest.raises(ApiClientError, match="contrato"):
        client.diagnose(b"fake-png-bytes", "lesion.png", "image/png")


def test_diagnose_raises_api_client_error_on_connection_failure() -> None:
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)

    # Act / Assert
    with pytest.raises(ApiClientError, match="no está disponible"):
        client.diagnose(b"fake-png-bytes", "lesion.png", "image/png")


def test_health_raises_api_client_error_on_timeout() -> None:
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = _client(handler)

    # Act / Assert
    with pytest.raises(ApiClientError, match="no respondió a tiempo"):
        client.health()


def test_importing_client_modules_does_not_load_torch_or_transformers() -> None:
    # Arrange: a fresh subprocess so `sys.modules` reflects only what these imports pull in.
    script = (
        "import sys\n"
        "import skin_lesion_classifier.api_client\n"
        "import skin_lesion_classifier.schemas\n"
        "import skin_lesion_classifier.labels\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'transformers' not in sys.modules, 'transformers was imported'\n"
        "print('OK')\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Assert
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
