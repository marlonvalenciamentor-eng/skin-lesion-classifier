# =============================================================================
# SERVICIO: Inference API (FastAPI)
# PROYECTO: SkinLesionClassifier
# RESPONSABILIDAD: Dueño exclusivo del modelo ViT + Grad-CAM. Expone el
#                  contrato Pydantic (schemas.py) por HTTP sobre uvicorn.
#
# Build (desde la raíz del repo, contexto = .):
#   docker build -f docker/api.Dockerfile -t skin-lesion-classifier-api .
# =============================================================================
FROM python:3.13-slim

# uv pinneado a la versión usada en desarrollo (0.12.9); nunca pip (CONSTITUTION.md).
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Capa de dependencias: se cachea por separado de src/ para que un cambio de
# código no invalide la (pesada) instalación de torch/transformers.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-default-groups --group api --no-install-project

# Capa de código: instala el paquete local una vez que src/ está presente.
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-default-groups --group api

# Usuario no-root; el volumen de caché de HF se monta en docker-compose.yml.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p "${HF_HOME}" \
    && chown -R appuser:appuser /app "${HF_HOME}"
USER appuser

EXPOSE 8000

# python:3.13-slim no trae curl; se usa urllib de la biblioteca estándar.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]

CMD ["uvicorn", "skin_lesion_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
