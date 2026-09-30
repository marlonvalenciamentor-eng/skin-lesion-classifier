# =============================================================================
# SERVICIO: Web UI (Streamlit)
# PROYECTO: SkinLesionClassifier
# RESPONSABILIDAD: Interfaz de usuario ligera. Habla con la API de inferencia
#                  por HTTP/JSON; nunca instala torch/transformers.
#
# Build (desde la raíz del repo, contexto = .):
#   docker build -f docker/ui.Dockerfile -t skin-lesion-classifier-ui .
# =============================================================================
FROM python:3.13-slim

# uv pinneado a la versión usada en desarrollo (0.12.9); nunca pip (CONSTITUTION.md).
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    API_URL=http://api:8000

WORKDIR /app

# Capa de dependencias: liviana a propósito (grupo 'ui'), sin torch/transformers.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-default-groups --group ui --no-install-project

# Capa de código.
COPY src/ ./src/
COPY app.py ./
RUN uv sync --locked --no-dev --no-default-groups --group ui

# Usuario no-root; data/samples se monta de solo lectura en docker-compose.yml.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# python:3.13-slim no trae curl; se usa urllib de la biblioteca estándar.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
