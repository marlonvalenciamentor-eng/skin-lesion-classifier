# =============================================================================
# SERVICIO ÚNICO: Streamlit UI + Fachada + Modelo
# PROYECTO: SkinLesionClassifier
# RESPONSABILIDAD: Un solo contenedor que levanta la interfaz Streamlit, la
#                  fachada orquestadora y el modelo ViT + Grad-CAM juntos. La UI
#                  consume la fachada directamente (sin API intermedia).
#
# Build (desde la raíz del repo, contexto = .):
#   docker build -f docker/Dockerfile -t skin-lesion-classifier .
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
RUN uv sync --locked --no-dev --no-install-project

# Capa de código: instala el paquete local una vez que src/ está presente.
COPY src/ ./src/
COPY app.py ./
RUN uv sync --locked --no-dev

# Usuario no-root; el volumen de caché de HF se monta en docker-compose.yml.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p "${HF_HOME}" \
    && chown -R appuser:appuser /app "${HF_HOME}"
USER appuser

EXPOSE 8501

# python:3.13-slim no trae curl; se usa urllib de la biblioteca estándar.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
