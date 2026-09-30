# Ticket #009 — API de Inferencia (FastAPI) + Empaquetado Docker

## Objetivo

Dividir el sistema en dos servicios con contratos fuertes (Pydantic v2) y empaquetarlos con
Docker para paridad de entorno (Módulo 3):

1. **Servicio de inferencia** — FastAPI, dueño exclusivo del modelo (ViT + Grad-CAM).
2. **Interfaz web** — Streamlit, consume la API por HTTP y nunca importa `torch`,
   `transformers` ni la fachada.

## Problema

`app.py` importa hoy `DermatologyDiagnosticFacade` en proceso, acoplando la interfaz de
usuario al stack completo de Deep Learning (PyTorch, Transformers, ~343 MB de pesos). Esto
impide desplegar la UI de forma liviana, dificulta escalar el modelo de forma independiente
y no ofrece paridad de entorno entre desarrollo y despliegue. Se requiere un límite de
proceso explícito con un contrato de datos validado en ambos extremos.

## Alcance autorizado

- Crear `src/skin_lesion_classifier/schemas.py` (contratos Pydantic v2, puro, sin
  `torch`/`numpy`).
- Crear `src/skin_lesion_classifier/api.py` (FastAPI: `create_app()` + `app`).
- Crear `src/skin_lesion_classifier/api_client.py` (cliente HTTP tipado con `httpx`).
- Reescribir `app.py` para consumir la API por HTTP en lugar de la fachada en proceso.
- Crear `tests/test_schemas.py`, `tests/test_api.py`, `tests/test_api_client.py`.
- Actualizar `tests/test_app.py` (smoke sin API real + chequeo estático sin
  `torch`/`transformers`/`facade`).
- Actualizar `pyproject.toml`: grupos de dependencias `api`/`ui`/`dev`,
  `default-groups = ["dev", "api", "ui"]`.
- Crear `docker/api.Dockerfile`, `docker/ui.Dockerfile`, `docker-compose.yml`,
  `.dockerignore`.
- Añadir el job `docker` a `.github/workflows/ci.yml` (`docker compose build`).
- Actualizar `README.md` (ejecución local en dos terminales, ejecución con Docker,
  nota de arquitectura).
- Mantener la implementación en la rama `feature/inference-api-docker`.
- No descargar ni incorporar pesos o datasets al repositorio; no construir imágenes Docker
  localmente (Docker no está instalado en esta máquina — la validación de build queda a
  cargo de CI).

## Restricciones

- La interfaz web **nunca** importa `torch`, `transformers` ni `skin_lesion_classifier.facade`;
  solo consume `api_client` + `schemas` + `labels`.
- `schemas.py` es puro: sin dependencias de bajo nivel, para poder validarse en ambos lados
  del límite de proceso sin instalar el stack de Deep Learning.
- `DiagnosisResponse` es inmutable (`frozen=True`, `extra="forbid"`) y se autovalida:
  las 7 clases HAM10000 presentes, probabilidades en `[0,1]` que suman 1 (tolerancia),
  `label` es el argmax y `confidence` coincide con su probabilidad (tolerancia).
- Un único envelope de error (`ErrorResponse`) para todos los errores de la API; nunca se
  filtran tracebacks.
- Los tests unitarios de `api.py` y `api_client.py` NUNCA cargan el modelo real ni tocan la
  red: usan fachadas/transportes falsos inyectados.
- El overlay Grad-CAM se reescala al aspect ratio original de la imagen de entrada (lado
  largo acotado, ej. 512 px) antes de codificarse en PNG+base64, corrigiendo la distorsión
  cuadrado-vs-rectángulo observada en la UI anterior.
- No se modifica el comportamiento ni los tests de `gradcam.py`/`inference.py`/
  `model_loader.py` salvo necesidad estrictamente justificada.
- `uv sync` simple sigue instalando todo (dev + api + ui) para desarrolladores y CI; los
  Dockerfiles usan instalaciones ligeras por grupo (`--no-default-groups --group <api|ui>`).
- Artefactos técnicos en inglés salvo el encabezado institucional y los mensajes de dominio/UI,
  que siguen la convención existente en español.
- Usar `uv` (nunca `pip`) para todas las instalaciones y verificaciones.

## Tareas

- [x] T009-1 Actualizar `pyproject.toml`: mover `matplotlib`/`numpy`/`torch`/`torchvision`/
      `transformers` al grupo `api`, `streamlit` al grupo `ui`, añadir `pydantic`/`pillow` a
      `[project].dependencies`, `fastapi`/`uvicorn`/`python-multipart` al grupo `api`,
      `httpx` a `ui` y `dev`, y `default-groups = ["dev", "api", "ui"]`. Regenerar `uv.lock`.
- [x] T009-2 Escribir `tests/test_schemas.py` (RED) e implementar `schemas.py` (GREEN):
      `LesionCode`, `Severity`, `DiagnosisResponse`, `HealthResponse`, `ErrorResponse`.
- [x] T009-3 Escribir `tests/test_api.py` (RED) e implementar `api.py` (GREEN): `/health`,
      `/predict`, manejo de errores (415/413/422/500/503), overlay reescalado a aspect ratio
      original.
- [x] T009-4 Escribir `tests/test_api_client.py` (RED) e implementar `api_client.py`
      (GREEN): `DiagnosisClient`, `ApiClientError`, chequeo de subproceso sin
      `torch`/`transformers`.
- [x] T009-5 Reescribir `app.py` para usar `api_client` (sidebar con entrada + estado de
      servicio, tarjeta de resultados, imágenes lado a lado, leyenda de color, expander
      explicativo, copy en español neutro) y actualizar `tests/test_app.py`.
- [x] T009-6 Crear `docker/api.Dockerfile`, `docker/ui.Dockerfile`, `docker-compose.yml`,
      `.dockerignore`.
- [x] T009-7 Añadir el job `docker` a `.github/workflows/ci.yml`.
- [x] T009-8 Actualizar `README.md` y registrar evidencia de verificación en este ticket.
- [x] T009-9 Ronda de corrección acotada (revisión independiente + smoke real del padre):
      lock (`threading.Lock`) serializando `facade.diagnose()` en `/predict`; middleware
      ASGI puro (`UploadSizeLimitMiddleware`) que rechaza cargas sobredimensionadas antes
      de que Starlette las bufferee/parsee (con y sin `Content-Length`); envelope único de
      error también para `RequestValidationError` (422) y rutas desconocidas (404);
      `Image.DecompressionBombError` mapeado a 422; mensajes de error fijos y no
      filtrantes para `InferenceError`/`GradCAMError`/`UndecodableImageError` (detalle real
      solo en logs de servidor); `API_TIMEOUT_SECONDS` configurable en `app.py`; mensaje
      de "servicio no disponible" del sidebar sin duplicación; calentamiento (`_warm_up`)
      de una diagnosis sintética tras cargar el modelo real en `_default_facade_factory`.

## Criterios de aceptación

- `schemas.py` valida contratos correctos y rechaza: confianza fuera de rango, etiqueta
  desconocida, clase faltante, campo extra, suma de probabilidades incorrecta, `label` que
  no es el argmax; las instancias son inmutables.
- `GET /health` responde `HealthResponse`; `POST /predict` responde `DiagnosisResponse` con
  `response_model` declarado, o `ErrorResponse` con el código HTTP correcto (415/413/422/500/503).
- El overlay Grad-CAM en la respuesta conserva el aspect ratio de la imagen original.
- `DiagnosisClient` valida las respuestas con los modelos Pydantic y traduce fallos de
  red/timeout/HTTP/contrato a `ApiClientError` con mensajes en español.
- Un subproceso que importa `skin_lesion_classifier.api_client`, `schemas` y `labels` no
  carga `torch` ni `transformers` en `sys.modules`.
- `app.py` no importa `torch`, `transformers` ni `skin_lesion_classifier.facade`.
- `uv tree --no-default-groups --group ui` no lista `torch` ni `transformers`.
- `uv sync` simple sigue instalando todo para desarrollo/CI.
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` y
  `uv run mypy --strict src tests` pasan.
- El job `docker` de CI construye ambas imágenes con `docker compose build`.

## Checks aplicables

- `uv lock --check`
- `uv run pytest -q -p no:cacheprovider --basetemp=<tmp>`
- `uv run pytest -q -p no:cacheprovider --basetemp=<tmp> --cov=skin_lesion_classifier --cov-report=term`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy --strict src tests`
- `uv tree --no-default-groups --group ui` (sin `torch`/`transformers`)

## Progreso

- Rama: `feature/inference-api-docker`.
- T009-1: `pyproject.toml` reorganizado — `[project].dependencies` = `pillow`, `pydantic`;
  grupo `api` = `fastapi`, `matplotlib`, `numpy`, `python-multipart`, `torch`, `torchvision`,
  `transformers`, `uvicorn[standard]`; grupo `ui` = `httpx`, `streamlit`; `dev` añade `httpx`.
  `default-groups = ["dev", "api", "ui"]`. `uv lock` regenerado (92 paquetes resueltos).
- T009-2: `schemas.py` puro (sin `torch`/`numpy`) con `LesionCode`, `Severity`,
  `DiagnosisResponse` (frozen, `extra="forbid"`, validadores de cobertura/suma/argmax),
  `HealthResponse`, `ErrorResponse`. `tests/test_schemas.py` con 13 casos AAA.
- T009-3: `api.py` con `create_app(facade_factory)`, lifespan que carga el modelo una sola
  vez, `/health`, `/predict` (415/413/422/500/503, `ErrorResponse` uniforme), overlay
  reescalado al aspect ratio original (lado largo <= 512 px). `tests/test_api.py` con
  `TestClient` + fachada falsa, sin red ni modelo real.
- T009-4: `api_client.py` con `DiagnosisClient` (`httpx`), `ApiClientError` (mensajes en
  español), validación de respuestas contra los modelos Pydantic. `tests/test_api_client.py`
  con `httpx.MockTransport` + chequeo de subproceso sin `torch`/`transformers` en
  `sys.modules`.
- T009-5: `app.py` reescrito — sidebar con `st.file_uploader`/selector de muestras/estado de
  servicio (`/health`); tarjeta de resultados primero; imágenes lado a lado con aspect ratio
  consistente; leyenda de color; expander "¿Cómo interpretar el mapa de calor?"; disclaimer
  CDSS; copy en español neutro (sin voseo). Sin `torch`/`transformers`/`facade`.
  `tests/test_app.py` actualizado: smoke sin API real + chequeo estático de imports.
- T009-6: `docker/api.Dockerfile`, `docker/ui.Dockerfile` (`python:3.13-slim`, `uv` 0.12.9
  pinneado, instalación por grupo, usuario no-root, `HEALTHCHECK` con `urllib` stdlib),
  `docker-compose.yml` (volumen nombrado para caché de HF, `depends_on` con
  `service_healthy`, `data/samples` montado read-only), `.dockerignore`.
- T009-7: Job `docker` añadido a `.github/workflows/ci.yml` (`docker compose build`).
- T009-8: README actualizado (ejecución local en dos terminales, Docker Compose,
  arquitectura). Evidencia de verificación real a continuación.
- T009-9 (ronda de corrección, TDD estricto — RED confirmado antes de cada fix):
  1. **HIGH — carrera de datos en el modelo compartido**: `_diagnose_synchronized()` +
     un `threading.Lock` por instancia de `create_app()` serializan el acceso real al
     modelo dentro del hilo de `run_in_threadpool` (el event loop sigue sin bloquearse).
     Antes, dos requests concurrentes podían cruzar activaciones/gradientes en el
     forward-hook de `ViTGradCAM`. Test: 6 requests concurrentes contra una fachada que
     detecta solapamiento (`tracker["max"]`); RED reproducible (`6 == 1` falló) antes del
     fix, GREEN después.
  2. **HIGH — límite de tamaño post-parseo**: nuevo `UploadSizeLimitMiddleware` (ASGI
     puro) rechaza `POST /predict` con 413 antes de que Starlette bufferee/parsee el
     cuerpo multipart completo: (a) `Content-Length` declarado por encima de
     `MAX_UPLOAD_BYTES + UPLOAD_SIZE_MARGIN_BYTES` (64 KB de margen para overhead de
     multipart) se rechaza sin leer el cuerpo; (b) sin `Content-Length` (chunked), el
     middleware bufferea él mismo con tope y aborta apenas se supera el límite. Se
     mantiene el chequeo existente post-lectura sobre los bytes decodificados. Tests:
     uno para cada camino (Content-Length declarado, y un test a nivel ASGI puro con
     `asyncio.run` simulando chunks sin Content-Length).
  3. **MEDIUM — envelope único de error**: nuevos handlers para `RequestValidationError`
     (422) y `StarletteHTTPException` (incluye 404 de rutas desconocidas) devuelven
     `ErrorResponse`, igual que el resto de errores. Tests: `POST /predict` sin `file`
     -> 422 `ErrorResponse`; `GET` a una ruta inexistente -> 404 `ErrorResponse`.
  4. **LOW — bomba de descompresión**: `Image.DecompressionBombError` añadido al tuple de
     excepciones capturadas al decodificar la imagen (antes caía en el catch-all
     genérico y devolvía 500). Test con `monkeypatch` forzando la excepción -> 422
     `UndecodableImageError`.
  5. **LOW — fuga de detalles internos**: los handlers de `InferenceError`, `GradCAMError`
     y `UndecodableImageError` ahora devuelven un mensaje fijo en español al cliente y
     registran el mensaje real de la excepción de dominio vía `logger.warning(...)`
     server-side. El smoke real del padre había mostrado
     `"...cannot identify image file <_io.BytesIO object at 0x...>"` en `detail`; ya no
     ocurre. Tests actualizados para afirmar que el texto simulado/interno NO aparece en
     `detail`.
  6. **LOW — timeout configurable del cliente**: `app.py` añade
     `_resolve_api_timeout_seconds()` (lee `API_TIMEOUT_SECONDS`, valida float positivo,
     usa el default `60.0` si falta/es inválido/no positivo) y lo usa en
     `DiagnosisClient`. Documentado en `README.md` y fijado en `docker-compose.yml`
     (`ui.environment.API_TIMEOUT_SECONDS`).
  7. **UX — mensaje del sidebar duplicado**: el `st.error` de "servicio no disponible"
     ya no concatena el mensaje propio de `ApiClientError` (que ya decía "no está
     disponible") con un prefijo que repetía la misma idea; ahora es una sola oración
     fija en español neutro.
  8. **UX — calentamiento del modelo**: `_warm_up()` ejecuta una diagnosis descartable
     sobre una imagen RGB sintética de 224x224 justo después de cargar el modelo real
     en `_default_facade_factory()` (nunca en el camino de fachadas inyectadas de test);
     cualquier fallo de calentamiento se registra y se ignora, sin impedir el arranque.
     El smoke real del padre había medido ~4563 ms en el primer request vs ~1100 ms en
     los siguientes. Tests: `_warm_up` llama `diagnose()` una vez con una imagen
     pequeña; una fachada que falla en `_warm_up` no propaga la excepción; y
     `_default_facade_factory` (con `load_inference_service`/`DermatologyDiagnosticFacade`
     monkeypatcheados) invoca `_warm_up` con la fachada recién construida.
  - Efecto colateral necesario en `app.py`: el cuerpo de la UI (a partir de
    `st.set_page_config`) se movió detrás de `if __name__ == "__main__":` para poder
    probar `_resolve_api_timeout_seconds()` con `runpy.run_path(..., run_name=...)` sin
    ejecutar la interfaz completa ni intentar una conexión real. Streamlit ejecuta el
    script con `__name__ == "__main__"` tanto en `streamlit run` como en `AppTest`, así
    que el comportamiento en producción y en el smoke existente no cambia (verificado:
    el smoke `AppTest` original sigue en verde).

### Evidencia de verificación

- `uv lock` → `Resolved 92 packages` (sin errores).
- `uv lock --check` → limpio (`Resolved 92 packages`, sin drift).
- `uv sync` (grupos por defecto: dev+api+ui) → limpio, entorno sincronizado.
- `uv sync --dry-run --locked --no-dev --no-default-groups --group api` → confirma que
  instala solo el grupo `api` (torch/transformers/fastapi/uvicorn/...), desinstalando
  streamlit/dev-tools.
- `uv sync --dry-run --locked --no-dev --no-default-groups --group ui` → confirma que
  instala solo el grupo `ui` (streamlit/httpx/pydantic/pillow), desinstalando
  torch/transformers/fastapi.
- `uv tree --no-default-groups --group ui` → árbol de dependencias sin `torch` ni
  `transformers` (streamlit + httpx + pydantic + pillow únicamente).
- `uv run pytest -q -p no:cacheprovider --basetemp=<scratchpad>/pytest` →
  **86 passed, 2 deselected** (13 pruebas nuevas de la ronda de corrección; los 2
  deselected siguen siendo los tests `@pytest.mark.integration` existentes).
- `uv run ruff check .` → **All checks passed!**
- `uv run ruff format .` → sin cambios pendientes (32 files left unchanged).
- `uv run ruff format --check .` → **32 files already formatted**.
- `uv run mypy --strict src tests` → **Success: no issues found in 20 source files**.
- `uv run pytest -q -p no:cacheprovider --basetemp=<scratchpad>/pytest --cov=skin_lesion_classifier --cov-report=term`
  → **90% total coverage** (594 stmts, 58 miss). Por módulo nuevo: `api.py` 91%
  (creció en líneas por el middleware y el calentamiento; algunas ramas defensivas
  del middleware quedan sin ejercitar), `api_client.py` 85%, `schemas.py` 98%
  (todos ≥ 80%, objetivo cumplido).
- No se ejecutó `pytest -m integration` ni se cargó el modelo real (restricción de memoria
  de la máquina de desarrollo); queda a cargo de una verificación manual posterior.
- No se construyeron las imágenes Docker localmente (Docker no está instalado en esta
  máquina); la validación de build queda delegada al job `docker` de CI
  (`docker compose build`). Se validó manualmente la sintaxis YAML de
  `docker-compose.yml` y `.github/workflows/ci.yml` con `yaml.safe_load`.

### Desviaciones respecto al diseño original

- Se añadió `run_in_threadpool` (FastAPI/Starlette) alrededor de `facade.diagnose()` en
  `POST /predict`, no mencionado explícitamente en el diseño original. La inferencia es
  síncrona y ligada a CPU (Grad-CAM incluido); sin este cambio bloquearía el event loop de
  `asyncio` e impediría que `/health` u otras solicitudes concurrentes respondieran mientras
  se procesa un diagnóstico. Cambio aislado a `api.py`, sin tocar `facade.py`/`inference.py`.
- Se corrigieron dos constantes de FastAPI/Starlette deprecadas detectadas en warnings de
  test (`HTTP_413_REQUEST_ENTITY_TOO_LARGE` → `HTTP_413_CONTENT_TOO_LARGE`,
  `HTTP_422_UNPROCESSABLE_ENTITY` → `HTTP_422_UNPROCESSABLE_CONTENT`); mismo código HTTP,
  sin impacto en el contrato.
