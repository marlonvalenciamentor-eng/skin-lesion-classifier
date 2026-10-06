# =============================================================================
# Makefile — SkinLesionClassifier
# Atajos para las tareas comunes de desarrollo (uv + Streamlit + pytest + ruff).
# Uso: make <target>   (ej. `make test`, `make run`, `make check`)
# =============================================================================

.PHONY: help run test integration lint format format-check typecheck check docker evaluate clean

help: ## Lista los targets disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

run: ## Levanta la app (Streamlit)
	uv run streamlit run app.py

test: ## Pruebas unitarias (sin red, sin descargar modelo)
	uv run pytest -v

integration: ## Pruebas de integración (descarga el modelo real)
	uv run pytest -m integration -v

lint: ## Linter estático (ruff)
	uv run ruff check .

format: ## Formatea el código (ruff)
	uv run ruff format .

format-check: ## Verifica el formato sin modificar
	uv run ruff format --check .

typecheck: ## Tipado estático (mypy --strict)
	uv run mypy --strict src tests

check: lint format-check typecheck test ## Todas las verificaciones de calidad

docker: ## Levanta la app con Docker
	docker compose up --build

evaluate: ## Evalúa el modelo (≈6 min, descarga ~700 MB)
	uv run python -m skin_lesion_classifier.evaluation

clean: ## Limpia cachés de Python, pytest, mypy y ruff
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
