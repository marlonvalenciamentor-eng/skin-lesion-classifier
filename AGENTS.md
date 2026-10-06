# AGENTS.md — Reglas de revisión de código

Reglas que aplica la revisión automática (gga) y cualquier agente de IA que
trabaje en este repositorio. Resumen accionable de [`CONSTITUTION.md`](CONSTITUTION.md);
ante cualquier duda, manda la constitución.

## Bloqueantes (rechazar el commit)

1. **Desacoplamiento UI ↔ modelo:** `app.py` no importa `torch`, `transformers`
   ni `safetensors`. Consume la fachada `DermatologyDiagnosticFacade`, que es el
   único componente que toca el modelo.
2. **Sin pesos ni datasets en git:** nada de `.safetensors`, `.pt`, `.bin`,
   imágenes clínicas masivas ni rutas `models/` o `data/` versionadas. Se
   descargan bajo demanda desde Hugging Face Hub.
3. **Sin notebooks:** prohibidos `.ipynb` y carpetas `notebooks/`. Todo es
   código modular `.py` bajo `src/`.
4. **Sin cajones de sastre:** prohibidos `utils.py`, `helpers.py`, `misc.py`.
   Cada módulo tiene una única responsabilidad.
5. **Solo `uv`:** no se usa `pip` suelto; toda dependencia queda en
   `pyproject.toml` y `uv.lock`.
6. **Sin secretos:** ni tokens, ni claves, ni credenciales en el código.

## Arquitectura

- La fachada `DermatologyDiagnosticFacade` orquesta preprocesamiento,
  inferencia y Grad-CAM; los demás componentes no se llaman entre sí saltándola.
- Inyección de dependencias: los servicios reciben modelo y procesador por
  parámetro; no instancian dependencias pesadas en el constructor.
- Salidas estructuradas como `@dataclass(frozen=True)`.
- Errores esperables (red, archivo corrupto, dimensiones inválidas) se envuelven
  en excepciones de dominio (`ModelLoadingError`, `InferenceError`,
  `GradCAMError`, …) con mensaje claro en español y `raise ... from err`.
- Los hooks de PyTorch se liberan siempre en un bloque `finally`.

## Tipado y estilo

- Type hints completos en todas las funciones; el código debe pasar
  `uv run mypy --strict src tests`.
- `ruff check` y `ruff format` limpios; línea máxima de 100 caracteres.
- Nada de `Any` innecesario ni `# type: ignore` sin justificación.

## Pruebas

- Toda lógica nueva trae pruebas en `tests/`, con estructura AAA
  (Arrange, Act, Assert) visible.
- Las pruebas unitarias no descargan modelos ni usan red: usan mocks o dobles.
- Las que usan pesos reales llevan `@pytest.mark.integration`; la latencia de
  inferencia en CPU se valida contra < 3.0 s.

## Commits

- Conventional Commits: `tipo(alcance): descripción en imperativo`.
  Tipos: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
