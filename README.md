# Skin Lesion Classifier

Skin lesion classification from dermoscopic images using a pretrained model,
with Grad-CAM visual explanations, a FastAPI inference service, and a
Streamlit web UI.

Course project — *Desarrollo de Proyectos de IA*, Universidad Autónoma de Occidente (2026-5B).

## Architecture

The system is split into two independently deployable services:

```
Streamlit UI  --HTTP/JSON (Pydantic contracts)-->  FastAPI Inference API  --> Facade --> Model
(no torch/transformers)                            (owns the ViT + Grad-CAM)
```

- **Inference API** (`skin_lesion_classifier.api`) owns the model: it loads the
  ViT once at startup, runs the diagnostic facade, and returns a validated
  `DiagnosisResponse` (Pydantic v2) over HTTP.
- **Web UI** (`app.py`) never imports `torch`, `transformers`, or the facade.
  It talks to the API through `skin_lesion_classifier.api_client`, validating
  every response against the same Pydantic contracts (`schemas.py`).

## Team

| Name | ID |
|------|----|
| Miguel Angel Ortiz Roldan | 6200485 |
| Marlon Valencia Velosa | 1113531444 |

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.13 automatically)

## Setup

```bash
git clone <repo-url>
cd skin-lesion-classifier
uv sync
```

`uv sync` installs everything (dev + api + ui groups) by default, so a single
command still sets up the full development environment.

## Run locally (two terminals)

```bash
# Terminal 1: inference API (owns the model)
uv run uvicorn skin_lesion_classifier.api:app --port 8000

# Terminal 2: web UI (talks to the API over HTTP)
uv run streamlit run app.py
```

Start the API first, then open the Streamlit app at http://localhost:8501.

The UI is configured via two environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_URL` | `http://localhost:8000` | Base URL of the inference API. |
| `API_TIMEOUT_SECONDS` | `60.0` | HTTP client timeout (seconds). Falls back to the default if unset, not a number, or not strictly positive. |

## Run with Docker

```bash
docker compose up --build
```

- API: http://localhost:8000 (docs at http://localhost:8000/docs)
- UI: http://localhost:8501

The API image only installs the `api` dependency group (torch, transformers,
FastAPI); the UI image only installs the `ui` group (Streamlit, httpx) and
never pulls in torch or transformers. Model weights are cached in a named
Docker volume so they download once, on first run, instead of being baked
into the image.

## Run tests

```bash
uv run pytest -v                 # unit tests (no model download)
uv run pytest -v -m integration  # downloads the real model from Hugging Face
```

## Evaluate the model

```bash
uv run python -m skin_lesion_classifier.evaluation
```

Evaluates the pinned model revision on the pinned `test` split of
[`marmal88/skin_cancer`](https://huggingface.co/datasets/marmal88/skin_cancer) (HAM10000).
It downloads about 700 MB (model + split) into the Hugging Face cache, never into the
repository, and runs in about 6 minutes on CPU. Results go to `reports/evaluation/`
(git-ignored): `metrics.json`, `confusion_matrix.png` and sample Grad-CAM overlays.
Use `--limit N` for a quick run, `--skip-overlap` to skip the train-overlap check, and
`--help` for all options.

The committed results and their interpretation, including the train/test overlap
limitation, are in [`docs/REPORTE_VALIDACION_MODELO.md`](docs/REPORTE_VALIDACION_MODELO.md).

## Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

## Project layout

```
src/skin_lesion_classifier/   # package: image loading, preprocessing, model, Grad-CAM,
                               # facade, Pydantic schemas, FastAPI app, HTTP client,
                               # evaluation (metrics, dataset loader, perturbations, CLI)
app.py                        # Streamlit UI (HTTP client only, no torch/transformers)
docker/                       # api.Dockerfile, ui.Dockerfile
docker-compose.yml            # api + ui services, named HF cache volume
tests/                        # pytest suite
odd/tasks/                    # SDD ticket specs (spec-driven development)
docs/propuesta.md             # formal project proposal (M2 deliverable, PDF in docs/)
CONSTITUTION.md               # architecture principles and dependency rules
docs/                         # mind maps and other deliverables
.github/                      # PR template and CI pipeline
```

## Branching model

- `main` — stable, protected. Only merged through pull requests.
- `feature/<short-name>` — one branch per Kanban ticket.

## Links

- [Kanban board (GitHub Projects)](https://github.com/users/miguelortizR/projects/2)
- [Model card: Anwarkh1/Skin_Cancer-Image_Classification](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)
- [Project proposal](docs/propuesta.md) ([PDF](docs/Propuesta_Proyecto.pdf))
- [Architecture constitution](CONSTITUTION.md)
