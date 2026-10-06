# Skin Lesion Classifier

Skin lesion classification from dermoscopic images using a pretrained model,
with Grad-CAM visual explanations and a Streamlit web UI.

Course project — *Desarrollo de Proyectos de IA*, Universidad Autónoma de Occidente (2026-5B).

## Architecture

The system runs as a single service: the Streamlit UI orchestrates everything
through the `DermatologyDiagnosticFacade`, which owns the model.

```
Streamlit UI  -->  DermatologyDiagnosticFacade  -->  Model (ViT + Grad-CAM)
(no torch/transformers)   (preprocessing + inference + Grad-CAM)
```

- **Facade** (`skin_lesion_classifier.facade`) orchestrates preprocessing,
  inference and Grad-CAM, exposing a single `diagnose()` method.
- **Web UI** (`app.py`) never imports `torch` or `transformers` directly: it
  consumes the facade, which is the only component that touches the model.

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

`uv sync` installs everything (runtime + dev) by default, so a single command
still sets up the full development environment.

## Run locally

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501. The first diagnosis loads the model (cached
afterwards for subsequent requests).

## Run with Docker

```bash
docker compose up --build
```

- UI: http://localhost:8501

A single image runs the Streamlit UI together with the model. Weights are
cached in a named Docker volume so they download once, on first run, instead
of being baked into the image.

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

Train IDs are read first, so a network failure aborts before any scenario runs.
`metrics.json` is rewritten atomically after each scenario with `status: partial` and
`completed_scenarios`, and becomes `status: complete` at the end; an optional Grad-CAM
failure is recorded under `gradcam` without discarding the results.

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
                               # facade, evaluation (metrics, dataset loader, perturbations, CLI)
app.py                        # Streamlit UI (facade only, no torch/transformers)
docker/Dockerfile             # single image: Streamlit UI + model
docker-compose.yml            # single service, named HF cache volume
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
