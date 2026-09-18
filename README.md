# Skin Lesion Classifier

Skin lesion classification from dermoscopic images using a pretrained model,
with Grad-CAM visual explanations and a Streamlit interface.

Course project — *Desarrollo de Proyectos de IA*, Universidad Autónoma de Occidente (2026-5B).

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

## Run tests

```bash
uv run pytest -v
```

## Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

## Project layout

```
src/skin_lesion_classifier/   # package: image loading, preprocessing, model, Grad-CAM
tests/                        # pytest suite
docs/propuesta.md             # formal project proposal (M2 deliverable)
.github/                      # PR template and CI pipeline
```

## Branching model

- `main` — stable, protected. Only merged through pull requests.
- `feature/<short-name>` — one branch per Kanban ticket.

## Links

- Kanban board (GitHub Projects): _pending_
- Model card: _pending_
