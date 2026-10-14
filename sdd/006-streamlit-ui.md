# Ticket #006 — Streamlit Web Interface

| Field | Value |
|---|---|
| Issue | #9 (closed) |
| Owner | Marlon Valencia Velosa |
| Estimate | 5 SP · due 2026-09-29 |
| Source spec | [`docs/propuesta.md`](../docs/propuesta.md) · detailed tasks: [`odd/tasks/ticket-006-streamlit-dashboard.md`](../odd/tasks/ticket-006-streamlit-dashboard.md) |

## Spec
Build the UI with `st.file_uploader`, a side-by-side view (original vs. Grad-CAM) and percentage
probability bars.

## Acceptance criteria
- [x] Clean design without rendering failures.
- [x] Friendly error handling when the image is not valid.

## Implementation
PR #16, integrated in PR #17; improved in PR #26 (three-column layout, patient name, downloadable
PDF report via [`report.py`](../src/skin_lesion_classifier/report.py)).

## Verification
`tests/test_app.py`; end-to-end demo run on 2026-10-06 (diagnosis, Grad-CAM and PDF download).
