# Interactive Mind Maps

## Objective
Create two self-contained, print-ready HTML mind maps for the `skin-lesion-classifier` project.

## Problem
The project has PDF mind-map deliverables but lacks editable, browser-based visual documentation for the Streamlit UI and the CRISP-DM diagnostic journey.

## Why
Provide polished artifacts that can be opened in Chrome and printed to landscape PDF without external runtime dependencies.

## Scope
- `docs/Mapas_Mentales/Mapa_Mental_UI.html`
- `docs/Mapas_Mentales/Mapa_Mental_Journey.html`

## Constraints
- Native HTML/CSS with inline SVG where useful; no heavy external libraries.
- English technical identifiers must remain exact; explanatory copy follows the project/request language.
- A4/Letter landscape print layout with no clipped content.
- Include project title and both authors in each artifact.

## Checklist
- [x] MM-01 Generate the hybrid UI map with Streamlit nodes and CSS mockup.
- [x] MM-02 Generate the CRISP-DM/user journey map with the six requested stages.
- [x] MM-03 Validate files, print CSS, links, and repository status.

## Acceptance criteria
- Both HTML files open standalone in Chrome.
- UI map visibly connects the Sidebar node to the mockup sidebar.
- Journey map includes Input, Facade, preprocessing, ViT inference, Grad-CAM, and DiagnosticResult output.
- Print media produces a clean landscape page with no deliberate clipping or hidden overflow.

## Checks
- Parse/inspect HTML structure and required labels.
- Run repository tests if applicable; no source-code behavior is changed.

## Progress
Completed. HTML parsing, required-label assertions, print CSS assertions, `git diff --check`, and `uv run pytest -q` passed. Browser visual inspection was not performed.

## Next step
Artifacts generated and structurally inspected. Work-unit commit pending.
