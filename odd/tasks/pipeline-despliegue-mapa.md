# Pipeline de despliegue y detección de bugs

## Objetivo
Reestructurar `docs/Mapas_Mentales/Mapa_Mental_Journey.html` como el mapa mental del entregable 2, conectando el pipeline de datos del clasificador dermatológico con CI/CD, pruebas, empaquetado y entrega continua en Streamlit.

## Alcance autorizado
- Mantener el artefacto autocontenido y modificar únicamente el HTML objetivo.
- Replicar la estructura visual e imprimible de `Mapa_Mental_UI.html`.
- Incorporar Hero + Journey interactivo en la hoja 1, desglose de ocho artículos en la hoja 2 y mapa conceptual de cuatro ramas en la hoja 3.
- Redactar todo en positivo, con contenido técnico específico del proyecto.

## Tareas
- [x] PJ-01 Construir la hoja 1 con hero, integrantes, impresión PDF y journey interactivo.
- [x] PJ-02 Construir la hoja 2 con ocho artículos técnicos del pipeline y despliegue.
- [x] PJ-03 Construir la hoja 3 con cuatro ramas conceptuales y etapas Build/Test/Deploy/Monitor.
- [x] PJ-04 Implementar interacción dinámica, impresión A4 horizontal y verificación estructural.

## Progreso y evidencia
- `docs/Mapas_Mentales/Mapa_Mental_Journey.html` reestructurado en 218 líneas autocontenidas.
- Verificado: 8 `.detail-item`, 4 `.concept-branch`, 6 contenidos JS, selectores de impresión A4 y `git diff --check` correcto.
- Cambios fuente limitados al HTML objetivo; no se creó commit.

## Criterios de aceptación
- Las tres hojas imprimen en A4 landscape con margen CSS de `5mm 10mm` y saltos limpios.
- Cada fase actualiza vista detallada con código, herramientas y mitigación de bugs.
- El contenido cubre ViT, Grad-CAM, `DermatologyDiagnosticFacade`, Dave Farley, SDD, AAA, PyTest y `uv`.
- La redacción utiliza formulaciones afirmativas.

## Checks
- Lectura estructural del HTML y búsqueda de requisitos obligatorios.
- Validación de scripts embebidos mediante parseo y revisión de selectores.
