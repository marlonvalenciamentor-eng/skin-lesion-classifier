# Dashboard unificado de mapas mentales

## Objetivo
Transformar `Mapa_Mental_UI.html` en un dashboard documental que presente primero la propuesta y permita consultar por separado la maqueta UI y el mapa conceptual de ingeniería ML-IA.

## Alcance autorizado
- `docs/Mapas_Mentales/Mapa_Mental_UI.html`
- `odd/tasks/dashboard-unificado.md`

## Tareas
- [x] DU-01: agregar la Primera Vista con título, integrantes y datos clave de ViT y Grad-CAM.
- [x] DU-02: encapsular la maqueta interactiva existente en el panel colapsable “Maqueta UI”.
- [x] DU-03: crear el panel colapsable “Mapa Conceptual” con el diagrama HTML/CSS de la rúbrica.
- [x] DU-04: actualizar estilos responsive y `@media print` para desplegar todas las secciones sin páginas en blanco.
- [x] DU-05: verificar estructura, interacción, impresión y diff; crear el commit solicitado.

## Criterios de aceptación
- La primera vista comunica título, Ortiz y Valencia, ViT y Grad-CAM.
- La maqueta UI existente permanece funcional dentro de un panel colapsable.
- El mapa conceptual conecta Retos del Software, ciclo de vida ML-IA, Agile, DevOps, MLOps, CI/CD y Pipeline.
- Las ramas del diagrama tienen colores diferenciados y se construyen solo con HTML/CSS.
- En impresión todos los paneles están desplegados, el contenido no queda oculto por `hidden` y no aparecen páginas en blanco.
- El documento conserva funcionamiento sin dependencias externas y todo el texto agregado está en español.

## Verificación
- `python3` con `html.parser`: PASS; confirmó 2 elementos `details`, 19 etiquetas requeridas, estructura balanceada y reglas de impresión para `details`, `[hidden]` y paneles.
- `git diff --check`: PASS, sin errores de espacios.
- Revisión del diff: PASS; solo se modificaron los dos archivos autorizados y se conservaron el JavaScript interactivo y la hoja detallada imprimible.
- Interacción y PDF: verificación estática; el HTML conserva los listeners existentes y fuerza la visibilidad de paneles y estados ocultos en `@media print`.

## Progreso
- Estado: completado.
- Evidencia de entrega: commit creado con el mensaje exacto `docs: unificar mapas en dashboard`.
