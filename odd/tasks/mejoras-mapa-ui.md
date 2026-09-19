# Mejoras del mapa mental de la interfaz

## Objetivo
Completar la navegación del mapa mental HTML con estado explícito, controles nativos y documentación de la integración entre Streamlit, la fachada, ViT y Grad-CAM, manteniendo la segunda hoja explicativa para impresión.

## Alcance autorizado
- `docs/Mapas_Mentales/Mapa_Mental_UI.html`
- `odd/tasks/mejoras-mapa-ui.md`

## Tareas
- [x] UI-01: agregar detalles interactivos al pasar el cursor sobre los componentes de la maqueta.
- [x] UI-02: agregar una segunda página imprimible con la explicación textual de cada componente.
- [x] UI-03: retirar tooltips flotantes y conectores SVG/CSS del mapa en pantalla.
- [x] UI-04: convertir las cajas izquierdas en pestañas accesibles con resaltado hover/foco de la zona correspondiente.
- [x] UI-05: mostrar una vista detallada del componente seleccionado dentro del panel derecho mediante clic.
- [x] UI-06: mantener la segunda hoja oculta en pantalla y visible con el desglose completo al imprimir.
- [x] UI-07: añadir estado y control accesible para volver a la imagen inicial.
- [x] UI-08: añadir botón nativo visible para imprimir a PDF.
- [x] UI-09: eliminar toda mención a puntos, días y estimaciones Fibonacci.
- [x] UI-10: documentar la integración técnica de cada capa con Streamlit, Facade, ViT y Grad-CAM.

## Criterios de aceptación
- No existen conectores punteados ni tooltips flotantes en la vista web.
- El hover o foco de cada caja izquierda resalta únicamente su zona correspondiente en la maqueta.
- El clic en cada caja reemplaza el contenido del panel derecho por una vista detallada de esa sección.
- La segunda hoja comienza con `page-break-before: always` dentro de `@media print`.
- La segunda hoja usa `display:none` en pantalla y `display:block` en impresión.
- La maqueta interactiva se conserva como referencia y el desglose técnico permanece visible en la hoja 2 al imprimir.
- El botón de volver restaura la maqueta, el resaltado inicial y el foco de forma accesible.
- El botón de impresión invoca la impresión nativa y permanece oculto en `@media print`.
- No quedan menciones a puntos, días ni Fibonacci en el HTML.
- Cada capa documenta su integración técnica sin simular dependencias directas desde el HTML.
- Todo el texto agregado está en español.
- El HTML conserva su funcionamiento sin dependencias externas.

## Verificación
- Validar estructura HTML y reglas CSS con lectura del archivo.
- Comprobar el diff y el estado de Git antes del commit.

## Progreso
- Estado: completado.
- Evidencia: las cajas izquierdas usan `role="tab"`, resaltan su zona mediante `data-target` y reemplazan el contenido de `#panel-detalle` con JavaScript basado en `textContent`.
- Evidencia de impresión: `.detail-sheet` conserva `display:none` en pantalla y `display:block` más `page-break-before:always` en impresión.
- Verificación: `HTMLParser`, comprobaciones de ausencia de tooltips/conectores, conteo de pestañas, sintaxis JavaScript y `git diff --check` exitosos.
