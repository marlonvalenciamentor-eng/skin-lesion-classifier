# Mejoras del mapa mental de la interfaz

## Objetivo
Rediseñar la interacción del mapa mental HTML como una navegación tabulada, sin conectores visuales ni tooltips flotantes, manteniendo la segunda hoja explicativa completa para impresión.

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

## Criterios de aceptación
- No existen conectores punteados ni tooltips flotantes en la vista web.
- El hover o foco de cada caja izquierda resalta únicamente su zona correspondiente en la maqueta.
- El clic en cada caja reemplaza el contenido del panel derecho por una vista detallada de esa sección.
- La segunda hoja comienza con `page-break-before: always` dentro de `@media print`.
- La segunda hoja usa `display:none` en pantalla y `display:block` en impresión.
- La maqueta interactiva se conserva como referencia y el desglose técnico permanece visible en la hoja 2 al imprimir.
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
