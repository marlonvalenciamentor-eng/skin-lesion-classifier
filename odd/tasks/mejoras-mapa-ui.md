# Mejoras del mapa mental de la interfaz

## Objetivo
Mejorar el mapa mental HTML de la interfaz con tooltips técnicos interactivos en pantalla y una segunda hoja explicativa completa para impresión.

## Alcance autorizado
- `docs/Mapas_Mentales/Mapa_Mental_UI.html`
- `odd/tasks/mejoras-mapa-ui.md`

## Tareas
- [x] UI-01: agregar detalles interactivos al pasar el cursor sobre los componentes de la maqueta.
- [x] UI-02: agregar una segunda página imprimible con la explicación textual de cada componente.
- [x] UI-03: mostrar tooltips flotantes con desglose de tareas y puntos/días Fibonacci al pasar sobre las zonas de la maqueta.
- [x] UI-04: mantener la segunda hoja oculta en pantalla y mostrarla completa, sin tooltips flotantes, al imprimir.

## Criterios de aceptación
- La maqueta muestra tooltips flotantes con tareas y tiempos Fibonacci al pasar el ratón o enfocarse con teclado.
- La segunda hoja comienza con `page-break-before: always` dentro de `@media print`.
- La segunda hoja usa `display:none` en pantalla y `display:block` en impresión.
- Los tooltips se ocultan en `@media print` y el desglose técnico permanece visible en la hoja 2.
- Todo el texto agregado está en español.
- El HTML conserva su funcionamiento sin dependencias externas.

## Verificación
- Validar estructura HTML y reglas CSS con lectura del archivo.
- Comprobar el diff y el estado de Git antes del commit.

## Progreso
- Estado: completado.
- Evidencia: el HTML incluye seis tooltips flotantes con eventos de ratón y foco, subtareas puntuadas con Fibonacci, y una sección `.detail-sheet` con `display:none` en pantalla y `display:block` más `page-break-before:always` en impresión. Los tooltips y estados visuales se ocultan al imprimir.
- Verificación: `HTMLParser`, comprobaciones estructurales de visibilidad/tooltips/Fibonacci y `git diff --check` exitosos.
