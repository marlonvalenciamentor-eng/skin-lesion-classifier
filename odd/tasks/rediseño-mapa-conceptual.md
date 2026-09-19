# Rediseño profundo del mapa conceptual

## Objetivo

Compactar la cabecera, iniciar los paneles principales contraídos, reemplazar la teoría superficial por tres ramas lógicas de ingeniería de software y asegurar una impresión completa del documento.

## Problema

La cabecera consume espacio con tarjetas redundantes, los paneles cargan abiertos y el mapa conceptual no expresa las relaciones teóricas solicitadas. Además, la impresión debe expandir el contenido sin depender del estado de pantalla.

## Alcance autorizado

- `docs/Mapas_Mentales/Mapa_Mental_UI.html`
- Este documento de seguimiento ODD.

## Checklist

- [x] Compactar la cabecera e integrar ViT/Grad-CAM sin bloques blancos grandes.
- [x] Eliminar `open` de los dos `details` principales.
- [x] Rediseñar las tres ramas: retos del software, metodologías ágiles y pipeline CI/CD/MLOps.
- [x] Garantizar expansión total de `details` en `@media print`.
- [x] Verificar estructura, contenido y diff; crear commit solicitado.

## Criterios de aceptación

- La vista inicial no desperdicia espacio con tarjetas independientes de Modelo y Explicación.
- `Maqueta UI` y `Mapa Conceptual` aparecen contraídos al cargar.
- Las tres ramas incluyen literalmente los conceptos y relaciones indicados por el usuario.
- La regla de impresión muestra el contenido de cualquier `details` cerrado.
- El HTML sigue siendo autocontenido y adaptable.

## Progreso

- Estado: completado.
- Evidencia: `git diff --check`; parseo HTML; búsqueda de ramas, conceptos, paneles cerrados y regla de impresión.
- Commit: pendiente de creación con el mensaje solicitado.
