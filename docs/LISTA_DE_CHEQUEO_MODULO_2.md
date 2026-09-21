# LISTA DE CHEQUEO OFICIAL — ENTREGA MÓDULO 2 (35%)
**Programa:** Especialización en Inteligencia Artificial — Universidad Autónoma de Occidente (UAO)  
**Curso:** Desarrollo de Soluciones de Inteligencia Artificial II  
**Docente:** Jan Polanco Velasco  
**Equipo de Trabajo:**
- Marlon Valencia Velosa — Código Estudiantil: `1113531444`
- Miguel Ángel Ortiz Roldán — Código Estudiantil: `6200485`  
**Fecha de Cierre:** 22 de Septiembre de 2026

---

## 📌 1. Entregables Obligatorios del Módulo 2

| # | Entregable Exigido en Rúbrica | Criterio de Cumplimiento | Archivo / Evidencia en Repositorio | Estado |
|---|-------------------------------|--------------------------|-----------------------------------|:------:|
| 1 | **Mapa Mental 1: Arquitectura y UI** | Jerarquía visual clara, estación Streamlit, patrón Fachada, modelo ViT y XAI Grad-CAM. | [`docs/Mapa Mental Entrega 1 - Desarrollo de Software y Metodología Ágil.html`](./Mapa%20Mental%20Entrega%201%20-%20Desarrollo%20de%20Software%20y%20Metodología%20Ágil.html)<br>[`docs/Mapa Mental Entrega 1 - Desarrollo de Software y Metodología Ágil.pdf`](./Mapa%20Mental%20Entrega%201%20-%20Desarrollo%20de%20Software%20y%20Metodología%20Ágil.pdf) | ✅ **CUMPLIDO** |
| 2 | **Mapa Mental 2: Pipeline y Despliegue** | Relación del flujo de datos vs despliegue operativo de software, prevención y mitigación de bugs. | [`docs/Mapa Mental Entrega 2 - Pipeline de Despliegue y Detección de Bugs.html`](./Mapa%20Mental%20Entrega%202%20-%20Pipeline%20de%20Despliegue%20y%20Detección%20de%20Bugs.html)<br>[`docs/Mapa Mental Entrega 2 - Pipeline de Despliegue y Detección de Bugs.pdf`](./Mapa%20Mental%20Entrega%202%20-%20Pipeline%20de%20Despliegue%20y%20Detección%20de%20Bugs.pdf) | ✅ **CUMPLIDO** |
| 3 | **Documento Formal de Propuesta (PDF)** | 6 capítulos obligatorios, formato APA/IEEE, Model Card y restricciones técnicas. | [`docs/propuesta.md`](./propuesta.md)<br>[`docs/Propuesta_Proyecto.html`](./Propuesta_Proyecto.html)<br>[`docs/Propuesta_Proyecto.pdf`](./Propuesta_Proyecto.pdf) | ✅ **CUMPLIDO** |
| 4 | **Repositorio GitHub Público** | Estructura canónica de proyecto, commits convencionales, `pyproject.toml` y `uv.lock`. | Repositorio GitHub con rama `docs/mind-maps` y `main` configuradas. | ✅ **CUMPLIDO** |
| 5 | **Tablero Kanban en GitHub Projects** | Tablero público con columnas `To Do`, `In Progress`, `Review`, `Done` y estimación Fibonacci. | Configuración en GitHub Projects vinculada a los tickets del cronograma. | ✅ **CUMPLIDO** |

---

## 🛠️ 2. Cumplimiento de Reglas de Ingeniería y Constitución del Proyecto

| Regla de la Clase (`Clase 2.md`) | Estándar Exigido | Implementación en este Proyecto | Estado |
|-----------------------------------|------------------|--------------------------------|:------:|
| **Gestión de Entorno** | **Prohibido `pip`**: Uso exclusivo de `uv` con Python 3.13. | Entorno gestionado 100% con `uv`. `uv sync` y `uv.lock` determinista en el repositorio. | ✅ **CUMPLIDO** |
| **Pruebas Unitarias** | Patrón **AAA (Arrange, Act, Assert)** en `pytest`. | Pruebas en `tests/` que validan dimensiones del tensor `(1, 3, 224, 224)`, rango `[0.0, 1.0]` y tipo `float32`. | ✅ **CUMPLIDO** |
| **Auditoría de Código y Linter** | Estándar PEP 8 auditado con `ruff`. | Configurado en `pyproject.toml` ejecutable vía `uv run ruff check .`. | ✅ **CUMPLIDO** |
| **Arquitectura de Software** | Alta cohesión y bajo acoplamiento. Prohibido `utils.py`. | Patrón Fachada centralizado en `DermatologyDiagnosticFacade`. Módulos con responsabilidad única. | ✅ **CUMPLIDO** |
| **De Vibe Coding a SDD** | Flujo `Explore → Propose → Spec → Implement & Verify`. | `CONSTITUTION.md` define principios arquitectónicos, contratos de interfaz y límites del sistema. | ✅ **CUMPLIDO** |
| **MaaS (Model-as-a-Service)** | Modelo público con Model Card y licencia permisiva. | Modelo seleccionado: `Anwarkh1/Skin_Cancer-Image_Classification` (Vision Transformer, Apache-2.0). | ✅ **CUMPLIDO** |
| **Explicabilidad Médica (XAI)** | Eliminar sesgo de caja negra clínica con Grad-CAM. | Extracción de activaciones de `vit.encoder.layer[-1]` superpuestas a la lesión original. | ✅ **CUMPLIDO** |

---

## 📝 3. Formato del Documento PDF de la Propuesta

- [x] **Tipografía:** Arial / Calibri / Times New Roman.
- [x] **Tamaño de Letra:** 12 pt para cuerpo de texto, 14 pt en negrita para títulos principales.
- [x] **Interlineado:** 1.5 líneas en todo el documento.
- [x] **Márgenes:** 2.5 cm por los cuatro lados (superior, inferior, izquierdo, derecho).
- [x] **Extensión:** Menor a 15 páginas (sin anexos).
- [x] **Portada formal con datos completos:** Nombre del proyecto, universidad, programa, integrantes con códigos y docente.

---

## 📚 4. Capítulos Obligatorios de la Propuesta

- [x] **Capítulo 1: Contexto del Proyecto**
  - Descripción del modelo base (`Anwarkh1/Skin_Cancer-Image_Classification` en Hugging Face).
  - Model Card completa (métricas reportadas: 96.95% accuracy, datos de entrenamiento HAM10000, sesgos Fitzpatrick y limitaciones éticas).
  - Restricciones técnicas y justificación de impacto clínico.
- [x] **Capítulo 2: Objetivos y Alcance**
  - Objetivo General claro y medible.
  - 3 Objetivos Específicos técnicos.
  - Matriz de Alcance: *Included*, *Nice to have*, *Not included*.
- [x] **Capítulo 3: Cronograma y Ruta Crítica**
  - Cronograma de 15 días con fases de Preparación, Desarrollo e Implementación.
  - Método de Ruta Crítica (CPM) con holguras e hitos clave.
- [x] **Capítulo 4: Marco de Investigación y CRISP-DM**
  - Mapeo integral de las 6 fases de la metodología CRISP-DM al flujo de trabajo del clasificador.
  - Descripción de datasets, técnicas de Vision Transformers y stack tecnológico (`torch`, `transformers`, `streamlit`).
- [x] **Capítulo 5: Gestión de Tareas (Kanban)**
  - Tablero en GitHub Projects con columnas canónicas (`To Do`, `In Progress`, `Review`, `Done`).
  - Anatomía de tickets con Acceptance Criteria (AC), responsables y estimación Fibonacci 1:1.
- [x] **Capítulo 6: Anexos y Repositorios**
  - Enlaces al repositorio base de GitHub y tablero Kanban.
  - Referencias bibliográficas en formato IEEE/APA.
