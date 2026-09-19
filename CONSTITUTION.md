# CONSTITUCIÓN TÉCNICA Y ARQUITECTÓNICA DEL PROYECTO (CONSTITUTION.md)

**Institución:** Universidad Autónoma de Occidente (UAO) — Facultad de Ingeniería  
**Programa:** Especialización en Inteligencia Artificial  
**Asignatura:** Desarrollo de Soluciones de Inteligencia Artificial II — Módulo 2 (35%)  
**Docente:** Jan Polanco Velasco  
**Equipo de Trabajo:**
- Marlon Valencia Velosa — Código Estudiantil: `1113531444`
- Miguel Ángel Ortiz Roldán — Código Estudiantil: `6200485`  
**Proyecto:** *SkinLesionClassifier — Sistema Asistivo para la Detección Temprana de Cáncer de Piel mediante Vision Transformers (ViT) y Explicabilidad Visual (Grad-CAM)*

---

## ⚖️ 1. MANDATOS Y PROHIBICIONES ESTRICTAS DE LA ASIGNATURA (`Reglas_Clase.md`)

Todo código desarrollado, revisado o fusionado en este repositorio debe respetar sin excepción las siguientes directrices institucionales:

1. 🚫 **PROHIBIDO EL "VIBE CODING":**
   - Queda vetado el código generado sin especificaciones formales, sin pruebas automáticas o basado en suposiciones.
   - Todo desarrollo debe regirse por el flujo **Spec-Driven Development (SDD)**: `Explore → Propose → Spec → Implement & Verify`.

2. 🚫 **PROHIBIDO JUPYTER NOTEBOOKS, GOOGLE COLAB Y KAGGLE NOTEBOOKS:**
   - Prohibido el uso de archivos `.ipynb`, Google Colab o carpetas `notebooks/`.
   - El 100% de la solución debe ser ingeniería de software modular en scripts `.py` bajo la carpeta `src/`.

3. 🚫 **PROHIBIDO `pip` SUELTO (USO EXCLUSIVO DE `uv`):**
   - El entorno virtual, la resolución de dependencias y el empaquetado deben gestionarse únicamente con **`uv`** bajo **Python 3.13**.
   - Toda instalación se registra en `pyproject.toml` y el archivo de bloqueo determinista `uv.lock`.

4. 🚫 **PROHIBIDO `utils.py`, `helpers.py`, `misc.py` O CAJONES DE SASTRE:**
   - Cada archivo debe tener un propósito semántico claro, alta cohesión y una única responsabilidad (*Single Responsibility Principle - SRP*).

5. 🚫 **PROHIBIDO SUBIR MODELOS PESADOS O DATASETS MASIVOS A GIT:**
   - Los binarios (`.safetensors`, `.pt`, `.bin`) y datasets clínicos deben estar ignorados en `.gitignore` (`models/`, `data/`) y descargarse bajo demanda o consumirse vía MaaS (Hugging Face Hub).

---

## 🏛️ 2. PRINCIPIOS ARQUITECTÓNICOS (CLEAN ARCHITECTURE & SOLID)

1. **Patrón Fachada (`DermatologyDiagnosticFacade`):**
   - El módulo orquestador centraliza las llamadas entre la lectura de imágenes, preprocesamiento, inferencia y explicabilidad Grad-CAM.
   - **Desacoplamiento Estricto:** La capa de presentación (interfaz web en Streamlit) **nunca** debe importar librerías de bajo nivel de Deep Learning (`torch`, `transformers`, `safetensors`) ni manipular tensores directamente. Solo interactúa con la Fachada mediante objetos de transferencia de datos (*DTOs*).

2. **Inyección de Dependencias:**
   - Los componentes no instancian dependencias rígidas en sus constructores. Por ejemplo, `InferenceService` recibe el modelo y el procesador inyectados, permitiendo pruebas unitarias ultrarrápidas con modelos sintéticos (*mocks*).

3. **Inmutabilidad y Tipado Estático:**
   - Las salidas estructuradas se definen mediante `@dataclass(frozen=True)` (ejemplo: `Prediction`).
   - Todo el código debe contar con **Type Hints** completos (PEP 484) y verificables con analizadores estáticos.

4. **Manejo Defensivo de Excepciones:**
   - Errores de red, archivos corruptos o dimensiones incompatibles no deben generar fallos no controlados. Se definen excepciones de dominio semánticas (`ModelLoadingError`, `InferenceError`) con mensajes claros y trazables en español.

---

## 🧪 3. ESTÁNDAR DE TESTING Y CALIDAD DE SOFTWARE

1. **Patrón AAA (Arrange, Act, Assert):**
   - Toda prueba unitaria en `tests/` debe estructurarse obligatoriamente bajo las tres fases:
     * **Arrange:** Preparación de datos sintéticos, configuración o mocks en memoria.
     * **Act:** Ejecución del método o función a evaluar.
     * **Assert:** Verificación unívoca de tipos, valores y excepciones esperadas.

2. **Separación de Pruebas Unitarias e Integración:**
   - **Pruebas Unitarias:** Deben ejecutarse en milisegundos sin conectividad externa ni descarga de modelos pesados (`uv run pytest -v`).
   - **Pruebas de Integración:** Marcadas explícitamente con `@pytest.mark.integration`, encargadas de validar modelos reales en disco y restricciones de latencia (inferencia en CPU < 3.0 segundos).

3. **Auditoría Continua de Código:**
   - Todo cambio debe pasar limpiamente la auditoría del linter `ruff` (`uv run ruff check .`), con longitud máxima de línea de 100 caracteres y estilo PEP 8.

---

## 🧬 4. MODELO MaaS Y EXPLICABILIDAD CLÍNICA (XAI)

| Dimensión | Especificación Oficial |
|---|---|
| **Modelo Base** | `Anwarkh1/Skin_Cancer-Image_Classification` (Hugging Face) |
| **Arquitectura** | Vision Transformer (`google/vit-base-patch16-224-in21k`) con 85.8M de parámetros |
| **Licencia** | Apache-2.0 (compatible con desarrollo académico y comercial) |
| **Métricas** | 96.95% de Exactitud (*Validation Accuracy*) |
| **Formato de Pesos** | `safetensors` F32 (~343 MB) |
| **Clases HAM10000** | 7 patologías: `mel` (Melanoma), `nv` (Nevus), `bcc` (Carcinoma Basocelular), `akiec` (Queratosis Actínica), `bkl` (Queratosis Benigna), `df` (Dermatofibroma), `vasc` (Lesión Vascular) |
| **Explicabilidad XAI** | Grad-CAM aplicado a la última capa de auto-atención (`vit.encoder.layer[-1]`) para generar mapas de calor sobre la lesión cutánea |

---

## 📋 5. GESTIÓN ÁGIL Y GOBERNANZA GIT

1. **Tablero Kanban en GitHub Projects:**
   - Tablero público con flujo continuo: `To Do` → `In Progress` → `In Review` → `Done`.
   - Cada tarea cuenta con: descripción técnica, asignado, criterios de aceptación (*Acceptance Criteria - AC*) y estimación en la serie Fibonacci (`1, 2, 3, 5, 8, 13`).

2. **Flujo de Ramas y Peer Review:**
   - Las funcionalidades se desarrollan en ramas temáticas (`feature/`, `docs/`).
   - La integración a `main` requiere obligatoriamente una **Pull Request** formal con revisión cruzada entre integrantes (*Peer Review*), verificación de CI en verde y aprobación explícita.

3. **Commits Convencionales (Conventional Commits):**
   - Formato estricto: `tipo(alcance): descripción concisa en imperativo`.
   - Tipos válidos: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
