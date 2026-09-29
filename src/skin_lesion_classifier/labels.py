"""
=============================================================================
MÓDULO: labels.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Responsabilidad Única (SRP)
RESPONSABILIDAD ÚNICA:
    Centralizar la taxonomía clínica del benchmark HAM10000: el nombre legible
    en español de cada una de las 7 patologías y su nivel de severidad, junto
    con el estilo de presentación (badge) asociado para la capa de presentación.

CLASES DEL BENCHMARK HAM10000:
    1. mel   -> Melanoma (maligno)
    2. nv    -> Nevus melanocítico (benigno)
    3. bcc   -> Carcinoma basocelular (maligno)
    4. akiec -> Queratosis actínica (precanceroso)
    5. bkl   -> Queratosis benigna (benigno)
    6. df    -> Dermatofibroma (benigno)
    7. vasc  -> Lesión vascular (benigno)
=============================================================================
"""

from typing import Final, Literal

# Códigos canónicos HAM10000 (reutiliza el conjunto definido en inference).
# La importación se hace localmente para no acoplar el módulo a detalles de
# inferencia; solo se usa con fines de validación defensiva.
HAM10000_CODES: Final[tuple[str, ...]] = (
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
)

# Nombre clínico legible en español para cada código HAM10000.
LABEL_TO_NAME: Final[dict[str, str]] = {
    "mel": "Melanoma",
    "nv": "Nevus melanocítico",
    "bcc": "Carcinoma basocelular",
    "akiec": "Queratosis actínica",
    "bkl": "Queratosis benigna",
    "df": "Dermatofibroma",
    "vasc": "Lesión vascular",
}

# Niveles de severidad clínica admitidos.
Severity = Literal["maligno", "precanceroso", "benigno"]

SEVERITY_MALIGNANT: Final[Severity] = "maligno"
SEVERITY_PRECANCEROUS: Final[Severity] = "precanceroso"
SEVERITY_BENIGN: Final[Severity] = "benigno"

# Severidad clínica por código HAM10000.
LABEL_TO_SEVERITY: Final[dict[str, Severity]] = {
    "mel": SEVERITY_MALIGNANT,
    "bcc": SEVERITY_MALIGNANT,
    "akiec": SEVERITY_PRECANCEROUS,
    "nv": SEVERITY_BENIGN,
    "bkl": SEVERITY_BENIGN,
    "df": SEVERITY_BENIGN,
    "vasc": SEVERITY_BENIGN,
}

# Estilo de presentación por severidad: (color del badge, icono Material).
# La capa de presentación lo consume directamente sin acoplarse a la lógica clínica.
SEVERITY_BADGE: Final[dict[Severity, tuple[str, str]]] = {
    SEVERITY_MALIGNANT: ("red", ":material/emergency:"),
    SEVERITY_PRECANCEROUS: ("orange", ":material/warning:"),
    SEVERITY_BENIGN: ("green", ":material/check_circle:"),
}


def human_name(label: str) -> str:
    """Devuelve el nombre clínico en español de un código HAM10000."""
    try:
        return LABEL_TO_NAME[label]
    except KeyError as err:
        raise ValueError(f"Código HAM10000 desconocido: {label}.") from err


def severity_of(label: str) -> Severity:
    """Devuelve la severidad clínica de un código HAM10000."""
    try:
        return LABEL_TO_SEVERITY[label]
    except KeyError as err:
        raise ValueError(f"Código HAM10000 desconocido: {label}.") from err


def severity_badge(severity: Severity) -> tuple[str, str]:
    """Devuelve (color, icono Material) para presentar una severidad como badge."""
    return SEVERITY_BADGE[severity]
