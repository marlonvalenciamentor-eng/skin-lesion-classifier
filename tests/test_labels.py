import pytest

from skin_lesion_classifier.labels import (
    LABEL_TO_NAME,
    LABEL_TO_SEVERITY,
    human_name,
    severity_badge,
    severity_of,
)

HAM10000_EXPECTED = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def test_covers_exactly_the_7_ham10000_classes() -> None:
    assert set(LABEL_TO_NAME) == HAM10000_EXPECTED
    assert set(LABEL_TO_SEVERITY) == HAM10000_EXPECTED


def test_human_name_maps_each_code_to_spanish_name() -> None:
    assert human_name("mel") == "Melanoma"
    assert human_name("bcc") == "Carcinoma basocelular"
    assert human_name("nv") == "Nevus melanocítico"
    assert human_name("vasc") == "Lesión vascular"


def test_severity_classification() -> None:
    assert severity_of("mel") == "maligno"
    assert severity_of("bcc") == "maligno"
    assert severity_of("akiec") == "precanceroso"
    assert severity_of("nv") == "benigno"
    assert severity_of("bkl") == "benigno"
    assert severity_of("df") == "benigno"
    assert severity_of("vasc") == "benigno"


def test_severity_badge_returns_color_and_material_icon() -> None:
    color, icon = severity_badge("maligno")
    assert color == "red"
    assert icon.startswith(":material/")

    color_benign, _ = severity_badge("benigno")
    assert color_benign == "green"


def test_unknown_code_raises_value_error() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        human_name("xyz")
    with pytest.raises(ValueError, match="desconocido"):
        severity_of("xyz")
