from PIL import Image

from skin_lesion_classifier.report import build_report_pdf

SEVEN_CLASS_PROBABILITIES = {
    "akiec": 0.02,
    "bcc": 0.03,
    "bkl": 0.02,
    "df": 0.01,
    "mel": 0.90,
    "nv": 0.01,
    "vasc": 0.01,
}


def test_build_report_pdf_returns_valid_pdf() -> None:
    # Arrange
    image = Image.new("RGB", (300, 200), color=(120, 80, 60))
    overlay = Image.new("RGB", (300, 200), color=(10, 20, 30))

    # Act
    pdf = build_report_pdf(
        patient_name="Paciente de prueba",
        image=image,
        overlay=overlay,
        label_name="Melanoma",
        severity="maligno",
        confidence=0.9,
        probabilities=SEVEN_CLASS_PROBABILITIES,
    )

    # Assert
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_build_report_pdf_falls_back_when_patient_name_is_blank() -> None:
    # Arrange
    image = Image.new("RGB", (224, 224), color=(120, 80, 60))

    # Act
    pdf = build_report_pdf(
        patient_name="   ",
        image=image,
        overlay=image,
        label_name="Nevus melanocítico",
        severity="benigno",
        confidence=0.99,
        probabilities=SEVEN_CLASS_PROBABILITIES,
    )

    # Assert
    assert pdf.startswith(b"%PDF")
