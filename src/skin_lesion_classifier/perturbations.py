"""
=============================================================================
MÓDULO: perturbations.py
PROYECTO: SkinLesionClassifier (Detección Temprana de Cáncer de Piel)
ARQUITECTURA: Clean Architecture / Funciones puras / SRP
RESPONSABILIDAD ÚNICA:
    Aplicar perturbaciones deterministas a imágenes dermatoscópicas (baja
    luminosidad y ruido gaussiano) para medir la robustez del modelo. Las
    funciones no mutan la entrada y siempre devuelven imágenes RGB del mismo
    tamaño.
=============================================================================
"""

import numpy as np
from PIL import Image, ImageEnhance


class PerturbationError(ValueError):
    """Error de dominio para parámetros de perturbación inválidos."""


def darken(image: Image.Image, factor: float = 0.4) -> Image.Image:
    """Reduce el brillo multiplicándolo por `factor` (0.0 = negro, 1.0 = sin cambio)."""
    if not 0.0 <= factor <= 1.0:
        raise PerturbationError(f"El factor de brillo debe estar en [0, 1]; recibido {factor}.")
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(factor)


def add_gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    """Suma ruido gaussiano N(0, sigma) por canal (escala 0-255), reproducible por `seed`."""
    if sigma < 0.0:
        raise PerturbationError(f"sigma debe ser no negativo; recibido {sigma}.")
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=pixels.shape)
    noisy = np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")
