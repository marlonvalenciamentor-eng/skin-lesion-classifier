import numpy as np
import pytest
from PIL import Image

from skin_lesion_classifier.perturbations import (
    PerturbationError,
    add_gaussian_noise,
    darken,
)


def make_image(mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(3)
    pixels = rng.integers(60, 200, size=(24, 32, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB").convert(mode)


def test_darken_reduces_mean_brightness_and_keeps_size_and_mode() -> None:
    image = make_image()

    result = darken(image, factor=0.4)

    assert result.size == image.size
    assert result.mode == "RGB"
    assert np.asarray(result).mean() < np.asarray(image).mean() * 0.5


def test_darken_with_factor_one_keeps_pixels() -> None:
    image = make_image()

    result = darken(image, factor=1.0)

    assert np.array_equal(np.asarray(result), np.asarray(image))


@pytest.mark.parametrize("factor", [-0.1, 1.5])
def test_darken_rejects_factor_outside_zero_one(factor: float) -> None:
    with pytest.raises(PerturbationError, match="factor"):
        darken(make_image(), factor=factor)


def test_darken_converts_non_rgb_input_to_rgb() -> None:
    result = darken(make_image("L"), factor=0.4)

    assert result.mode == "RGB"


def test_add_gaussian_noise_is_deterministic_for_the_same_seed() -> None:
    image = make_image()

    first = add_gaussian_noise(image, sigma=20.0, seed=1)
    second = add_gaussian_noise(image, sigma=20.0, seed=1)

    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_add_gaussian_noise_differs_across_seeds_and_from_original() -> None:
    image = make_image()

    first = add_gaussian_noise(image, sigma=20.0, seed=1)
    second = add_gaussian_noise(image, sigma=20.0, seed=2)

    assert not np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(image))
    assert first.size == image.size
    assert first.mode == "RGB"


def test_add_gaussian_noise_with_zero_sigma_keeps_pixels() -> None:
    image = make_image()

    result = add_gaussian_noise(image, sigma=0.0, seed=1)

    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_add_gaussian_noise_rejects_negative_sigma() -> None:
    with pytest.raises(PerturbationError, match="sigma"):
        add_gaussian_noise(make_image(), sigma=-1.0, seed=1)


def test_add_gaussian_noise_does_not_mutate_the_input() -> None:
    image = make_image()
    before = np.asarray(image).copy()

    add_gaussian_noise(image, sigma=30.0, seed=5)

    assert np.array_equal(np.asarray(image), before)
