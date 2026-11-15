import numpy as np
import pytest

from fuzcolor.math import as_rgb, hsi_to_rgb, rgb_to_hsi

# Colors whose hue, saturation and intensity follow from the definition.
KNOWN = {
    "red": ([1, 0, 0], [0, 1, 1 / 3]),
    "green": ([0, 1, 0], [120, 1, 1 / 3]),
    "blue": ([0, 0, 1], [240, 1, 1 / 3]),
    "yellow": ([1, 1, 0], [60, 1, 2 / 3]),
    "gray": ([0.5, 0.5, 0.5], [0, 0, 0.5]),
    "black": ([0, 0, 0], [0, 0, 0]),
    "pink": ([1, 0.5, 0.5], [0, 0.25, 2 / 3]),
}


@pytest.mark.parametrize(("rgb", "hsi"), KNOWN.values(), ids=KNOWN.keys())
def test_hsi_follows_its_definition(rgb, hsi):
    assert rgb_to_hsi(np.array(rgb, dtype=float)) == pytest.approx(hsi, abs=1e-12)


def test_hsi_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    hsi = rgb_to_hsi(rgb)
    assert hsi.shape == rgb.shape
    assert (hsi[..., 0] >= 0).all()
    assert (hsi[..., 0] < 360).all()
    assert hsi_to_rgb(hsi) == pytest.approx(rgb, abs=1e-9)


def test_whole_numbers_are_8_bit_levels():
    assert as_rgb(np.array([[0, 51, 255]], dtype=np.uint8)).tolist() == [[0, 0.2, 1]]
    assert as_rgb([0, 51, 255]).tolist() == [0, 0.2, 1]
    assert as_rgb([0.0, 0.2, 1.0]).tolist() == [0, 0.2, 1]
    with pytest.raises(ValueError, match="3 components"):
        as_rgb([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="3 components"):
        as_rgb(5)
