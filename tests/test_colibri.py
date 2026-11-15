import numpy as np
import pytest

from fuzcolor.models import COLIBRI
from fuzcolor.models.colibri import HUE, INTENSITY, SATURATION, SHADE

NAMED = {
    (0, 0, 0): "black",
    (60, 60, 60): "dark gray",
    (113, 113, 113): "gray",
    (200, 200, 200): "light gray",
    (255, 255, 255): "white",
    (255, 0, 0): "red/high/dark",
    (90, 160, 60): "green/medium/medium",
    (250, 220, 40): "yellow/high/light",
    (30, 60, 200): "blue/high/medium",
    (150, 110, 110): "red/low/medium",
}


def test_the_model_has_86_colors():
    assert repr(COLIBRI) == "Model('COLIBRI', HSI: 81 chromatic, 5 achromatic)"
    assert len(COLIBRI.colors) == 86
    assert COLIBRI.colors[:3] == ("red/low/dark", "red/low/medium", "red/low/light")
    assert COLIBRI.colors[-5:] == ("black", "dark gray", "gray", "light gray", "white")


@pytest.mark.parametrize(("rgb", "name"), NAMED.items(), ids=NAMED.values())
def test_colors_get_their_names(rgb, name):
    assert COLIBRI.label(rgb) == name


def test_every_color_belongs_to_the_model_completely():
    codes = np.arange(0, 2**24, 257)
    rgb = np.stack([codes >> 16, codes >> 8 & 255, codes & 255], axis=-1)
    degrees = COLIBRI(rgb.astype(np.uint8))
    assert degrees.sum(axis=0) == pytest.approx(1.0)
    assert COLIBRI.classify(rgb).tolist() == degrees.argmax(axis=0).tolist()


def test_the_partitions_keep_the_published_cores():
    assert HUE.periodic
    hues = [357.5, 20, 45, 100, 175, 190, 230, 275, 325]
    assert HUE(hues).tolist() == np.eye(9).tolist()
    assert SATURATION([0, 0.15, 0.4, 0.8]).tolist() == np.eye(4).tolist()
    levels = np.array([15, 75, 112, 200, 255]) / 255
    assert INTENSITY(levels).tolist() == np.eye(5).tolist()


def test_shades_are_the_intensities_joined():
    x = np.linspace(0, 1, 1001)
    black, dark_gray, gray, light_gray, white = INTENSITY(x)
    dark, medium, light = SHADE(x)
    assert dark == pytest.approx(black + dark_gray)
    assert medium == pytest.approx(gray)
    assert light == pytest.approx(light_gray + white)
