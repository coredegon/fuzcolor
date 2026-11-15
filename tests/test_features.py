import fuzlog as fz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import fuzcolor as fc
from fuzcolor.features.tables import NONE
from fuzcolor.models import COLIBRI

IMAGE = np.random.default_rng(0).integers(0, 256, (30, 40, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_palette_is_the_mean_degree_of_the_pixels():
    weights = fc.palette(COLIBRI, IMAGE)
    assert weights.shape == (86,)
    assert weights.sum() == pytest.approx(1.0)
    assert weights == pytest.approx(COLIBRI(IMAGE).mean(axis=(1, 2)))
    assert fc.palette(COLIBRI, IMAGE / 255.0) == pytest.approx(weights)
    red = fc.palette(COLIBRI, [255, 0, 0])  # one pixel: its own degrees
    assert red.tolist() == COLIBRI([255, 0, 0]).tolist()


def test_a_crisp_palette_counts_pixels():
    weights = fc.palette(COLIBRI, IMAGE, crisp=True)
    counts = np.bincount(COLIBRI.classify(IMAGE).ravel(), minlength=86)
    assert weights == pytest.approx(counts / IMAGE[..., 0].size)
    assert weights.sum() == pytest.approx(1.0)


def test_a_large_image_is_described_in_blocks():
    image = np.random.default_rng(1).integers(0, 256, (300, 300, 3), dtype=np.uint8)
    mean = COLIBRI(image).mean(axis=(1, 2))
    assert fc.palette(COLIBRI, image) == pytest.approx(mean)


def test_a_palette_needs_pixels():
    with pytest.raises(ValueError, match="without pixels"):
        fc.palette(COLIBRI, np.zeros((0, 3)))
    with pytest.raises(ValueError, match="3 components"):
        fc.palette(COLIBRI, np.zeros((4, 4)))


@pytest.fixture(scope="module")
def table():
    return fc.tabulate(COLIBRI)


def test_a_table_answers_as_the_model(table, tmp_path):
    assert repr(table) == (
        "Table(red[256] x green[256] x blue[256], nearest, from COLIBRI)"
    )
    assert table.values.dtype == np.uint8
    red, green, blue = np.moveaxis(IMAGE, -1, 0)
    expected = COLIBRI.classify(IMAGE).tolist()
    assert table.values[red, green, blue].tolist() == expected
    assert table(red=red, green=green, blue=blue).tolist() == expected
    table.save(tmp_path / "colibri.npz")
    loaded = fz.Table.load(tmp_path / "colibri.npz")
    assert np.array_equal(loaded.values, table.values)


def test_a_table_marks_the_pixels_of_no_color():
    saturation = fz.partition("saturation", (0, 1), ["gray", "pale", "vivid"])
    vivid = fc.Model("vivid", fc.HSI, fc.Family("v", fc.only(saturation, "vivid")))
    table = fc.tabulate(vivid)
    assert table.values[255, 0, 0] == 0
    assert table.values[90, 90, 90] == NONE


def test_plot_draws_the_terms_of_every_family():
    figure = fc.plot(COLIBRI)
    shown = [ax for ax in figure.axes if ax.axison and ax.get_xlabel()]
    channels = ["hue", "saturation", "intensity", "saturation", "intensity"]
    assert [ax.get_xlabel() for ax in shown] == channels
    assert [len(ax.lines) for ax in shown] == [9, 3, 3, 1, 5]
    assert shown[0].get_ylabel() == "chromatic\nmembership"
    assert shown[3].get_ylabel() == "achromatic\nmembership"


def test_swatches_show_every_color_once():
    figure = fc.swatches(COLIBRI)
    chromatic, achromatic = (np.asarray(ax.images[0].get_array()) for ax in figure.axes)
    assert chromatic.shape == (9, 9, 3)
    assert achromatic.shape == (1, 5, 3)
    typical = COLIBRI.prototypes()
    assert chromatic[1, 0].tolist() == typical[1].tolist()  # red, low, medium
    assert chromatic[0, 1].tolist() == typical[9].tolist()  # orange, low, dark
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert labels[:2] == ["low, dark", "low, medium"]


def test_pie_merges_the_small_shares():
    weights = np.zeros(86)
    weights[[0, 40, 85]] = 0.5, 0.49, 0.01
    ax = fc.pie(COLIBRI, weights)
    written = sorted(text.get_text() for text in ax.texts)
    names = ["red/low/dark", COLIBRI.colors[40], "other"]
    assert written == sorted([*names, "50%", "49%", "1%"])
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx(COLIBRI.prototypes()[0])
    _, given = plt.subplots()
    assert fc.pie(COLIBRI, weights, given, least=0) is given
    assert len(given.patches) == 3
    with pytest.raises(ValueError, match="86 weights"):
        fc.pie(COLIBRI, weights[:5])
