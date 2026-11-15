import fuzlog as fz
import numpy as np
import pytest

import fuzcolor as fc

HUE = fz.partition("hue", (0, 360), ["red", "green", "blue"], periodic=True)
SATURATION = fz.partition("saturation", (0, 1), ["gray", "pale", "vivid"])
INTENSITY = fz.partition("intensity", (0, 1), ["dark", "light"])
TINTS = fc.Family("tint", HUE, fc.only(SATURATION, "pale", "vivid"))
GRAYS = fc.Family("gray", INTENSITY, fc.only(SATURATION, "gray"))
MODEL = fc.Model("toy", fc.HSI, TINTS, GRAYS)


def test_a_family_names_every_combination_of_terms():
    assert TINTS.colors == (
        "red/pale",
        "red/vivid",
        "green/pale",
        "green/vivid",
        "blue/pale",
        "blue/vivid",
    )
    assert GRAYS.colors == ("dark", "light")  # one saturation: not in the name
    assert fc.Family("gray", fc.only(SATURATION, "gray")).colors == ("gray",)
    assert repr(TINTS) == "Family('tint': hue x saturation)"
    with pytest.raises(ValueError, match="different channels"):
        fc.Family("twice", HUE, HUE)
    with pytest.raises(ValueError, match="different channels"):
        fc.Family("empty")


def test_only_keeps_the_named_terms():
    kept = fc.only(HUE, "blue", "red")
    assert list(kept.terms) == ["blue", "red"]
    assert (kept.name, kept.universe, kept.periodic) == ("hue", (0, 360), True)
    assert kept.terms["red"] is HUE.terms["red"]
    with pytest.raises(ValueError, match="no terms"):
        fc.only(HUE, "pink")
    with pytest.raises(ValueError, match="no terms"):
        fc.only(HUE)


def test_a_model_gives_every_pixel_a_degree_in_every_color():
    assert MODEL.colors == (*TINTS.colors, *GRAYS.colors)
    assert repr(MODEL) == "Model('toy', HSI: 6 tint, 2 gray)"
    red = dict(zip(MODEL.colors, MODEL([255, 0, 0]).tolist(), strict=True))
    assert red["red/vivid"] == 1.0
    assert sum(red.values()) == 1.0
    pink = dict(zip(MODEL.colors, MODEL([1.0, 0.5, 0.5]).tolist(), strict=True))
    assert pink["red/pale"] == pytest.approx(0.5)  # saturation 0.25
    assert pink["light"] == pytest.approx(0.5 * 2 / 3)
    image = np.random.default_rng(0).integers(0, 256, (7, 5, 3), dtype=np.uint8)
    degrees = MODEL(image)
    assert degrees.shape == (8, 7, 5)
    assert degrees.sum(axis=0) == pytest.approx(1.0)
    assert MODEL(image / 255.0) == pytest.approx(degrees)


def test_classify_names_the_strongest_color():
    assert MODEL.colors[MODEL.classify([255, 0, 0])] == "red/vivid"
    assert MODEL.colors[MODEL.classify([20, 20, 20])] == "dark"
    image = np.random.default_rng(1).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    assert MODEL.classify(image).tolist() == MODEL(image).argmax(axis=0).tolist()
    by_minimum = fc.Model("toy", fc.HSI, TINTS, GRAYS, tnorm=np.minimum)
    best = by_minimum.classify(image)
    degrees = by_minimum(image)
    strongest = np.take_along_axis(degrees, best[None], axis=0)[0]
    assert strongest.tolist() == degrees.max(axis=0).tolist()


def test_a_pixel_of_no_color_gets_minus_one():
    vivid = fc.Model("vivid", fc.HSI, fc.Family("v", fc.only(SATURATION, "vivid")))
    assert vivid.classify([[255, 0, 0], [90, 90, 90]]).tolist() == [0, -1]
    assert vivid.classify([90, 90, 90]) == -1


def test_prototypes_are_typical_colors():
    typical = dict(zip(MODEL.colors, MODEL.prototypes().tolist(), strict=True))
    assert typical["dark"] == pytest.approx([0, 0, 0], abs=1e-3)
    assert typical["light"] == pytest.approx([1, 1, 1], abs=1e-3)
    red, green, blue = typical["red/vivid"]
    assert red > 0.9
    assert (green, blue) == pytest.approx((0, 0), abs=1e-3)
    inside = MODEL.prototypes()
    assert inside.shape == (8, 3)
    assert inside.min() >= 0
    assert inside.max() <= 1


def test_a_model_checks_its_families():
    with pytest.raises(ValueError, match="not a channel of HSI"):
        fc.Model("m", fc.HSI, fc.Family("f", fz.partition("value", (0, 1), ["a", "b"])))
    with pytest.raises(ValueError, match="not a channel of HSI"):
        fc.Model("m", fc.HSI, fc.Family("f", fz.partition("hue", (0, 1), ["a", "b"])))
    with pytest.raises(ValueError, match="colors that differ"):
        fc.Model("m", fc.HSI, GRAYS, fc.Family("again", INTENSITY))
    with pytest.raises(ValueError, match="colors that differ"):
        fc.Model("m", fc.HSI)
