"""Draw the figures of the documentation: ``uv run docs/figures.py``."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fuzlog as fz
import matplotlib.pyplot as plt
import numpy as np

import fuzcolor as fc
from fuzcolor.models import COLIBRI

IMAGES = Path(__file__).parent / "img"


def sunset(height=240, width=360):
    """A picture to describe: a sky over a sea, with a low sun."""
    y, x = np.mgrid[0:height, 0:width]
    y, x = y / height, x / width
    sky = np.array([0.15, 0.25, 0.60]) + (y[..., None] / 0.6) ** 2 * [0.80, 0.35, -0.35]
    sea = np.array([0.05, 0.20, 0.35]) + (1 - y[..., None]) * [0.25, 0.20, 0.10]
    image = np.where(y[..., None] < 0.6, sky, sea)
    glow = np.exp(-(((x - 0.5) / 0.08) ** 2 + ((y - 0.52) / 0.10) ** 2))
    image += glow[..., None] * [0.9, 0.7, 0.3]
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def save(figure, name):
    figure.savefig(IMAGES / name, dpi=144)
    plt.close(figure)


def palette():
    """An image, the colors its pixels are classified as, and its palette."""
    image = sunset()
    figure, (left, middle, right) = plt.subplots(
        1, 3, figsize=(10, 3.4), width_ratios=(3, 3, 3.4), layout="constrained"
    )
    left.imshow(image)
    left.set_title("image")
    middle.imshow(COLIBRI.prototypes()[COLIBRI.classify(image)])
    middle.set_title("COLIBRI.classify(image)")
    for ax in (left, middle):
        ax.set_axis_off()
    fc.pie(COLIBRI, fc.palette(COLIBRI, image), right, least=0.04)
    right.set_title("fc.palette(COLIBRI, image)")
    save(figure, "palette.png")


def toy():
    """The model that the guide builds."""
    hue = fz.partition("hue", (0, 360), ["red", "green", "blue"], periodic=True)
    saturation = fz.partition("saturation", (0, 1), ["gray", "pale", "vivid"])
    intensity = fz.partition("intensity", (0, 1), ["dark", "light"])
    return fc.Model(
        "toy",
        fc.HSI,
        tint=[hue, fc.only(saturation, "pale", "vivid")],
        gray=[intensity, fc.only(saturation, "gray")],
    )


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    palette()  # README.md
    save(fc.swatches(toy()), "toy.png")  # guide.md
    save(fc.plot(COLIBRI), "terms.png")  # colibri.md
    save(fc.swatches(COLIBRI), "swatches.png")  # colibri.md
