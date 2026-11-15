"""Draw the figures of the documentation: ``uv run docs/figures.py``."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fuzlog as fz
import matplotlib.pyplot as plt

import fuzcolor as fc
from fuzcolor.models import COLIBRI

IMAGES = Path(__file__).parent / "img"


def save(figure, name):
    figure.savefig(IMAGES / name, dpi=144)
    plt.close(figure)


def palette():
    """A picture, the colors its pixels are classified as, and its palette."""
    image = plt.imread(IMAGES / "knight.jpg")
    figure, (left, middle, right) = plt.subplots(
        1, 3, figsize=(11, 5), width_ratios=(2, 2, 3.6), layout="constrained"
    )
    left.imshow(image)
    left.set_title("image")
    middle.imshow(COLIBRI.prototypes()[COLIBRI.classify(image)])
    middle.set_title("COLIBRI.classify(image)")
    for ax in (left, middle):
        ax.set_axis_off()
    fc.pie(COLIBRI, fc.palette(COLIBRI, image), right, least=0.015)
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
    save(fc.swatches(toy())[0].figure, "toy.png")  # guide.md
    save(fc.plot(COLIBRI)[0, 0].figure, "terms.png")  # colibri.md
    save(fc.swatches(COLIBRI)[0].figure, "swatches.png")  # colibri.md
