"""COLIBRI: the color categories of people, on the HSI space.

P. Shamoi et al., "COLIBRI Fuzzy Model: Color Linguistic-Based Representation
and Interpretation", IEEE Access 13, 2025 (arXiv:2507.11488). The partitions
of hue, saturation and intensity come from a large survey of how people name
colors.

A color of very low saturation is achromatic and named after its intensity
alone: black, dark gray, gray, light gray or white. Every other color is
chromatic and named after its hue, its saturation and a coarser intensity,
which gives 9 x 3 x 3 + 5 = 86 colors.
"""

import fuzlog as fz

from fuzcolor.model import Family, Model, only
from fuzcolor.spaces import HSI

HUE = fz.partition(
    "hue",
    (0, 360),
    {
        "red": (350, 5),
        "orange": (20, 35),
        "yellow": 45,
        "green": (65, 130),
        "cyan": 175,
        "light blue": 190,
        "blue": (215, 240),
        "violet": (270, 280),
        "magenta": (320, 330),
    },
    periodic=True,
)

SATURATION = fz.partition(
    "saturation",
    (0, 1),
    {"very low": 0, "low": 0.15, "medium": (0.375, 0.45), "high": (0.725, 1)},
)

INTENSITY = fz.partition(
    "intensity",
    (0, 1),
    {
        "black": (0, 30 / 255),
        "dark gray": 75 / 255,
        "gray": (110 / 255, 115 / 255),
        "light gray": (190 / 255, 215 / 255),
        "white": 1,
    },
)

# Black with dark gray, gray, light gray with white: neighbours of a partition
# add up to a term whose core runs from the first core to the last.
SHADE = fz.partition(
    "intensity",
    (0, 1),
    dark=(0, 75 / 255),
    medium=(110 / 255, 115 / 255),
    light=(190 / 255, 1),
)

COLIBRI = Model(
    "COLIBRI",
    HSI,
    Family("chromatic", HUE, only(SATURATION, "low", "medium", "high"), SHADE),
    Family("achromatic", INTENSITY, only(SATURATION, "very low")),
)
