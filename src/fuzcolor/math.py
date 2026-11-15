"""Color conversions as plain NumPy functions.

Colors are arrays whose last axis holds the components. RGB with components in
[0, 1] is the exchange format: every other space converts from and to it.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Array = NDArray[np.float64]

_SQRT_3 = math.sqrt(3.0)


def as_rgb(colors: ArrayLike) -> Array:
    """Return colors as float RGB in [0, 1]: whole numbers are 8-bit levels."""
    given = np.asarray(colors)
    if given.ndim == 0 or given.shape[-1] != 3:
        raise ValueError(f"RGB needs a last axis of 3 components, got {given.shape}")
    if np.issubdtype(given.dtype, np.integer):
        return given / 255.0
    return given.astype(np.float64, copy=False)


def rgb_to_hsi(rgb: Array) -> Array:
    """Convert RGB in [0, 1] to hue in [0, 360), saturation and intensity in [0, 1].

    Hue is the angle of the color around the gray axis, 0 for red. Grays get
    hue 0 and black gets saturation 0.
    """
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    intensity = (red + green + blue) / 3.0
    weakest = np.minimum(np.minimum(red, green), blue)
    gray = np.divide(
        weakest, intensity, out=np.ones_like(intensity), where=intensity > 0
    )
    saturation = np.clip(1.0 - gray, 0.0, 1.0)
    angle = np.arctan2(_SQRT_3 * (green - blue), 2.0 * red - green - blue)
    hue = np.mod(np.degrees(angle), 360.0)
    hue = np.where(hue < 360.0, hue, 0.0)  # mod rounds tiny negative angles up to 360
    return np.stack([hue, saturation, intensity], axis=-1)


def hsi_to_rgb(hsi: Array) -> Array:
    """Convert hue in degrees, saturation and intensity in [0, 1] to RGB.

    Not every combination is a color: a light one cannot be saturated. Such
    combinations give components above 1.
    """
    hue, saturation, intensity = hsi[..., 0], hsi[..., 1], hsi[..., 2]
    # Each third of the hue circle has its own weakest primary: blue, red, green.
    third = np.floor_divide(hue, 120.0) % 3
    angle = np.radians(hue - 120.0 * third)
    weakest = intensity * (1.0 - saturation)
    leading = intensity * (
        1.0 + saturation * np.cos(angle) / np.cos(math.pi / 3 - angle)
    )
    remaining = 3.0 * intensity - (weakest + leading)
    first, second = third == 0, third == 1
    red = np.select([first, second], [leading, weakest], remaining)
    green = np.select([first, second], [remaining, leading], weakest)
    blue = np.select([first, second], [weakest, remaining], leading)
    return np.stack([red, green, blue], axis=-1)
