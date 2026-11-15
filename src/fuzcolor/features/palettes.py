"""Palettes: how much of an image belongs to each color of a model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from fuzcolor.math import Array
from fuzcolor.model import Model

_BLOCK = 1 << 15  # pixels described at once: bounds the pixels x colors degrees


def palette(model: Model, image: ArrayLike, *, crisp: bool = False) -> Array:
    """Return the share of an image that belongs to each color of a model.

    ```python
    weights = fc.palette(model, image)       # weights[i] is of model.colors[i]
    dict(zip(model.colors, weights))
    ```

    The image has shape `(..., 3)`: 8-bit integers or floats in [0, 1]. Every
    pixel adds its degrees, so a weight is the mean degree of the pixels in
    that color. With `crisp` a pixel counts for its strongest color only, and
    the weights are fractions of the pixels.
    """
    pixels = np.asarray(image)
    if pixels.ndim == 0 or pixels.shape[-1] != 3:
        raise ValueError(f"RGB needs a last axis of 3 components, got {pixels.shape}")
    pixels = pixels.reshape(-1, 3)
    if not len(pixels):
        raise ValueError("an image without pixels has no palette")
    totals = np.zeros(len(model.colors))
    for start in range(0, len(pixels), _BLOCK):
        block = pixels[start : start + _BLOCK]
        if crisp:
            best = model.classify(block)
            totals += np.bincount(best[best >= 0], minlength=len(totals))
        else:
            totals += model(block).sum(axis=1)
    return totals / len(pixels)
