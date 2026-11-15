"""Color spaces: named channels and the way from and to RGB."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from fuzcolor.math import Array, hsi_to_rgb, rgb_to_hsi


class Space(NamedTuple):
    """A coordinate system for colors.

    ```python
    HSI = Space(
        "HSI",
        {"hue": (0, 360), "saturation": (0, 1), "intensity": (0, 1)},
        rgb_to_hsi,
        hsi_to_rgb,
    )
    ```

    `channels` gives every coordinate its range, in the order of the last axis
    that `from_rgb` returns and `to_rgb` takes. Both work on float arrays of
    shape `(..., 3)`, with RGB in [0, 1].
    """

    name: str
    channels: dict[str, tuple[float, float]]
    from_rgb: Callable[[Array], Array]
    to_rgb: Callable[[Array], Array]


HSI = Space(
    "HSI",
    {"hue": (0.0, 360.0), "saturation": (0.0, 1.0), "intensity": (0.0, 1.0)},
    rgb_to_hsi,
    hsi_to_rgb,
)
