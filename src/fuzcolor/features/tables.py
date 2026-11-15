"""Tables: a model worked out once for every 8-bit color."""

from __future__ import annotations

import fuzlog as fz
import numpy as np

from fuzcolor.math import Array
from fuzcolor.model import Model

NONE = 255  # what a table holds for a pixel that belongs to no color


def tabulate(model: Model) -> fz.Table:
    """Classify every 8-bit RGB color and keep the answers in a fuzlog table.

    ```python
    table = fc.tabulate(model)                   # 256 x 256 x 256 of uint8
    red, green, blue = np.moveaxis(image, -1, 0)
    table.values[red, green, blue]               # model.classify(image), looked up
    table.save("colibri.npz")                    # fz.Table.load reads it
    ```

    The table holds the position of the strongest color in `model.colors`, and
    `NONE`, 255, for a color that belongs to none. Nothing is rounded: it
    answers as the model does for 8-bit pixels.
    """
    if len(model.colors) > NONE:
        raise ValueError(f"a table counts up to {NONE} colors, got {len(model.colors)}")

    def strongest(red: Array, green: Array, blue: Array) -> Array:
        best = model.classify(np.stack([red, green, blue], axis=-1) / 255.0)
        return np.where(best < 0, NONE, best)

    strongest.__name__ = model.name
    levels = (0, 255, 256)
    axes = {"red": levels, "green": levels, "blue": levels}
    return fz.tabulate(strongest, axes, method="nearest", dtype=np.uint8)
