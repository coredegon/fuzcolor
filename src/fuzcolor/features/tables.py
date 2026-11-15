"""Tables: a model worked out once for every 8-bit color."""

from __future__ import annotations

from typing import Any

import fuzlog as fz
import numpy as np
from numpy.typing import ArrayLike

from fuzcolor.math import Array
from fuzcolor.model import Model

NONE = 255  # what a table holds for a pixel that belongs to no color


class Table(fz.Table):
    """The strongest color of every 8-bit pixel, looked up; `tabulate` makes one.

    ```python
    table = fc.tabulate(model)
    table(image)                  # as model.classify(image), for 8-bit pixels
    table.save("colibri.npz")
    table = fc.Table.load("colibri.npz")
    ```

    It is a fuzlog table over `red`, `green` and `blue` that also takes pixels
    of shape `(..., 3)`. It answers with `uint8` positions in `model.colors`,
    and with `NONE`, 255, for a pixel that belongs to no color.
    """

    __slots__ = ()

    def __call__(self, rgb: ArrayLike | None = None, /, **inputs: ArrayLike) -> Any:
        """Look up the strongest color of 8-bit pixels."""
        if rgb is None:
            return super().__call__(**inputs)
        pixels = np.asarray(rgb)
        if pixels.dtype != np.uint8 or pixels.ndim == 0 or pixels.shape[-1] != 3:
            raise TypeError(
                f"a table takes 8-bit pixels of shape (..., 3), "
                f"got {pixels.dtype} of shape {pixels.shape}"
            )
        return self.values[pixels[..., 0], pixels[..., 1], pixels[..., 2]]


def tabulate(model: Model) -> Table:
    """Classify every 8-bit color with a model and keep the answers in a `Table`.

    ```python
    table = fc.tabulate(model)    # 256 x 256 x 256 of uint8: 16 MB, in seconds
    ```
    """
    if len(model.colors) > NONE:
        raise ValueError(f"a table counts up to {NONE} colors, got {len(model.colors)}")

    def strongest(red: Array, green: Array, blue: Array) -> Array:
        best = model.classify(np.stack([red, green, blue], axis=-1) / 255.0)
        return np.where(best < 0, NONE, best)

    levels = (0, 255, 256)
    axes = {"red": levels, "green": levels, "blue": levels}
    made = fz.tabulate(strongest, axes, method="nearest", dtype=np.uint8)
    return Table(made.axes, made.values, made.method, model.name)
