"""Plots: matplotlib views of models and palettes.

Fonts, sizes and the colour of text follow matplotlib's configuration; what is
painted are the colors of the model themselves. Matplotlib is imported by the
first plot.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import fuzlog as fz
import numpy as np
from numpy.typing import ArrayLike

from fuzcolor.model import Model

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_OTHER = (0.76, 0.76, 0.72)  # the slice of the colors too small to show
_EDGE = "#898781"


def plot(model: Model) -> Figure:
    """Draw the terms of a model: a row per family, a column per channel.

    ```python
    fc.plot(COLIBRI)
    ```
    """
    import matplotlib.pyplot as plt

    channels = list(model.space.channels)
    figure, axes = plt.subplots(
        len(model.families),
        len(channels),
        squeeze=False,
        figsize=(4.2 * len(channels), 2.6 * len(model.families)),
        layout="constrained",
    )
    for row, family in zip(axes, model.families, strict=True):
        described = {variable.name: variable for variable in family.variables}
        for ax, channel in zip(row, channels, strict=True):
            if channel in described:
                fz.plot(described[channel], ax)
            else:
                ax.set_axis_off()
        shown = next(ax for ax in row if ax.axison)
        shown.set_ylabel(f"{family.name}\nmembership")
    figure.suptitle(f"{model.name}: {len(model.colors)} colors")
    return figure


def swatches(model: Model) -> Figure:
    """Draw the typical color of every color of a model, a panel per family.

    ```python
    fc.swatches(COLIBRI)
    ```

    The terms of a family's first variable run across, the combinations of the
    terms of its other variables run down.
    """
    import matplotlib.pyplot as plt

    across = [len(family.variables[0].terms) for family in model.families]
    down = [
        len(family.colors) // n
        for family, n in zip(model.families, across, strict=True)
    ]
    figure, axes = plt.subplots(
        len(model.families),
        1,
        squeeze=False,
        figsize=(1.1 * max(across) + 2.0, 0.45 * sum(down) + 0.9 * len(down)),
        height_ratios=down,
        layout="constrained",
    )
    blocks = np.split(model.prototypes(), np.cumsum(np.multiply(across, down))[:-1])
    for ax, family, block, columns in zip(
        axes[:, 0], model.families, blocks, across, strict=True
    ):
        first, *rest = family.variables
        names = itertools.product(*(variable.terms for variable in rest))
        grid = block.reshape(columns, -1, 3).swapaxes(0, 1)
        ax.imshow(grid, aspect="auto")
        ax.set_xticks(range(columns), labels=list(first.terms))
        ax.set_yticks(range(len(grid)), labels=[", ".join(name) for name in names])
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(0.5, columns - 1), minor=True)
        ax.set_yticks(np.arange(0.5, len(grid) - 1), minor=True)
        ax.grid(which="minor", color=ax.get_facecolor(), linewidth=2)
        ax.tick_params(which="both", length=0)
        ax.set_title(family.name, loc="left")
        ax.spines[:].set(color=_EDGE, linewidth=0.5)
    figure.suptitle(f"{model.name}: {len(model.colors)} colors")
    return figure


def pie(
    model: Model, weights: ArrayLike, ax: Axes | None = None, *, least: float = 0.02
) -> Axes:
    """Draw a palette as a pie whose slices wear the colors they stand for.

    ```python
    fc.pie(COLIBRI, fc.palette(COLIBRI, image))
    ```

    Colors with a share below `least` are summed into one gray slice.
    """
    import matplotlib.pyplot as plt

    shares = np.asarray(weights, dtype=np.float64)
    if shares.shape != (len(model.colors),):
        raise ValueError(
            f"a palette of {model.name} has {len(model.colors)} weights, "
            f"got the shape {shares.shape}"
        )
    ranked = np.argsort(-shares, kind="stable")
    shown = [int(i) for i in ranked if shares[i] > 0.0 and shares[i] >= least]
    sizes = shares[shown].tolist()
    names = [model.colors[i] for i in shown]
    fills = [tuple(rgb) for rgb in model.prototypes()[shown].tolist()]
    rest = float(shares.sum()) - sum(sizes)
    if rest > 1e-9:
        sizes.append(rest)
        names.append("other")
        fills.append(_OTHER)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    _, _, percents = ax.pie(
        sizes,
        labels=names,
        colors=fills,
        autopct="%.0f%%",
        pctdistance=0.8,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": _EDGE, "linewidth": 0.5},
    )
    for percent, fill in zip(percents, fills, strict=True):
        dark = (
            math.fsum(
                w * c for w, c in zip((0.2126, 0.7152, 0.0722), fill, strict=True)
            )
            < 0.5
        )
        percent.set_color("white" if dark else "black")
    return ax
