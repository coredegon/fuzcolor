"""Plots: matplotlib views of models and palettes.

Every function returns the Axes it drew on, never the Figure: a notebook shows
a new figure once by itself, and would show a returned Figure a second time.
Fonts, sizes and the colour of text follow matplotlib's configuration; what is
painted are the colors of the model themselves. Matplotlib is imported by the
first plot.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

import fuzlog as fz
import numpy as np
from numpy.typing import ArrayLike, NDArray

from fuzcolor.math import Array
from fuzcolor.model import Family, Model, _centers

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_OTHER = (0.86, 0.86, 0.84)  # the slice of the colors too small to show
_EDGE = "#898781"
_PALE = 0.8  # a color brighter than this gets an outline to stand out on white
_RING = 0.42  # the width of the ring of a pie
_BAND = (-0.13, -0.03)  # where the band of a channel runs, under its curves


def plot(model: Model) -> NDArray[np.object_]:
    """Draw the terms of a model: a row per family, a column per channel.

    ```python
    axes = fc.plot(COLIBRI)
    axes[0, 0].figure.savefig("colibri.png")
    ```

    Every term wears the color it stands for and is named in a legend below,
    and a band under the axis shows how the channel runs through the colors of
    the family. A channel with more terms gets a wider column. Returns the
    grid of Axes, as `plt.subplots` does; a channel a family does not describe
    has its Axes turned off.
    """
    import matplotlib.pyplot as plt

    channels = list(model.space.channels)
    variables = [v for family in model.families for v in family.variables]
    widths = [
        2.4 + 0.5 * max((len(v.terms) for v in variables if v.name == c), default=0)
        for c in channels
    ]
    figure, axes = plt.subplots(
        len(model.families),
        len(channels),
        squeeze=False,
        figsize=(sum(widths), 3.3 * len(model.families)),
        width_ratios=widths,
        layout="constrained",
    )
    for row, family in zip(axes, model.families, strict=True):
        described = _described(family)
        for ax, channel in zip(row, channels, strict=True):
            if channel in described:
                _curves(model, family, described[channel], ax)
            else:
                ax.set_axis_off()
        shown = next(ax for ax in row if ax.axison)
        shown.set_ylabel(f"{family.name}\nmembership")
    figure.suptitle(f"{model.name}: {len(model.colors)} colors")
    return axes


def _described(family: Family) -> dict[str, fz.Variable]:
    return {variable.name: variable for variable in family.variables}


def _curves(model: Model, family: Family, variable: fz.Variable, ax: Axes) -> None:
    """Draw the terms of one variable of a family, each in its own color."""
    import matplotlib.patheffects as effects

    low, high = variable.universe
    x = np.linspace(low, high, 1001)
    tints = _tints(model, family, variable, _centers(variable)).tolist()
    curves = []
    for term, curve, tint in zip(variable.terms, variable(x), tints, strict=True):
        outline = []
        if _luminance(tint) > _PALE:
            outline = [
                effects.Stroke(linewidth=3.4, foreground=_EDGE),
                effects.Normal(),
            ]
        ax.fill_between(x, curve, color=tint, alpha=0.22, linewidth=0)
        # Only where the term is, so the zeros don't pile up along the bottom.
        inside = np.convolve(curve > 0, np.ones(3), mode="same") > 0
        curves += ax.plot(
            x,
            np.where(inside, curve, np.nan),
            color=tint,
            linewidth=2.2,
            path_effects=outline,
            label=term,
        )
    band = _tints(model, family, variable, np.linspace(low, high, 256))
    ax.imshow(
        band[None],
        extent=(low, high, *_BAND),
        aspect="auto",
        interpolation="bilinear",
    )
    ax.set(xlim=(low, high), ylim=(_BAND[0], 1.04), yticks=(0.0, 0.5, 1.0))
    ax.set(xlabel=variable.name, ylabel="membership")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_bounds(0.0, 1.0)
    ax.tick_params(axis="x", length=0)
    _legend_below(ax, curves)


def _legend_below(ax: Axes, handles: list[Any]) -> None:
    """Name the handles in a legend under the axis label, five to a row at most.

    A legend fills column after column; the handles are dealt so that they
    read in order along the rows.
    """
    columns = math.ceil(len(handles) / math.ceil(len(handles) / 5))
    dealt = [
        handles[i] for j in range(columns) for i in range(j, len(handles), columns)
    ]
    ax.legend(
        handles=dealt,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=4.0,  # in font sizes: clear of the ticks and the axis label
        ncol=columns,
        frameon=False,
        fontsize="small",
        handlelength=1.6,
        columnspacing=1.2,
    )


def _tints(model: Model, family: Family, variable: fz.Variable, values: Array) -> Array:
    """The RGB color of values of one variable among the colors of its family.

    The other channels stay at the only term of a variable of the family that
    has one term, and at the middle of their range otherwise.
    """
    channels = list(model.space.channels)
    point = [(low + high) / 2 for low, high in model.space.channels.values()]
    for other in family.variables:
        if len(other.terms) == 1:
            point[channels.index(other.name)] = float(_centers(other)[0])
    rows = np.tile(point, (len(values), 1))
    rows[:, channels.index(variable.name)] = values
    rgb = model.space.to_rgb(rows)
    return np.clip(rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1.0), 0.0, 1.0)


def _luminance(rgb: tuple[float, ...] | list[float]) -> float:
    return math.fsum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), rgb, strict=True))


def swatches(model: Model) -> NDArray[np.object_]:
    """Draw the typical color of every color of a model, a panel per family.

    ```python
    fc.swatches(COLIBRI)
    ```

    The terms of a family's first variable run across, the combinations of the
    terms of its other variables run down. Returns the Axes of the families.
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
    return axes[:, 0]


def pie(
    model: Model, weights: ArrayLike, ax: Axes | None = None, *, least: float = 0.02
) -> Axes:
    """Draw a palette as a ring whose slices wear the colors they stand for.

    ```python
    fc.pie(COLIBRI, fc.palette(COLIBRI, image))
    ```

    The legend names every slice with its share, the largest first; a slice
    of 5% or more has its share written on it too. Colors with a share below
    `least` are summed into one gray slice.
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
        _, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    total = math.fsum(sizes)
    wedges, _ = ax.pie(
        sizes,
        colors=fills,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": _RING, "edgecolor": _EDGE, "linewidth": 0.6},
    )
    for wedge, size, fill in zip(wedges, sizes, fills, strict=True):
        if size / total < 0.05:
            continue
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
        ax.text(
            0.79 * math.cos(angle),
            0.79 * math.sin(angle),
            f"{size / total:.0%}",
            ha="center",
            va="center",
            fontsize="small",
            fontweight="bold",
            color="black" if _luminance(fill) > 0.45 else "white",
        )
    if len(wedges) > 1:  # the same gap between every two slices, cut over them
        for wedge in wedges:
            angle = math.radians(wedge.theta1)
            ax.plot(
                [0.98 * (1 - _RING) * math.cos(angle), 1.02 * math.cos(angle)],
                [0.98 * (1 - _RING) * math.sin(angle), 1.02 * math.sin(angle)],
                color=ax.get_facecolor(),
                linewidth=1.8,
                solid_capstyle="butt",
            )
    ax.text(0, 0, model.name, ha="center", va="center")
    legend = ax.legend(
        wedges,
        [
            f"{size / total:4.0%}  {name}"
            for size, name in zip(sizes, names, strict=True)
        ],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        labelspacing=0.5,
        prop={"family": "monospace", "size": "small"},
    )
    for handle in legend.legend_handles:
        if handle is not None:
            handle.set(edgecolor=_EDGE, linewidth=0.5)
    ax.set(aspect="equal")
    return ax
