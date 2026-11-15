"""Fuzzy color models: colors as combinations of the terms of fuzlog variables."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import fuzlog as fz
import numpy as np
from fuzlog.types import Norm
from numpy.typing import ArrayLike, NDArray

from fuzcolor.math import Array, as_rgb
from fuzcolor.spaces import Space


def only(variable: fz.Variable, *terms: str) -> fz.Variable:
    """Return a variable with just the named terms of another, in that order.

    ```python
    only(saturation, "low", "medium", "high")
    ```
    """
    unknown = [term for term in terms if term not in variable.terms]
    if unknown or not terms:
        raise ValueError(
            f"{variable.name!r} has no terms {unknown}, only {list(variable.terms)}"
        )
    kept = {term: variable.terms[term] for term in terms}
    return fz.Variable(
        variable.name, variable.universe, kept, periodic=variable.periodic
    )


class Family:
    """The colors made of one term from each variable.

    ```python
    chromatic = Family("chromatic", hue, only(saturation, "low", "high"), shade)
    chromatic.colors    # ("red/low/dark", "red/low/medium", ...)
    ```

    Every variable describes the channel of a color space that it is named
    after. A color is named after the terms that tell it apart from the others:
    a variable with one term stays out of the name, and a family of one color
    takes the name of the family.
    """

    __slots__ = ("colors", "name", "variables")

    def __init__(self, name: str, *variables: fz.Variable) -> None:
        channels = [variable.name for variable in variables]
        if not variables or len(set(channels)) < len(channels):
            raise ValueError(
                f"the family {name!r} needs variables of different channels, "
                f"got {channels}"
            )
        self.name = name
        self.variables = variables
        telling = [len(variable.terms) > 1 for variable in variables]
        self.colors = tuple(
            "/".join(itertools.compress(terms, telling)) or name
            for terms in itertools.product(*(variable.terms for variable in variables))
        )

    def degrees(
        self, channels: Mapping[str, Array], tnorm: Norm = np.multiply
    ) -> Array:
        """Return the degree of every color: shape `(len(colors), *shape)`.

        `channels` holds the values of every channel by name; the degrees of
        the terms of a color are joined by `tnorm`.
        """
        terms = [variable(channels[variable.name]) for variable in self.variables]
        out = terms[0]
        for more in terms[1:]:  # every color so far with every term of the next
            out = tnorm(out[:, None], more[None]).reshape(-1, *more.shape[1:])
        return out

    def strongest(
        self, channels: Mapping[str, Array], tnorm: Norm = np.multiply
    ) -> tuple[NDArray[np.intp], Array]:
        """Return the position of the strongest color and its degree.

        A t-norm never decreases, so the strongest color is made of the
        strongest term of every variable: no other color has to be worked out.
        """
        terms = [variable(channels[variable.name]) for variable in self.variables]
        position, degree = terms[0].argmax(axis=0), terms[0].max(axis=0)
        for more in terms[1:]:
            position = position * len(more) + more.argmax(axis=0)
            degree = tnorm(degree, more.max(axis=0))
        return position, degree

    def __repr__(self) -> str:
        return f"Family({self.name!r}: {' x '.join(v.name for v in self.variables)})"


class Model:
    """A fuzzy color model: the families of colors named in a color space.

    ```python
    model = fc.Model("COLIBRI", fc.HSI, chromatic, achromatic)
    model.colors                 # the names of all colors, family by family
    model([200, 30, 30])         # the degree of the pixel in every color
    model.classify(image)        # the number of the strongest color per pixel
    ```

    Pixels have shape `(..., 3)`: 8-bit integers or floats in [0, 1].

    Args:
        name: What the model is called.
        space: The color space whose channels the variables describe.
        *families: The families of colors.
        tnorm: How the degrees of the terms of a color are joined. With the
            product, families of partitions give degrees that sum to 1.
    """

    __slots__ = ("colors", "families", "name", "space", "tnorm")

    def __init__(
        self, name: str, space: Space, *families: Family, tnorm: Norm = np.multiply
    ) -> None:
        for family in families:
            for variable in family.variables:
                if space.channels.get(variable.name) != variable.universe:
                    raise ValueError(
                        f"{family.name!r}: {variable.name!r} on {variable.universe} "
                        f"is not a channel of {space.name}: {space.channels}"
                    )
        colors = tuple(color for family in families for color in family.colors)
        if not colors or len(set(colors)) < len(colors):
            raise ValueError(f"{name!r} needs colors that differ, got {colors}")
        self.name = name
        self.space = space
        self.families = families
        self.tnorm = tnorm
        self.colors = colors

    def __call__(self, rgb: ArrayLike) -> Array:
        """Return the degree of the pixels in every color.

        The shape is `(len(colors), *rgb.shape[:-1])`, so the result unpacks
        into one array per color.
        """
        channels = self._channels(rgb)
        blocks = [family.degrees(channels, self.tnorm) for family in self.families]
        return np.concatenate(blocks)

    def classify(self, rgb: ArrayLike) -> NDArray[np.intp]:
        """Return the position in `colors` of the strongest color of every pixel.

        A pixel that belongs to no color gets -1.
        """
        channels = self._channels(rgb)
        first = 0
        best: NDArray[np.intp] = np.intp(-1)  # ty: ignore[invalid-assignment]
        degree: Array = np.float64(0.0)  # ty: ignore[invalid-assignment]
        for family in self.families:
            position, strength = family.strongest(channels, self.tnorm)
            best = np.where(strength > degree, first + position, best)
            degree = np.maximum(degree, strength)
            first += len(family.colors)
        return best

    def _channels(self, rgb: ArrayLike) -> dict[str, Array]:
        """The values of every channel of the space, by name."""
        values = self.space.from_rgb(as_rgb(rgb))
        return {name: values[..., i] for i, name in enumerate(self.space.channels)}

    def prototypes(self) -> Array:
        """Return a typical RGB color for every color: shape `(len(colors), 3)`.

        A channel takes the middle of the core of the color's term, or the
        middle of its range if the family does not describe it. What is too
        bright to be a color is darkened until it fits into RGB.
        """
        middle = [(low + high) / 2 for low, high in self.space.channels.values()]
        places = {channel: i for i, channel in enumerate(self.space.channels)}
        rows = []
        for family in self.families:
            centers = [_centers(variable).tolist() for variable in family.variables]
            for values in itertools.product(*centers):
                row = list(middle)
                for variable, value in zip(family.variables, values, strict=True):
                    row[places[variable.name]] = value
                rows.append(row)
        rgb = self.space.to_rgb(np.array(rows))
        return np.clip(rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1.0), 0.0, 1.0)

    def __repr__(self) -> str:
        families = ", ".join(f"{len(f.colors)} {f.name}" for f in self.families)
        return f"Model({self.name!r}, {self.space.name}: {families})"


def _centers(variable: fz.Variable) -> Array:
    """The most typical value of every term: the middle of where it is highest.

    On a periodic universe the middle is found going around the circle.
    """
    x = variable.grid(4001)
    degrees = variable(x)
    core = (degrees >= degrees.max(axis=1, keepdims=True) - 1e-9).astype(np.float64)
    if not variable.periodic:
        return (core @ x) / core.sum(axis=1)
    low, high = variable.universe
    angle = (x - low) * (math.tau / (high - low))
    middle = np.arctan2(core @ np.sin(angle), core @ np.cos(angle)) % math.tau
    return low + middle * ((high - low) / math.tau)
