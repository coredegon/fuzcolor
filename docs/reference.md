# Reference

Everything is reached from the package: `import fuzcolor as fc`.

| Module | Holds |
|---|---|
| [`fuzcolor.model`](#model) | `Model`, `only` |
| [`fuzcolor.spaces`](#spaces) | `Space`, `HSI` |
| [`fuzcolor.features`](#features) | `palettes`, `tables`, `plots` |
| [`fuzcolor.models`](#models) | `COLIBRI` |
| [`fuzcolor.math`](#math) | color conversions |

Pixels, `rgb`, are arrays of shape `(..., 3)`: 8-bit integers or floats in [0, 1].

## Model

**`Model(name, space, *, tnorm=np.multiply, **families)`**: every keyword is a
family, a list of fuzlog variables named after channels of `space`.

| Member | Meaning |
|---|---|
| `model(rgb)` | degrees: shape `(len(colors), *rgb.shape[:-1])` |
| `model.classify(rgb)` | the position in `colors` of the strongest color; -1 for a pixel of no color |
| `model.label(rgb)` | the name of the strongest color: a string for one pixel, else an array; empty for a pixel of no color |
| `model.prototypes()` | a typical RGB color for every color: shape `(len(colors), 3)` |
| `model.colors` | the names of all colors, family by family; in a family the last variable changes fastest |
| `model.families` | the families: each has `name`, `variables` and `colors` |
| `name`, `space`, `tnorm` | as given |

**`only(variable, *terms)`** returns a variable with just the named terms.

## Spaces

**`Space(name, channels, from_rgb, to_rgb)`**: `channels` maps the name of every
channel to its range; the two functions convert float arrays of shape `(..., 3)`.

**`HSI`**: `hue` in (0, 360), `saturation` and `intensity` in (0, 1).

## Features

**`palette(model, image, *, crisp=False)`** returns the share of the image in
every color, in the order of `model.colors`: the mean degree of the pixels, or
with `crisp` the fraction of the pixels that the color is the strongest of.

**`tabulate(model)`** returns a `Table`: a
[fuzlog table](https://github.com/coredegon/fuzlog/blob/main/docs/reference.md#features)
over `red`, `green` and `blue`.

| Member of `Table` | Meaning |
|---|---|
| `table(rgb)` | `model.classify(rgb)` for 8-bit pixels, as `uint8`; 255 for a pixel of no color |
| `table.save(path)`, `Table.load(path)` | a NumPy `.npz` archive |
| `values`, `source` | the 256 × 256 × 256 positions, the name of the model |

**`plot(model)`** draws the terms of every family, each in the color it stands
for and named in a legend below, over a band of the channel, and returns the
grid of Axes; **`swatches(model)`** draws the typical color of every color and
returns an Axes per family.
**`pie(model, weights, ax=None, *, least=0.02)`** draws a palette as a ring with
a legend of the shares and returns the Axes; colors with a share below `least`
join one gray slice.

## Models

**`fuzcolor.models.COLIBRI`**: 86 colors on HSI. [Parameters](colibri.md).

## Math

| Function | Meaning |
|---|---|
| `as_rgb(colors)` | float RGB in [0, 1]; whole numbers are 8-bit levels |
| `rgb_to_hsi(rgb)` | hue in [0, 360), saturation and intensity in [0, 1] |
| `hsi_to_rgb(hsi)` | back to RGB; above 1 for what is no color |
