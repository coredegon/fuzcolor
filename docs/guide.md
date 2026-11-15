# Guide

[Models](#models) · [Palettes](#palettes) · [Tables](#tables) · [Plots](#plots) ·
[Your own model](#your-own-model) · [Math](#math)

```python
import fuzlog as fz
import numpy as np
import fuzcolor as fc
from fuzcolor.models import COLIBRI
```

Pixels are arrays of shape `(..., 3)`: 8-bit integers or floats in [0, 1].

## Models

A model gives every pixel a degree in every one of its colors.

```python
print(COLIBRI)  # Model('COLIBRI', HSI: 81 chromatic, 5 achromatic)
print(COLIBRI.colors[:2], COLIBRI.colors[-2:])
# ('red/low/dark', 'red/low/medium') ('light gray', 'white')

degrees = COLIBRI([200, 30, 30])  # one degree per color
strongest = {
    COLIBRI.colors[i]: round(float(degrees[i]), 2) for i in np.argsort(-degrees)[:3]
}
print(strongest)
# {'red/high/dark': 0.49, 'red/high/medium': 0.25, 'red/medium/dark': 0.17}

image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
print(COLIBRI(image).shape)  # (86, 48, 64)
print(COLIBRI.classify(image).shape)  # (48, 64)
```

| Member | Meaning |
|---|---|
| `model(rgb)` | the degree of the pixels in every color: shape `(len(colors), *rgb.shape[:-1])` |
| `model.classify(rgb)` | the position in `colors` of the strongest color; -1 where a pixel belongs to none |
| `model.prototypes()` | a typical RGB color for every color: shape `(len(colors), 3)` |
| `name`, `space`, `families`, `colors`, `tnorm` | as given; `colors` lists the names, family by family |

## Palettes

`palette` tells how much of an image belongs to each color: `weights[i]` is the
share of `model.colors[i]`.

```python
weights = fc.palette(COLIBRI, image)
print(weights.shape, round(float(weights.sum()), 6))  # (86,) 1.0
counted = fc.palette(COLIBRI, image, crisp=True)  # each pixel for its strongest color
```

![An image, the color of every pixel and the palette of the image](img/palette.png)

| | Weight of a color |
|---|---|
| `palette(model, image)` | the mean degree of the pixels in it |
| `palette(model, image, crisp=True)` | the fraction of the pixels it is the strongest color of |

## Tables

`tabulate` classifies every 8-bit color once and returns a
[fuzlog table](https://github.com/coredegon/fuzlog/blob/main/docs/guide.md#tables):
256 × 256 × 256 numbers of `uint8`, 16 MB. It takes seconds to build and answers
as the model does, by lookup.

```python
table = fc.tabulate(COLIBRI)
print(table)  # Table(red[256] x green[256] x blue[256], nearest, from COLIBRI)

red, green, blue = np.moveaxis(image, -1, 0)
found = table.values[red, green, blue]
print(found.dtype, np.array_equal(found, COLIBRI.classify(image)))  # uint8 True

table.save("colibri.npz")
table = fz.Table.load("colibri.npz")  # needs fuzlog only
```

A color that belongs to no color of the model is stored as 255.

## Plots

```python
fc.plot(COLIBRI)  # the terms of every family
fc.swatches(COLIBRI)  # the typical color of every color
fc.pie(COLIBRI, weights)  # a palette
```

![The terms of the two families of COLIBRI](img/terms.png)

![The 86 colors of COLIBRI](img/swatches.png)

`plot` and `swatches` return the Figure, `pie` draws on an Axes of your own and
returns it. In a pie, colors with a share below `least=0.02` join one gray slice.

## Your own model

A color is one term from each variable of a family, and a variable is named
after the channel it describes.

```python
hue = fz.partition("hue", (0, 360), ["red", "green", "blue"], periodic=True)
saturation = fz.partition("saturation", (0, 1), ["gray", "pale", "vivid"])
intensity = fz.partition("intensity", (0, 1), ["dark", "light"])

toy = fc.Model(
    "toy",
    fc.HSI,
    fc.Family("tint", hue, fc.only(saturation, "pale", "vivid")),
    fc.Family("gray", intensity, fc.only(saturation, "gray")),
)
print(toy)  # Model('toy', HSI: 6 tint, 2 gray)
print(toy.colors)
# ('red/pale', 'red/vivid', 'green/pale', 'green/vivid', 'blue/pale', 'blue/vivid', 'dark', 'light')
print(toy.colors[toy.classify([0, 0, 255])])  # blue/vivid
```

| Part | Meaning |
|---|---|
| `Family(name, *variables)` | the colors made of one term from each variable |
| `only(variable, *terms)` | a variable with just these terms |
| `Model(name, space, *families, tnorm=np.multiply)` | the families together; `tnorm` joins the degrees of the terms of a color |
| `Space(name, channels, from_rgb, to_rgb)` | the channels with their ranges, and the way from and to RGB |

A color is named after the terms that tell it apart: a variable with one term
stays out of the name. With the product and partitions, the degrees of a pixel
sum to 1.

## Math

`fuzcolor.math` converts colors with plain NumPy functions on float arrays of
shape `(..., 3)`.

```python
from fuzcolor.math import as_rgb, hsi_to_rgb, rgb_to_hsi

hsi = rgb_to_hsi(as_rgb([255, 128, 128]))  # 8-bit levels to floats in [0, 1]
print(hsi.round(3).tolist())  # [0.0, 0.249, 0.668]
print((hsi_to_rgb(hsi) * 255).round().tolist())  # [255.0, 128.0, 128.0]
```
