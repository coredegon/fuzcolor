# fuzcolor

Fuzzy color models: the colors people name, as fuzzy sets. A pixel belongs to
every color to a degree instead of falling into one bin. Built on
[fuzlog](https://github.com/coredegon/fuzlog) and NumPy, typed, Python 3.12+.

```sh
pip install fuzcolor
```

```python
import numpy as np
import fuzcolor as fc
from fuzcolor.models import COLIBRI

image = np.array(
    [[[200, 30, 30], [250, 220, 40]], [[30, 60, 200], [60, 60, 60]]], dtype=np.uint8
)

print(COLIBRI)  # Model('COLIBRI', HSI: 81 chromatic, 5 achromatic)
print([COLIBRI.colors[i] for i in COLIBRI.classify(image).flat])
# ['red/high/dark', 'yellow/high/light', 'blue/high/medium', 'dark gray']

weights = fc.palette(COLIBRI, image)  # the share of the image in each of 86 colors
fc.pie(COLIBRI, weights)
```

![An image, the color of every pixel and the palette of the image](https://raw.githubusercontent.com/coredegon/fuzcolor/main/docs/img/palette.png)

## Inside

| | |
|---|---|
| **Models** | `Model` and `Family` name colors as combinations of the terms of fuzlog variables |
| **Spaces** | `Space` describes the channels of a color space; `HSI` comes with the library |
| **Features** | `palette` describes an image, `tabulate` works a model out for every 8-bit color, `plot`, `swatches` and `pie` draw |
| **`fuzcolor.models`** | the models that come with the library: `COLIBRI` |
| **`fuzcolor.math`** | the color conversions as plain NumPy functions |

## Documentation

- [Guide](https://github.com/coredegon/fuzcolor/blob/main/docs/guide.md): every part, with pictures
- [COLIBRI](https://github.com/coredegon/fuzcolor/blob/main/docs/colibri.md): the model and its parameters

## Development

```sh
uv sync                 # environment with the dev tools
uv run pytest           # tests, including every example in the documentation
uv run ruff check       # lint; `uv run ruff format` formats
uv run ty check         # types
```

MIT license.
