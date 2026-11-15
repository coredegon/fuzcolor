"""Fuzzy color models: the colors people name, as fuzzy sets.

```python
import fuzcolor as fc
from fuzcolor.models import COLIBRI

COLIBRI.label([200, 30, 30])            # "red/high/dark"
weights = fc.palette(COLIBRI, image)    # the share of the image in every color
fc.pie(COLIBRI, weights)
```

- models: `Model`, built from the variables of fuzlog;
- spaces: `Space` and `HSI`;
- features: `palette`, `tabulate` and `Table`, the plots `plot`, `swatches`, `pie`;
- `fuzcolor.models`: the models that come with the library;
- `fuzcolor.math`: the color conversions as plain NumPy functions.
"""

from fuzcolor import features, math, models
from fuzcolor.features.palettes import palette
from fuzcolor.features.plots import pie, plot, swatches
from fuzcolor.features.tables import Table, tabulate
from fuzcolor.model import Model, only
from fuzcolor.spaces import HSI, Space

__all__ = [
    "HSI",
    "Model",
    "Space",
    "Table",
    "features",
    "math",
    "models",
    "only",
    "palette",
    "pie",
    "plot",
    "swatches",
    "tabulate",
]
