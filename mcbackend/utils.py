"""Contains helper functions that are independent of McBackend components."""
from typing import Sequence

import numpy as np


def as_array_from_ragged(arrs: Sequence[np.ndarray]) -> np.ndarray:
    shapes = {np.shape(arr) for arr in arrs}
    if len(shapes) > 1:
        return np.array(arrs, dtype=object)
    return np.array(arrs)
