from __future__ import annotations

from typing import Iterable

import numpy as np


def convolution_full(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    """Вычисляет полную линейную свертку прямым двойным суммированием."""
    x_np = np.asarray(list(x), dtype=np.float64)
    y_np = np.asarray(list(y), dtype=np.float64)

    n = len(x_np)
    m = len(y_np)
    out = np.zeros(n + m - 1, dtype=np.float64)

    # Каждый отсчет x умножается на каждый отсчет y и добавляется в позицию i + j.
    for i in range(n):
        xi = x_np[i]
        for j in range(m):
            out[i + j] += xi * y_np[j]

    return out


def correlation_full(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    """Вычисляет полную корреляцию как свертку с развернутым вторым сигналом."""
    y_reversed = list(y)[::-1]
    return convolution_full(x, y_reversed)
