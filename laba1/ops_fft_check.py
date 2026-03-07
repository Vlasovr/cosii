from __future__ import annotations

from typing import Iterable

import numpy as np

from laba1.fft_dif import fft_dif, ifft_dif, next_power_of_two


def _pad_to_length(signal: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float64)
    out[: len(signal)] = signal
    return out


def convolution_via_fft_dif(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    x_np = np.asarray(list(x), dtype=np.float64)
    y_np = np.asarray(list(y), dtype=np.float64)

    linear_size = len(x_np) + len(y_np) - 1
    fft_size = next_power_of_two(linear_size)

    x_pad = _pad_to_length(x_np, fft_size)
    y_pad = _pad_to_length(y_np, fft_size)

    X = fft_dif(x_pad)
    Y = fft_dif(y_pad)

    result = ifft_dif(X * Y)
    return np.real(result[:linear_size])


def correlation_via_fft_dif(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    y_reversed = np.asarray(list(y), dtype=np.float64)[::-1]
    return convolution_via_fft_dif(x, y_reversed)

