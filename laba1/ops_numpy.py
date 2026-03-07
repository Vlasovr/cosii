from __future__ import annotations

import numpy as np


def convolution_full_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.convolve(x, y, mode="full")


def correlation_full_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.correlate(x, y, mode="full")


def fft_numpy(x: np.ndarray) -> np.ndarray:
    return np.fft.fft(x)


def ifft_numpy(x: np.ndarray) -> np.ndarray:
    return np.fft.ifft(x)

