from __future__ import annotations

import numpy as np


def convolution_full_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Библиотечная реализация полной свертки для проверки ручного алгоритма."""
    return np.convolve(x, y, mode="full")


def correlation_full_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Библиотечная реализация полной корреляции для проверки ручного алгоритма."""
    return np.correlate(x, y, mode="full")


def fft_numpy(x: np.ndarray) -> np.ndarray:
    """Библиотечное прямое преобразование Фурье."""
    return np.fft.fft(x)


def ifft_numpy(x: np.ndarray) -> np.ndarray:
    """Библиотечное обратное преобразование Фурье."""
    return np.fft.ifft(x)
