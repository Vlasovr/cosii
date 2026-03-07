from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np


MeasureFn = Callable[[Iterable[float], Iterable[float]], np.ndarray]


def _measure_seconds(fn: MeasureFn, x: np.ndarray, y: np.ndarray, repeats: int = 3) -> float:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(x, y)
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings))


def benchmark_pair(
    x: np.ndarray,
    y: np.ndarray,
    direct_fn: MeasureFn,
    fft_fn: MeasureFn,
    sizes: list[int],
) -> tuple[list[int], list[float], list[float]]:
    direct_times: list[float] = []
    fft_times: list[float] = []

    for n in sizes:
        x_n = x[:n]
        y_n = y[:n]
        direct_times.append(_measure_seconds(direct_fn, x_n, y_n))
        fft_times.append(_measure_seconds(fft_fn, x_n, y_n))

    return sizes, direct_times, fft_times

