from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FFTFunction = Callable[[np.ndarray], np.ndarray]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_time_domain(
    signal: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
    max_samples: int | None = None,
) -> None:
    _ensure_dir(path)
    samples = len(signal) if max_samples is None else min(max_samples, len(signal))
    time_axis = np.arange(samples) / sample_rate

    plt.figure(figsize=(10, 3.5))
    plt.plot(time_axis, signal[:samples], linewidth=1.0)
    plt.title(title)
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_amplitude_phase_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    fft_fn: FFTFunction,
    path: Path,
    title: str,
) -> None:
    _ensure_dir(path)
    spectrum = fft_fn(signal)
    n = len(spectrum)

    frequencies = np.arange(n) * sample_rate / n
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    half = n // 2
    frequencies = frequencies[:half]
    amplitude = amplitude[:half]
    phase = phase[:half]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(frequencies, amplitude)
    plt.title(f"{title}: амплитудный спектр")
    plt.xlabel("Частота, Гц")
    plt.ylabel("|X(f)|")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(frequencies, phase)
    plt.title(f"{title}: фазовый спектр")
    plt.xlabel("Частота, Гц")
    plt.ylabel("arg(X(f)), рад")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_correlation_by_lag(
    correlation: np.ndarray,
    len_x: int,
    len_y: int,
    path: Path,
    title: str,
    max_points: int | None = None,
) -> None:
    _ensure_dir(path)
    lags = np.arange(-(len_y - 1), len_x)

    corr = correlation
    lag_axis = lags
    if max_points is not None and len(corr) > max_points:
        step = int(np.ceil(len(corr) / max_points))
        corr = corr[::step]
        lag_axis = lags[::step]

    plt.figure(figsize=(10, 3.5))
    plt.plot(lag_axis, corr, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Сдвиг (lag), отсчеты")
    plt.ylabel("Амплитуда")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_benchmark(
    sizes: list[int],
    direct_conv: list[float],
    fft_conv: list[float],
    direct_corr: list[float],
    fft_corr: list[float],
    path: Path,
) -> None:
    _ensure_dir(path)
    plt.figure(figsize=(10, 5))

    plt.plot(sizes, direct_conv, "o-", label="Свертка напрямую O(N^2)")
    plt.plot(sizes, fft_conv, "o-", label="Свертка через БПФ O(N log N)")
    plt.plot(sizes, direct_corr, "s-", label="Корреляция напрямую O(N^2)")
    plt.plot(sizes, fft_corr, "s-", label="Корреляция через БПФ O(N log N)")

    plt.xlabel("N (отсчеты)")
    plt.ylabel("Среднее время выполнения, с")
    plt.title("Сравнение эффективности при разных N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
