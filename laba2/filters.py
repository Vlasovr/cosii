from __future__ import annotations

import numpy as np


def moving_average_kernel(window_size: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("Размер окна должен быть положительным.")
    return np.ones(window_size, dtype=np.float64) / window_size


def design_rectangular_notch_fir(
    sample_rate: int,
    stop_low_hz: float,
    stop_high_hz: float,
    taps: int,
) -> np.ndarray:
    if taps % 2 == 0:
        raise ValueError("Число коэффициентов taps должно быть нечетным для симметричного КИХ.")
    if not (0 < stop_low_hz < stop_high_hz < sample_rate / 2):
        raise ValueError("Полоса подавления должна лежать в диапазоне (0, Fs/2).")

    omega1 = 2.0 * np.pi * stop_low_hz / sample_rate
    omega2 = 2.0 * np.pi * stop_high_hz / sample_rate
    m = taps // 2

    h = np.zeros(taps, dtype=np.float64)
    for n in range(taps):
        k = n - m
        if k == 0:
            h[n] = 1.0 - (omega2 - omega1) / np.pi
        else:
            h[n] = -(
                (np.sin(omega2 * k) - np.sin(omega1 * k))
                / (np.pi * k)
            )

    # Прямоугольное окно: все коэффициенты окна равны единице.
    window = np.ones(taps, dtype=np.float64)
    return h * window


def apply_fir(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(signal, kernel, mode="same")


def one_pole_lpf_alpha(sample_rate: int, cutoff_hz: float) -> float:
    if cutoff_hz <= 0:
        raise ValueError("Частота среза должна быть положительной.")
    return float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))


def apply_one_pole_lpf(signal: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(signal, dtype=np.float64)
    if len(signal) == 0:
        return y

    y[0] = (1.0 - alpha) * signal[0]
    for n in range(1, len(signal)):
        y[n] = (1.0 - alpha) * signal[n] + alpha * y[n - 1]

    return y


def fir_frequency_response(kernel: np.ndarray, sample_rate: int, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    h = np.fft.rfft(kernel, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, np.abs(h)


def one_pole_lpf_frequency_response(alpha: float, sample_rate: int, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    w = np.linspace(0.0, np.pi, n_fft // 2 + 1)
    h = (1.0 - alpha) / (1.0 - alpha * np.exp(-1j * w))
    freqs = (w / (2.0 * np.pi)) * sample_rate
    return freqs, np.abs(h)
