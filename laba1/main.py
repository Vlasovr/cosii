from __future__ import annotations

from pathlib import Path

import numpy as np

from laba1.benchmark import benchmark_pair
from laba1.fft_dif import fft_dif, ifft_dif, pad_to_power_of_two
from laba1.ops_fft_check import convolution_via_fft_dif, correlation_via_fft_dif
from laba1.ops_manual import convolution_full, correlation_full
from laba1.ops_numpy import (
    convolution_full_numpy,
    correlation_full_numpy,
    fft_numpy,
    ifft_numpy,
)
from laba1.plotting import (
    plot_amplitude_phase_spectrum,
    plot_benchmark,
    plot_time_domain,
)
from laba1.signal_io import generate_periodic_signals, save_wav_mono_16


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

SAMPLE_RATE = 2_048
DURATION_SEC = 2.0


def _max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n])))


def _manual_fft(signal: np.ndarray) -> np.ndarray:
    return fft_dif(pad_to_power_of_two(signal))


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    signal_x, signal_y = generate_periodic_signals(SAMPLE_RATE, DURATION_SEC)

    save_wav_mono_16(OUTPUT / "periodic_signal_1.wav", SAMPLE_RATE, signal_x)
    save_wav_mono_16(OUTPUT / "periodic_signal_2.wav", SAMPLE_RATE, signal_y)

    conv_manual = convolution_full(signal_x, signal_y)
    corr_manual = correlation_full(signal_x, signal_y)
    conv_numpy = convolution_full_numpy(signal_x, signal_y)
    corr_numpy = correlation_full_numpy(signal_x, signal_y)

    save_wav_mono_16(OUTPUT / "signals_after_convolution.wav", SAMPLE_RATE, conv_manual)
    save_wav_mono_16(OUTPUT / "signals_after_correlation.wav", SAMPLE_RATE, corr_manual)

    conv_from_fft = convolution_via_fft_dif(signal_x, signal_y)
    corr_from_fft = correlation_via_fft_dif(signal_x, signal_y)

    x_pad = pad_to_power_of_two(signal_x)
    y_pad = pad_to_power_of_two(signal_y)
    conv_pad = pad_to_power_of_two(conv_manual)

    x_manual_fft = fft_dif(x_pad)
    y_manual_fft = fft_dif(y_pad)
    conv_manual_fft = fft_dif(conv_pad)

    x_manual_ifft = ifft_dif(x_manual_fft)
    y_manual_ifft = ifft_dif(y_manual_fft)
    conv_manual_ifft = ifft_dif(conv_manual_fft)

    x_numpy_fft = fft_numpy(x_pad)
    y_numpy_fft = fft_numpy(y_pad)
    conv_numpy_fft = fft_numpy(conv_pad)

    x_numpy_ifft = ifft_numpy(x_numpy_fft)
    y_numpy_ifft = ifft_numpy(y_numpy_fft)
    conv_numpy_ifft = ifft_numpy(conv_numpy_fft)

    plot_time_domain(signal_x, SAMPLE_RATE, OUTPUT / "signal_x_time.png", "Сигнал X во временной области")
    plot_time_domain(signal_y, SAMPLE_RATE, OUTPUT / "signal_y_time.png", "Сигнал Y во временной области")
    plot_time_domain(conv_manual, SAMPLE_RATE, OUTPUT / "signal_conv_time.png", "Результат свертки во временной области")
    plot_time_domain(corr_manual, SAMPLE_RATE, OUTPUT / "signal_corr_time.png", "Результат корреляции во временной области")

    plot_amplitude_phase_spectrum(signal_x, SAMPLE_RATE, _manual_fft, OUTPUT / "signal_x_spectrum.png", "Сигнал X")
    plot_amplitude_phase_spectrum(signal_y, SAMPLE_RATE, _manual_fft, OUTPUT / "signal_y_spectrum.png", "Сигнал Y")
    plot_amplitude_phase_spectrum(conv_manual, SAMPLE_RATE, _manual_fft, OUTPUT / "signal_conv_spectrum.png", "Результат свертки")

    sizes = [256, 512, 1024, 2048]
    sizes, direct_conv, fft_conv = benchmark_pair(
        signal_x,
        signal_y,
        convolution_full,
        convolution_via_fft_dif,
        sizes,
    )
    _, direct_corr, fft_corr = benchmark_pair(
        signal_x,
        signal_y,
        correlation_full,
        correlation_via_fft_dif,
        sizes,
    )
    plot_benchmark(
        sizes,
        direct_conv,
        fft_conv,
        direct_corr,
        fft_corr,
        OUTPUT / "algorithm_efficiency.png",
    )

    report_lines = [
        "Сводка по лабораторной работе 1",
        f"Частота дискретизации: {SAMPLE_RATE} Гц",
        f"Длительность: {DURATION_SEC} с",
        "",
        "Ошибки: ручная реализация vs NumPy",
        f"FFT(X): {_max_abs_error(x_manual_fft, x_numpy_fft):.6e}",
        f"IFFT(X): {_max_abs_error(np.real(x_manual_ifft), np.real(x_numpy_ifft)):.6e}",
        f"FFT(Y): {_max_abs_error(y_manual_fft, y_numpy_fft):.6e}",
        f"IFFT(Y): {_max_abs_error(np.real(y_manual_ifft), np.real(y_numpy_ifft)):.6e}",
        f"FFT(Conv): {_max_abs_error(conv_manual_fft, conv_numpy_fft):.6e}",
        f"IFFT(Conv): {_max_abs_error(np.real(conv_manual_ifft), np.real(conv_numpy_ifft)):.6e}",
        "",
        "Проверка свертки/корреляции через теорему Фурье",
        f"Свертка напрямую vs через БПФ: {_max_abs_error(conv_manual, conv_from_fft):.6e}",
        f"Корреляция напрямую vs через БПФ: {_max_abs_error(corr_manual, corr_from_fft):.6e}",
        "",
        "Свертка/корреляция: ручная реализация vs NumPy",
        f"Свертка ручная vs NumPy: {_max_abs_error(conv_manual, conv_numpy):.6e}",
        f"Корреляция ручная vs NumPy: {_max_abs_error(corr_manual, corr_numpy):.6e}",
        "",
        "Результаты бенчмарка (в секундах):",
    ]

    for index, n in enumerate(sizes):
        report_lines.append(
            f"N={n}: свертка_напрямую={direct_conv[index]:.6f}, свертка_БПФ={fft_conv[index]:.6f}, "
            f"корреляция_напрямую={direct_corr[index]:.6f}, корреляция_БПФ={fft_corr[index]:.6f}"
        )

    (OUTPUT / "summary.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("Лабораторная работа 1 выполнена.")
    print(f"Результаты сохранены в: {OUTPUT}")


if __name__ == "__main__":
    run()
