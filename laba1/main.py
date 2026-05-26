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
    plot_correlation_by_lag,
    plot_time_domain,
)
from laba1.signal_io import generate_periodic_signals, save_wav_mono_16


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

SAMPLE_RATE = 2_048
DURATION_SEC = 2.0



def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    _print_step(
        "Старт",
        "Лабораторная работа 1: БПФ_Ч 0, свертка и корреляция",
        f"Результаты будут сохранены в: {OUTPUT}",
    )

    # 1. Генерация двух периодических сигналов и сохранение их в WAV-файлы.
    signal_x, signal_y = generate_periodic_signals(SAMPLE_RATE, DURATION_SEC)

    save_wav_mono_16(OUTPUT / "периодический_сигнал_1.wav", SAMPLE_RATE, signal_x)
    save_wav_mono_16(OUTPUT / "периодический_сигнал_2.wav", SAMPLE_RATE, signal_y)
    _print_step(
        "1. Сигналы готовы",
        f"Частота дискретизации: {SAMPLE_RATE} Гц",
        f"Длительность каждого сигнала: {DURATION_SEC:.1f} с",
        f"Длина X: {len(signal_x)} отсчетов, длина Y: {len(signal_y)} отсчетов",
        "Сигналы сохранены как 16-битные моно WAV-файлы.",
    )

    # 2. Ручные и библиотечные операции свертки и корреляции.
    conv_manual = convolution_full(signal_x, signal_y)
    corr_manual = correlation_full(signal_x, signal_y)
    conv_numpy = convolution_full_numpy(signal_x, signal_y)
    corr_numpy = correlation_full_numpy(signal_x, signal_y)
    conv_numpy_error = _max_abs_error(conv_manual, conv_numpy)
    corr_numpy_error = _max_abs_error(corr_manual, corr_numpy)

    save_wav_mono_16(OUTPUT / "сигнал_после_свертки.wav", SAMPLE_RATE, conv_manual)
    save_wav_mono_16(OUTPUT / "сигнал_после_корреляции.wav", SAMPLE_RATE, corr_manual)
    _print_step(
        "2. Свертка и корреляция посчитаны",
        f"Полная длина результата: N + M - 1 = {len(conv_manual)} отсчетов",
        f"Ручная свертка vs NumPy: {conv_numpy_error:.3e}",
        f"Ручная корреляция vs NumPy: {corr_numpy_error:.3e}",
        "Смысл: свертка объединяет сигналы, корреляция показывает сходство при сдвигах.",
    )

    # 3. Проверка теоремы Фурье для свертки и корреляции.
    conv_from_fft = convolution_via_fft_dif(signal_x, signal_y)
    corr_from_fft = correlation_via_fft_dif(signal_x, signal_y)
    conv_fft_error = _max_abs_error(conv_manual, conv_from_fft)
    corr_fft_error = _max_abs_error(corr_manual, corr_from_fft)
    _print_step(
        "3. Теорема Фурье проверена",
        f"Свертка напрямую vs через БПФ: {conv_fft_error:.3e}",
        f"Корреляция напрямую vs через БПФ: {corr_fft_error:.3e}",
        "Вывод: прямой расчет и расчет через БПФ дают одинаковый результат.",
    )

    # 4. Подготовка сигналов к БПФ: длины должны быть степенями двойки.
    x_pad = pad_to_power_of_two(signal_x)
    y_pad = pad_to_power_of_two(signal_y)
    conv_pad = pad_to_power_of_two(conv_manual)
    linear_conv_length = len(signal_x) + len(signal_y) - 1
    fft_conv_length = 1 << (linear_conv_length - 1).bit_length()

    x_manual_fft = fft_dif(x_pad)
    y_manual_fft = fft_dif(y_pad)
    conv_manual_fft = fft_dif(conv_pad)

    # 5. Обратное БПФ проверяет восстановление сигналов из спектра.
    x_manual_ifft = ifft_dif(x_manual_fft)
    y_manual_ifft = ifft_dif(y_manual_fft)
    conv_manual_ifft = ifft_dif(conv_manual_fft)

    # 6. Библиотечные FFT/IFFT нужны для численного сравнения.
    x_numpy_fft = fft_numpy(x_pad)
    y_numpy_fft = fft_numpy(y_pad)
    conv_numpy_fft = fft_numpy(conv_pad)

    x_numpy_ifft = ifft_numpy(x_numpy_fft)
    y_numpy_ifft = ifft_numpy(y_numpy_fft)
    conv_numpy_ifft = ifft_numpy(conv_numpy_fft)
    fft_x_error = _max_abs_error(x_manual_fft, x_numpy_fft)
    ifft_x_error = _max_abs_error(np.real(x_manual_ifft), np.real(x_numpy_ifft))
    fft_y_error = _max_abs_error(y_manual_fft, y_numpy_fft)
    ifft_y_error = _max_abs_error(np.real(y_manual_ifft), np.real(y_numpy_ifft))
    fft_conv_error = _max_abs_error(conv_manual_fft, conv_numpy_fft)
    ifft_conv_error = _max_abs_error(np.real(conv_manual_ifft), np.real(conv_numpy_ifft))
    _print_step(
        "4. БПФ_Ч 0 и обратное БПФ проверены",
        f"Длина БПФ для линейной свертки с нулями: {fft_conv_length}",
        f"FFT(X) против NumPy: {fft_x_error:.3e}",
        f"IFFT(X) против NumPy: {ifft_x_error:.3e}",
        f"FFT(Y) против NumPy: {fft_y_error:.3e}",
        f"IFFT(Y) против NumPy: {ifft_y_error:.3e}",
        f"FFT(свертки) против NumPy: {fft_conv_error:.3e}",
        f"IFFT(свертки) против NumPy: {ifft_conv_error:.3e}",
        "Вывод: ошибки имеют порядок машинной погрешности.",
    )

    # 7. Построение графиков во временной и частотной областях.
    plot_time_domain(
        signal_x,
        SAMPLE_RATE,
        OUTPUT / "сигнал_x_временная_область.png",
        "Сигнал X во временной области",
        max_samples=None,
    )
    plot_time_domain(
        signal_y,
        SAMPLE_RATE,
        OUTPUT / "сигнал_y_временная_область.png",
        "Сигнал Y во временной области",
        max_samples=None,
    )
    plot_time_domain(
        conv_manual,
        SAMPLE_RATE,
        OUTPUT / "результат_свертки_временная_область.png",
        "Результат свертки (полная длина N+M-1) во временной области",
        max_samples=None,
    )
    plot_correlation_by_lag(
        corr_manual,
        len(signal_x),
        len(signal_y),
        OUTPUT / "результат_корреляции_сдвиг.png",
        "Результат корреляции по оси сдвига (lag)",
    )

    plot_amplitude_phase_spectrum(signal_x, SAMPLE_RATE, _manual_fft, OUTPUT / "сигнал_x_спектр.png", "Сигнал X")
    plot_amplitude_phase_spectrum(signal_y, SAMPLE_RATE, _manual_fft, OUTPUT / "сигнал_y_спектр.png", "Сигнал Y")
    plot_amplitude_phase_spectrum(conv_manual, SAMPLE_RATE, _manual_fft, OUTPUT / "результат_свертки_спектр.png", "Результат свертки")
    _print_step(
        "5. Графики построены",
        "Сформированы временные графики, спектры и график корреляции по сдвигу.",
    )

    # 8. Бенчмарк показывает зависимость времени работы от количества отсчетов N.
    sizes = [256, 512, 1024, 2048]
    _print_step(
        "6. Запускается бенчмарк",
        "Сравнивается прямой расчет O(N^2) и расчет через БПФ O(N log N).",
        "Этот этап может выполняться дольше остальных.",
    )
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
        OUTPUT / "сравнение_эффективности_алгоритмов.png",
    )
    conv_speedup = direct_conv[-1] / fft_conv[-1]
    corr_speedup = direct_corr[-1] / fft_corr[-1]
    _print_step(
        "7. Бенчмарк завершен",
        f"При N={sizes[-1]} свертка через БПФ быстрее примерно в {conv_speedup:.1f} раза.",
        f"При N={sizes[-1]} корреляция через БПФ быстрее примерно в {corr_speedup:.1f} раза.",
    )

    # 9. Текстовая сводка фиксирует параметры эксперимента, ошибки и время выполнения.
    report_lines = [
        "Сводка по лабораторной работе 1",
        f"Частота дискретизации: {SAMPLE_RATE} Гц",
        f"Длительность: {DURATION_SEC} с",
        "",
        "Длины сигналов и временные оси:",
        f"N = {len(signal_x)}, M = {len(signal_y)}, N+M-1 = {linear_conv_length}",
        f"Длительность X: {len(signal_x) / SAMPLE_RATE:.6f} с",
        f"Длительность Y: {len(signal_y) / SAMPLE_RATE:.6f} с",
        f"Длительность полной свертки: {len(conv_manual) / SAMPLE_RATE:.6f} с",
        "",
        "Zero-padding для свертки через БПФ:",
        "Чтобы через БПФ получить линейную свертку (а не циклическую),",
        "оба сигнала дополняются нулями до длины Lfft >= N+M-1.",
        f"N+M-1 = {linear_conv_length}, выбранная длина БПФ Lfft = {fft_conv_length}",
        "",
        "Ошибки: ручная реализация vs NumPy",
        f"FFT(X): {fft_x_error:.6e}",
        f"IFFT(X): {ifft_x_error:.6e}",
        f"FFT(Y): {fft_y_error:.6e}",
        f"IFFT(Y): {ifft_y_error:.6e}",
        f"FFT(Conv): {fft_conv_error:.6e}",
        f"IFFT(Conv): {ifft_conv_error:.6e}",
        "",
        "Проверка свертки/корреляции через теорему Фурье",
        f"Свертка напрямую vs через БПФ: {conv_fft_error:.6e}",
        f"Корреляция напрямую vs через БПФ: {corr_fft_error:.6e}",
        "",
        "Свертка/корреляция: ручная реализация vs NumPy",
        f"Свертка ручная vs NumPy: {conv_numpy_error:.6e}",
        f"Корреляция ручная vs NumPy: {corr_numpy_error:.6e}",
        "",
        "Результаты бенчмарка (в секундах):",
    ]

    for index, n in enumerate(sizes):
        report_lines.append(
            f"N={n}: свертка_напрямую={direct_conv[index]:.6f}, свертка_БПФ={fft_conv[index]:.6f}, "
            f"корреляция_напрямую={direct_corr[index]:.6f}, корреляция_БПФ={fft_corr[index]:.6f}"
        )

    report_lines.extend(
        [
            "",
            "Вывод:",
            "Ручная реализация БПФ_Ч 0 совпадает с NumPy с точностью до машинной погрешности.",
            "Свертка и корреляция, рассчитанные напрямую и через БПФ, дают одинаковый результат.",
            f"При N={sizes[-1]} свертка через БПФ быстрее примерно в {conv_speedup:.1f} раза.",
            f"При N={sizes[-1]} корреляция через БПФ быстрее примерно в {corr_speedup:.1f} раза.",
        ]
    )

    (OUTPUT / "сводка_результатов.txt").write_text("\n".join(report_lines), encoding="utf-8")
    _print_step(
        "8. Сводка сохранена",
        "Лабораторная работа 1 выполнена.",
        f"Результаты сохранены в: {OUTPUT}",
    )


def _max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Возвращает максимальную абсолютную погрешность между двумя массивами."""
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n])))


def _manual_fft(signal: np.ndarray) -> np.ndarray:
    """Обертка для построения спектров ручной реализацией БПФ DIF."""
    return fft_dif(pad_to_power_of_two(signal))


def _print_step(title: str, *lines: str) -> None:
    """Печатает этап выполнения сразу, без ожидания завершения программы."""
    print()
    print(title)
    for line in lines:
        print(f"  {line}")
    print(flush=True)


if __name__ == "__main__":
    run()
