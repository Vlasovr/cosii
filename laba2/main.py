from __future__ import annotations

from pathlib import Path

import numpy as np

from laba1.signal_io import generate_periodic_signals, save_wav_mono_16
from laba2.filters import (
    apply_fir,
    apply_one_pole_lpf,
    design_rectangular_notch_fir,
    fir_frequency_response,
    moving_average_kernel,
    one_pole_lpf_alpha,
    one_pole_lpf_frequency_response,
)
from laba2.plotting import plot_filter_responses, plot_time_comparison


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

SAMPLE_RATE = 2_048
DURATION_SEC = 2.0

# Параметры варианта:
# КИХ: "Р с прямоугольным окном" -> режекторный (полосо-заграждающий) КИХ.
# БИХ: однополюсный НЧ-фильтр.
HOMOGENEOUS_WINDOW = 9
NOTCH_TAPS = 101
NOTCH_LOW_HZ = 650.0
NOTCH_HIGH_HZ = 750.0
ONE_POLE_CUTOFF_HZ = 450.0


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    clean, _ = generate_periodic_signals(SAMPLE_RATE, DURATION_SEC)

    rng = np.random.default_rng(42)
    t = np.arange(len(clean)) / SAMPLE_RATE
    interference = 0.35 * np.sin(2 * np.pi * 700.0 * t)
    random_noise = 0.03 * rng.normal(size=len(clean))
    distorted = clean + interference + random_noise

    h_homogeneous = moving_average_kernel(HOMOGENEOUS_WINDOW)
    h_notch = design_rectangular_notch_fir(
        sample_rate=SAMPLE_RATE,
        stop_low_hz=NOTCH_LOW_HZ,
        stop_high_hz=NOTCH_HIGH_HZ,
        taps=NOTCH_TAPS,
    )
    alpha = one_pole_lpf_alpha(SAMPLE_RATE, ONE_POLE_CUTOFF_HZ)

    y_homogeneous = apply_fir(distorted, h_homogeneous)
    y_fir = apply_fir(distorted, h_notch)
    y_iir = apply_one_pole_lpf(distorted, alpha)

    save_wav_mono_16(OUTPUT / "чистый_сигнал.wav", SAMPLE_RATE, clean)
    save_wav_mono_16(OUTPUT / "искаженный_сигнал.wav", SAMPLE_RATE, distorted)
    save_wav_mono_16(OUTPUT / "выход_однородный_фильтр.wav", SAMPLE_RATE, y_homogeneous)
    save_wav_mono_16(OUTPUT / "выход_ких_прямоугольное_окно.wav", SAMPLE_RATE, y_fir)
    save_wav_mono_16(OUTPUT / "выход_бих_однополюсный_нч.wav", SAMPLE_RATE, y_iir)

    homogeneous_fr = fir_frequency_response(h_homogeneous, SAMPLE_RATE)
    fir_fr = fir_frequency_response(h_notch, SAMPLE_RATE)
    iir_fr = one_pole_lpf_frequency_response(alpha, SAMPLE_RATE)

    plot_filter_responses(
        homogeneous_fr=homogeneous_fr,
        fir_fr=fir_fr,
        iir_fr=iir_fr,
        path=OUTPUT / "ачх_фильтров.png",
    )
    plot_time_comparison(
        clean=clean,
        distorted=distorted,
        homogeneous=y_homogeneous,
        fir=y_fir,
        iir=y_iir,
        sample_rate=SAMPLE_RATE,
        path=OUTPUT / "сравнение_сигналов_во_времени.png",
    )

    lines = [
        "Сводка по лабораторной работе 2",
        f"Частота дискретизации: {SAMPLE_RATE} Гц",
        f"Длительность: {DURATION_SEC} с",
        "",
        "Однородный фильтр:",
        f"Размер окна: {HOMOGENEOUS_WINDOW}",
        f"Коэффициенты h[n] = {np.array2string(h_homogeneous, precision=6)}",
        "",
        "КИХ режекторный фильтр (Р, прямоугольное окно):",
        f"Полоса подавления: {NOTCH_LOW_HZ:.1f}..{NOTCH_HIGH_HZ:.1f} Гц",
        f"Число коэффициентов: {NOTCH_TAPS}",
        f"Центральный коэффициент h[M] = {h_notch[NOTCH_TAPS // 2]:.8f}",
        f"Первые 10 коэффициентов = {np.array2string(h_notch[:10], precision=8)}",
        "",
        "БИХ однополюсный НЧ-фильтр:",
        f"Частота среза: {ONE_POLE_CUTOFF_HZ:.1f} Гц",
        f"alpha = exp(-2*pi*fc/fs) = {alpha:.8f}",
        "Разностное уравнение: y[n] = (1-alpha)*x[n] + alpha*y[n-1]",
    ]
    (OUTPUT / "сводка_результатов.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Лабораторная работа 2 выполнена.")
    print(f"Результаты сохранены в: {OUTPUT}")


if __name__ == "__main__":
    run()
