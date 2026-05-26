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

    _print_step(
        "Старт",
        "Лабораторная работа 2: цифровая фильтрация сигналов",
        "Вариант: КИХ-фильтр типа Р с прямоугольным окном и однополюсный БИХ НЧ-фильтр.",
        f"Результаты будут сохранены в: {OUTPUT}",
    )

    # 1. В качестве входа используется первый периодический сигнал из ЛР1.
    clean, _ = generate_periodic_signals(SAMPLE_RATE, DURATION_SEC)

    # 2. В сигнал вносится контролируемая помеха: тон 700 Гц и слабый белый шум.
    rng = np.random.default_rng(42)
    t = np.arange(len(clean)) / SAMPLE_RATE
    interference = 0.35 * np.sin(2 * np.pi * 700.0 * t)
    random_noise = 0.03 * rng.normal(size=len(clean))
    distorted = clean + interference + random_noise
    distorted_rms_error = _rms_difference(distorted, clean)
    _print_step(
        "1. Входной сигнал и помеха готовы",
        f"Частота дискретизации: {SAMPLE_RATE} Гц",
        f"Длительность: {DURATION_SEC:.1f} с",
        f"Количество отсчетов: {len(clean)}",
        "К чистому сигналу добавлена синусоидальная помеха 700 Гц и слабый белый шум.",
        f"RMS-отклонение искаженного сигнала от чистого: {distorted_rms_error:.6f}",
    )

    # 3. Расчет коэффициентов трех фильтров
    h_homogeneous = moving_average_kernel(HOMOGENEOUS_WINDOW)
    h_notch = design_rectangular_notch_fir(
        sample_rate=SAMPLE_RATE,
        stop_low_hz=NOTCH_LOW_HZ,
        stop_high_hz=NOTCH_HIGH_HZ,
        taps=NOTCH_TAPS,
    )
    alpha = one_pole_lpf_alpha(SAMPLE_RATE, ONE_POLE_CUTOFF_HZ)
    _print_step(
        "2. Коэффициенты фильтров рассчитаны",
        f"Однородный фильтр: окно {HOMOGENEOUS_WINDOW}, коэффициент {h_homogeneous[0]:.6f}",
        f"КИХ Р: полоса подавления {NOTCH_LOW_HZ:.1f}..{NOTCH_HIGH_HZ:.1f} Гц, коэффициентов {NOTCH_TAPS}",
        f"КИХ Р: центральный коэффициент {h_notch[NOTCH_TAPS // 2]:.8f}",
        f"БИХ НЧ: частота среза {ONE_POLE_CUTOFF_HZ:.1f} Гц, alpha = {alpha:.8f}",
        f"БИХ НЧ: y[n] = {1.0 - alpha:.8f}*x[n] + {alpha:.8f}*y[n-1]",
    )

    # 4. Пропуск искаженного сигнала через однородный, КИХ и БИХ фильтры.
    y_homogeneous = apply_fir(distorted, h_homogeneous)
    y_fir = apply_fir(distorted, h_notch)
    y_iir = apply_one_pole_lpf(distorted, alpha)
    homogeneous_rms_error = _rms_difference(y_homogeneous, clean)
    fir_rms_error = _rms_difference(y_fir, clean)
    iir_rms_error = _rms_difference(y_iir, clean)
    _print_step(
        "3. Фильтрация выполнена",
        f"После однородного фильтра RMS-отклонение: {homogeneous_rms_error:.6f}",
        f"После КИХ Р с прямоугольным окном RMS-отклонение: {fir_rms_error:.6f}",
        f"После БИХ однополюсного НЧ RMS-отклонение: {iir_rms_error:.6f}",
        "Чем меньше RMS-отклонение, тем ближе результат к чистому сигналу.",
    )

    # 5. Сохранение исходного, искаженного и отфильтрованных сигналов.
    save_wav_mono_16(OUTPUT / "чистый_сигнал.wav", SAMPLE_RATE, clean)
    save_wav_mono_16(OUTPUT / "искаженный_сигнал.wav", SAMPLE_RATE, distorted)
    save_wav_mono_16(OUTPUT / "выход_однородный_фильтр.wav", SAMPLE_RATE, y_homogeneous)
    save_wav_mono_16(OUTPUT / "выход_ких_прямоугольное_окно.wav", SAMPLE_RATE, y_fir)
    save_wav_mono_16(OUTPUT / "выход_бих_однополюсный_нч.wav", SAMPLE_RATE, y_iir)
    _print_step(
        "4. WAV-файлы сохранены",
        "Сохранены чистый, искаженный и три отфильтрованных сигнала.",
    )

    # 6. Расчет АЧХ нужен для визуальной проверки свойств фильтров.
    homogeneous_fr = fir_frequency_response(h_homogeneous, SAMPLE_RATE)
    fir_fr = fir_frequency_response(h_notch, SAMPLE_RATE)
    iir_fr = one_pole_lpf_frequency_response(alpha, SAMPLE_RATE)

    homogeneous_700_hz = _response_at_frequency(homogeneous_fr, 700.0)
    fir_700_hz = _response_at_frequency(fir_fr, 700.0)
    iir_700_hz = _response_at_frequency(iir_fr, 700.0)
    _print_step(
        "5. АЧХ рассчитаны",
        f"Однородный фильтр на 700 Гц: |H| = {homogeneous_700_hz:.6f}, {_to_db(homogeneous_700_hz):.2f} дБ",
        f"КИХ Р на 700 Гц: |H| = {fir_700_hz:.6f}, {_to_db(fir_700_hz):.2f} дБ",
        f"БИХ НЧ на 700 Гц: |H| = {iir_700_hz:.6f}, {_to_db(iir_700_hz):.2f} дБ",
        "Чем меньше |H|, тем сильнее фильтр подавляет частоту помехи.",
    )

    # 7. Построение графиков АЧХ и временных диаграмм.
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
    _print_step(
        "6. Графики построены",
        "Сформированы АЧХ фильтров и сравнение сигналов во временной области.",
    )

    # 8. Сводка фиксирует параметры и рассчитанные коэффициенты для отчета.
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
        "",
        "Проверка подавления помехи 700 Гц по АЧХ:",
        f"Однородный фильтр: |H(700 Гц)| = {homogeneous_700_hz:.6f} ({_to_db(homogeneous_700_hz):.2f} дБ)",
        f"КИХ Р с прямоугольным окном: |H(700 Гц)| = {fir_700_hz:.6f} ({_to_db(fir_700_hz):.2f} дБ)",
        f"БИХ однополюсный НЧ: |H(700 Гц)| = {iir_700_hz:.6f} ({_to_db(iir_700_hz):.2f} дБ)",
        "",
        "RMS-отклонение от чистого сигнала:",
        f"Искаженный сигнал: {distorted_rms_error:.6f}",
        f"После однородного фильтра: {homogeneous_rms_error:.6f}",
        f"После КИХ Р с прямоугольным окном: {fir_rms_error:.6f}",
        f"После БИХ однополюсного НЧ: {iir_rms_error:.6f}",
        "",
        "Вывод:",
        "Однородный фильтр сглаживает сигнал, но не настроен точно на частоту помехи.",
        "КИХ режекторный фильтр подавляет область 650..750 Гц, где находится помеха 700 Гц.",
        "БИХ однополюсный НЧ-фильтр ослабляет высокие частоты выше частоты среза 450 Гц.",
    ]
    (OUTPUT / "сводка_результатов.txt").write_text("\n".join(lines), encoding="utf-8")
    _print_step(
        "7. Сводка сохранена",
        "Вывод: фильтрация уменьшает добавленную высокочастотную помеху.",
        "КИХ режекторный фильтр точечно подавляет область 650..750 Гц.",
        "БИХ НЧ-фильтр в целом ослабляет частоты выше 450 Гц.",
        f"Результаты сохранены в: {OUTPUT}",
    )


def _rms_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Возвращает среднеквадратичное отклонение двух сигналов."""
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def _response_at_frequency(response: tuple[np.ndarray, np.ndarray], frequency_hz: float) -> float:
    """Берет ближайшее значение АЧХ к заданной частоте."""
    freqs, magnitude = response
    index = int(np.argmin(np.abs(freqs - frequency_hz)))
    return float(magnitude[index])


def _to_db(magnitude: float) -> float:
    """Переводит модуль АЧХ в децибелы."""
    return float(20.0 * np.log10(max(magnitude, 1e-12)))


def _print_step(title: str, *lines: str) -> None:
    """Печатает этап выполнения сразу, без ожидания завершения программы."""
    print()
    print(title)
    for line in lines:
        print(f"  {line}")
    print(flush=True)


if __name__ == "__main__":
    run()
