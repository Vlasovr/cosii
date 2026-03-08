from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_time_comparison(
    clean: np.ndarray,
    distorted: np.ndarray,
    homogeneous: np.ndarray,
    fir: np.ndarray,
    iir: np.ndarray,
    sample_rate: int,
    path: Path,
    max_samples: int = 2500,
) -> None:
    _ensure_dir(path)

    samples = min(max_samples, len(clean))
    t = np.arange(samples) / sample_rate

    plt.figure(figsize=(11, 9))

    plt.subplot(5, 1, 1)
    plt.plot(t, clean[:samples])
    plt.title("Входной сигнал (из ЛР1)")
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 2)
    plt.plot(t, distorted[:samples])
    plt.title("Искаженный сигнал")
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 3)
    plt.plot(t, homogeneous[:samples])
    plt.title("Выход после однородного фильтра")
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 4)
    plt.plot(t, fir[:samples])
    plt.title("Выход после КИХ-фильтра с прямоугольным окном")
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 5)
    plt.plot(t, iir[:samples])
    plt.title("Выход после однополюсного НЧ-фильтра")
    plt.xlabel("Время, с")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_filter_responses(
    homogeneous_fr: tuple[np.ndarray, np.ndarray],
    fir_fr: tuple[np.ndarray, np.ndarray],
    iir_fr: tuple[np.ndarray, np.ndarray],
    path: Path,
) -> None:
    _ensure_dir(path)
    h_f, h_a = homogeneous_fr
    f_f, f_a = fir_fr
    i_f, i_a = iir_fr

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(h_f, 20 * np.log10(np.maximum(h_a, 1e-12)))
    plt.title("АЧХ однородного фильтра")
    plt.ylabel("Амплитуда, дБ")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(f_f, 20 * np.log10(np.maximum(f_a, 1e-12)))
    plt.title("АЧХ КИХ-фильтра (прямоугольное окно)")
    plt.ylabel("Амплитуда, дБ")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(i_f, 20 * np.log10(np.maximum(i_a, 1e-12)))
    plt.title("АЧХ однополюсного НЧ-фильтра")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда, дБ")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
