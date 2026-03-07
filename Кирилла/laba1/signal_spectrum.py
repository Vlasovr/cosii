import math
import matplotlib.pyplot as plt

def plot_spectrum(x, sample_rate, filename, title="Spectrum"):
    n = len(x)

    # частоты
    freqs = [i * sample_rate / n for i in range(n)]

    # амплитуда |X|
    amplitude = [abs(_x) for _x in x]

    # фаза arg(X)
    phase = [math.atan2(_x.imag, _x.real) for _x in x]

    # только первая половина спектра (симметрия для реальных сигналов)
    half = n // 2
    freqs = freqs[:half]
    amplitude = amplitude[:half]
    phase = phase[:half]

    plt.figure(figsize=(10, 6))

    # амплитудный спектр
    plt.subplot(2, 1, 1)
    plt.plot(freqs, amplitude)
    plt.title(f"{title} — Amplitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

    # фазовый спектр
    plt.subplot(2, 1, 2)
    plt.plot(freqs, phase)
    plt.title(f"{title} — Phase Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")

    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()