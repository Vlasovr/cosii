import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import cmath


def bit_reversal_permutation(x):
    N = len(x)
    bits = int(np.log2(N))
    reversed_indices = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(N)]
    return np.array([x[i] for i in reversed_indices], dtype=complex)


def fft_dif(x):
    N = len(x)
    if N == 1:
        return np.array(x)

    x = np.array(x, dtype=complex)
    half = N // 2

    even = x[:half] + x[half:]
    odd = (x[:half] - x[half:]) * np.exp(-2j * np.pi * np.arange(half) / N)

    fft_even = fft_dif(even)
    fft_odd = fft_dif(odd)

    return np.concatenate([fft_even, fft_odd])


def ifft_dif(X):
    N = len(X)
    conj_X = np.conj(X)
    fft_conj = fft_dif(conj_X)
    result = np.conj(bit_reversal_permutation(fft_conj)) / N
    return result


sample_rate = 128
duration = 6
frequency_sin = 2
frequency_cos = 3
multiply1 = 1
multiply2 = 1

t_full = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal_full = multiply1 * np.sin(2 * np.pi * frequency_sin * t_full) + \
              multiply2 * np.cos(2 * np.pi * frequency_cos * t_full)

signal_full /= np.max(np.abs(signal_full))

N = 2 ** int(np.floor(np.log2(len(signal_full))))
signal = signal_full[:N]
t = t_full[:N]

signal_int16 = np.int16(signal * 32767)
write("signal.wav", sample_rate, signal_int16)

signal_complex = [complex(x) for x in signal]
fft_raw = fft_dif(signal_complex)
fft_result = bit_reversal_permutation(fft_raw)
fft_magnitude = np.abs(fft_result)
frequencies = np.fft.fftfreq(N, d=1 / sample_rate)

ifft_result = ifft_dif(fft_result).real

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title(f"Сигнал: {multiply1}sin({frequency_sin}x) + {multiply2}cos({frequency_cos}x)")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.xlim(0, duration)

plt.subplot(3, 1, 2)
plt.stem(frequencies[:N // 2], fft_magnitude[:N // 2], basefmt=" ")
plt.title("Амплитудный спектр (БПФ, DIF + перестановка)")
plt.xlabel("Частота (Hz)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.xlim(0, 10)

plt.subplot(3, 1, 3)
plt.plot(t, ifft_result)
plt.title("Восстановленный сигнал после обратного БПФ")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.xlim(0, duration)

plt.tight_layout()
plt.show()
