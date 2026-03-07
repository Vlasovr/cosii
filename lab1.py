import wave
import numpy as np
import matplotlib.pyplot as plt

WAV_FILENAME = "input.wav"
N = 16

def parse_file(filename, n):
    try:
        wav = wave.open(filename, "rb")
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        exit(1)

    n_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    n_frames = wav.getnframes()

    raw_data = wav.readframes(n_frames)
    wav.close()

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 1:
        dtype = np.uint8
    else:
        print("Неизвестный sample_width =", sample_width)
        exit(1)
    audio = np.frombuffer(raw_data, dtype=dtype)

    if n_channels > 1:
        audio = audio[::n_channels]

    if len(audio) < n:
        print(f"Ошибка: в файле меньше {n} отсчётов.")
        exit(1)
    return audio[:n].astype(float)

def dft(x):
    N = len(x)
    X = [0j] * N  # список комплексных чисел
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def idft(X):
    N = len(X)
    x = [0j] * N
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N
    return x

def fft(x):
    n = len(x)
    if n <= 1:
        return list(x) if not isinstance(x, list) else x[:]

    half = n // 2
    g = [x[k] + x[k + half] for k in range(half)]
    h = [(x[k] - x[k + half]) * np.exp(-2j * np.pi * k / n) for k in range(half)]

    G = fft(g)
    H = fft(h)

    X = []
    for k in range(half):
        X.append(G[k])
        X.append(H[k])
    return X

def ifft(X):
    n = len(X)
    if n <= 1:
        return list(X) if not isinstance(X, list) else X[:]

    half = n // 2
    g = [X[k] + X[k + half] for k in range(half)]
    h = [(X[k] - X[k + half]) * np.exp(2j * np.pi * k / n) for k in range(half)]

    G = ifft(g)
    H = ifft(h)

    Y = []
    for k in range(half):
        Y.append(G[k])
        Y.append(H[k])
    return Y

signal = parse_file(WAV_FILENAME, N)
t = np.arange(N)

dft_result = dft(signal)
fft_result = fft(signal)
np_fft = np.fft.fft(signal)

idft_result = np.array(idft(dft_result)).real
ifft_result = np.array(ifft(fft_result)).real
np_ifft = np.fft.ifft(np_fft).real

fig, axes = plt.subplots(3, 3, figsize=(20, 10))

# Верхний центральный график (входной сигнал)
axes[0, 1].stem(t, signal, basefmt=" ")
axes[0, 1].set_title("Входной сигнал")
axes[0, 1].set_xlabel("Время")
axes[0, 1].set_ylabel("Амплитуда")

# Спрячем пустые ячейки
axes[0, 0].axis("off")
axes[0, 2].axis("off")

# Первая строка: сравнение прямых преобразований
axes[1, 0].stem(t, np.abs(dft_result), basefmt=" ")
axes[1, 0].set_title("DFT")
axes[1, 0].set_xlabel("Частота")
axes[1, 0].set_ylabel("Амплитуда")

axes[1, 1].stem(t, np.abs(fft_result), basefmt=" ")
axes[1, 1].set_title("FFT")
axes[1, 1].set_xlabel("Частота")
axes[1, 1].set_ylabel("Амплитуда")

axes[1, 2].stem(t, np.abs(np_fft), basefmt=" ")
axes[1, 2].set_title("NumPy FFT")
axes[1, 2].set_xlabel("Частота")
axes[1, 2].set_ylabel("Амплитуда")

# Вторая строка: сравнение обратных преобразований
axes[2, 0].stem(t, idft_result, basefmt=" ")
axes[2, 0].set_title("IDFT")
axes[2, 0].set_xlabel("Время")
axes[2, 0].set_ylabel("Амплитуда")

axes[2, 1].stem(t, ifft_result, basefmt=" ")
axes[2, 1].set_title("IFFT")
axes[2, 1].set_xlabel("Время")
axes[2, 1].set_ylabel("Амплитуда")

axes[2, 2].stem(t, np_ifft, basefmt=" ")
axes[2, 2].set_title("NumPy IFFT")
axes[2, 2].set_xlabel("Время")
axes[2, 2].set_ylabel("Амплитуда")

# Выравнивание и сохранение
plt.tight_layout()
output_fname = "result.png"
plt.savefig(output_fname)
print(f"Графики сохранены в «{output_fname}». Откройте этот файл, чтобы увидеть результат.")

