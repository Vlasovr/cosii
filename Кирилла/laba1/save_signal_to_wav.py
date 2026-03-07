from scipy.io.wavfile import write, read
import numpy as np

def save_signal_to_wav(new_signal, filename, sample_rate=44100):
    signal = np.array(new_signal, dtype=float)

    # Нормализуем значения которые > 32767 или < -32768, так как .wav формат 16-битный
    signal = signal / np.max(np.abs(signal))

    audio = (signal * 32767).astype(np.int16)
    # filename = "signals_after_correlation.wav"
    write(filename + ".wav", sample_rate, audio)

    print("Сигнал успешно преобразован в .wav файл!")