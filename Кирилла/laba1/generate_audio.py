import numpy as np
from scipy.io.wavfile import write

def create_wav_file(amount_of_files, sample_rate, duration):
    result = []

    for i in range(amount_of_files):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = 2 * np.sin(2 * np.pi * 300 * t) + np.cos(2 * np.pi * 200 * t + 1)
        signal = signal / np.max(np.abs(signal))
        audio = (signal * 32767).astype(np.int16)

        filename = "periodic_signal_" + str(i + 1) + ".wav"

        write(filename, sample_rate, audio)
        print("Файл: " + filename + " cоздан!")

        result.append(filename)

    return result