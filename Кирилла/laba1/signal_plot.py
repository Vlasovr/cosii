import matplotlib.pyplot as plt


def plot_time_signal(signal, sample_rate, filename, title="Signal", max_samples=2000):
    signal = signal[:max_samples]

    # время в секундах
    time = [i / sample_rate for i in range(len(signal))]

    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(f"{title} — Time Domain")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.savefig(filename, dpi=300)
    plt.close()