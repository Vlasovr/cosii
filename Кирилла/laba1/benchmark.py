import time
import matplotlib.pyplot as plt

def measure_time(func, *args):
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    return end - start

def benchmark_algorithms(x, y, conv_func, conv_fft_func):
    sizes = [256, 512, 1024, 2048, 4096]

    conv_times = []
    fft_times = []

    for N in sizes:
        print(f"Testing N = {N}")

        x_n = x[:N]
        y_n = y[:N]

        # обычная свёртка
        t1 = measure_time(conv_func, x_n, y_n)
        conv_times.append(t1)

        # FFT свёртка
        t2 = measure_time(conv_fft_func, x_n, y_n)
        fft_times.append(t2)

    return sizes, conv_times, fft_times

def plot_efficiency(sizes, conv_times, fft_times):
    plt.figure(figsize=(8, 5))

    plt.plot(sizes, conv_times, "o-", label="Time-domain convolution O(N²)")
    plt.plot(sizes, fft_times, "o-", label="FFT convolution O(N log N)")

    plt.xlabel("Number of samples (N)")
    plt.ylabel("Execution time (seconds)")
    plt.title("Algorithm Efficiency Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig("algorithm_efficiency.png", dpi=300)
    plt.show()