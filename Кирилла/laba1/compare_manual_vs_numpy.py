import time
import numpy as np


def measure_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start


def max_error(a, b):
    n = min(len(a), len(b))
    return max(abs(a[i] - b[i]) for i in range(n))


def compare_manual_vs_numpy(
    x,
    y,
    manual_fft,
    manual_ifft,
    manual_conv,
    manual_corr,
):
    print("\n=== MANUAL vs NUMPY COMPARISON ===\n")

    # --- FFT ---
    x_complex = [complex(v) for v in x]

    X_manual, t_fft_manual = measure_time(manual_fft, x_complex.copy())
    X_numpy, t_fft_numpy = measure_time(np.fft.fft, x)

    fft_error = max_error(X_manual, X_numpy)

    # --- IFFT ---
    x_rec_manual, t_ifft_manual = measure_time(manual_ifft, X_manual.copy())
    x_rec_numpy, t_ifft_numpy = measure_time(np.fft.ifft, X_numpy)

    ifft_error = max_error(
        [v.real for v in x_rec_manual],
        np.real(x_rec_numpy)
    )

    # --- Convolution ---
    conv_manual, t_conv_manual = measure_time(manual_conv, x, y)
    conv_numpy, t_conv_numpy = measure_time(np.convolve, x, y)

    conv_error = max_error(conv_manual, conv_numpy)

    # --- Correlation ---
    corr_manual, t_corr_manual = measure_time(manual_corr, x, y)
    corr_numpy, t_corr_numpy = measure_time(np.correlate, x, y, "full")

    corr_error = max_error(corr_manual, corr_numpy)

    # --- Таблица ---
    print("{:<15} {:<15} {:<15} {:<15}".format(
        "Operation", "Manual time", "NumPy time", "Max error"
    ))
    print("-" * 60)

    print("{:<15} {:<15.6f} {:<15.6f} {:<15.6e}".format(
        "FFT", t_fft_manual, t_fft_numpy, fft_error
    ))

    print("{:<15} {:<15.6f} {:<15.6f} {:<15.6e}".format(
        "IFFT", t_ifft_manual, t_ifft_numpy, ifft_error
    ))

    print("{:<15} {:<15.6f} {:<15.6f} {:<15.6e}".format(
        "Convolution", t_conv_manual, t_conv_numpy, conv_error
    ))

    print("{:<15} {:<15.6f} {:<15.6f} {:<15.6e}".format(
        "Correlation", t_corr_manual, t_corr_numpy, corr_error
    ))

    print("\nLower time → faster algorithm")
    print("Error ≈ 0 → implementations match\n")