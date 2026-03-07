import numpy as np

def convolution_numpy(x, y):
    return np.convolve(x, y)

def correlation_numpy(x, y):
    return np.correlate(x, y, mode="full")

def fft_numpy(signal):
    return np.fft.fft(signal)

def ifft_numpy(spectrum):
    return np.fft.ifft(spectrum)
