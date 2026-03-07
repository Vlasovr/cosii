from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def pad_to_power_of_two(values: Iterable[complex | float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.complex128)
    target = next_power_of_two(len(arr))
    if len(arr) == target:
        return arr.copy()
    padded = np.zeros(target, dtype=np.complex128)
    padded[: len(arr)] = arr
    return padded


def bit_reverse_index(index: int, bits: int) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (index & 1)
        index >>= 1
    return result


def bit_reverse_permutation(values: np.ndarray) -> np.ndarray:
    n = len(values)
    bits = int(math.log2(n))
    out = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        out[bit_reverse_index(i, bits)] = values[i]
    return out


def _fft_dif_recursive(values: np.ndarray, direction: int) -> np.ndarray:
    n = len(values)
    if n == 1:
        return values

    half = n // 2
    angle = -2.0 * math.pi * direction / n
    omega_n = complex(math.cos(angle), math.sin(angle))
    omega = 1.0 + 0.0j

    for k in range(half):
        top = values[k]
        bottom = values[k + half]
        values[k] = top + bottom
        values[k + half] = (top - bottom) * omega
        omega *= omega_n

    first_half = _fft_dif_recursive(values[:half], direction)
    second_half = _fft_dif_recursive(values[half:], direction)
    return np.concatenate((first_half, second_half))


def fft_dif(signal: Iterable[complex | float]) -> np.ndarray:
    prepared = pad_to_power_of_two(signal)
    raw = _fft_dif_recursive(prepared.copy(), direction=1)
    return bit_reverse_permutation(raw)


def ifft_dif(spectrum: Iterable[complex | float]) -> np.ndarray:
    spectrum_np = np.asarray(list(spectrum), dtype=np.complex128)
    n = len(spectrum_np)
    conj = np.conj(spectrum_np)
    raw = fft_dif(conj)
    return np.conj(raw) / n

