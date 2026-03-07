from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


INT16_SCALE = 32767


def normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal.astype(np.float64)
    return (signal / peak).astype(np.float64)


def save_wav_mono_16(path: Path, sample_rate: int, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize(signal)
    int_data = np.clip(normalized * INT16_SCALE, -INT16_SCALE, INT16_SCALE).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_data.tobytes())


def read_wav_mono_16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Поддерживается только 16-битный PCM, получен {sample_width * 8}-битный формат.")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / INT16_SCALE

    if channels > 1:
        data = data[::channels]

    return sample_rate, data


def generate_periodic_signals(sample_rate: int, duration_sec: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration_sec, 1.0 / sample_rate)

    signal_x = (
        0.70 * np.sin(2 * np.pi * 220 * t)
        + 0.40 * np.sin(2 * np.pi * 440 * t + 0.3)
        + 0.20 * np.cos(2 * np.pi * 110 * t + 1.0)
    )
    signal_y = (
        0.80 * np.sin(2 * np.pi * 262 * t + 0.1)
        + 0.35 * np.cos(2 * np.pi * 392 * t + 0.8)
        + 0.15 * np.sin(2 * np.pi * 524 * t)
    )

    return normalize(signal_x), normalize(signal_y)
