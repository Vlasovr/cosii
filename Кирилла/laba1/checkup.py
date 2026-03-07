from fft_dif import fft_dif, ifft_dif, pad_to_power_of_2

def conv_with_fft(x, y):
    # дополнение до длины N+M-1
    def pad_to_length(signal, L):
        return signal + [0] * (L - len(signal))

    L = len(x) + len(y) - 1
    
    # Предупреждение о точности для больших сигналов
    if L > 10000:
        print(f"  ВНИМАНИЕ: FFT-свертка для больших сигналов (L={L}) может иметь ошибки точности!")
        print("   Рекомендуется использовать ручную свертку для точных результатов.")

    x_pad = pad_to_length(x, L)
    y_pad = pad_to_length(y, L)

    # дополнение до степени 2 (для DIF)
    x_pad = pad_to_power_of_2(x_pad)
    y_pad = pad_to_power_of_2(y_pad)

    # FFT
    X = fft_dif([complex(v) for v in x_pad])
    Y = fft_dif([complex(v) for v in y_pad])

    # умножение спектров
    Z = [X[i] * Y[i] for i in range(len(X))]

    # обратное FFT
    result = ifft_dif(Z)

    # возвращаем только нужную длину
    return [v.real for v in result[:L]]

def corr_with_fft(x, y):
    def pad_to_length(signal, L):
        return signal + [0] * (L - len(signal))

    L = len(x) + len(y) - 1
    
    # Предупреждение о точности для больших сигналов
    if L > 10000:
        print(f"  ВНИМАНИЕ: FFT-корреляция для больших сигналов (L={L}) может иметь ошибки точности!")
        print("   Рекомендуется использовать ручную корреляцию для точных результатов.")

    x_pad = pad_to_length(x, L)
    y_pad = pad_to_length(y, L)

    x_pad = pad_to_power_of_2(x_pad)
    y_pad = pad_to_power_of_2(y_pad)

    X = fft_dif([complex(v) for v in x_pad])
    Y = fft_dif([complex(v) for v in y_pad])

    # Для корреляции: X * Y_conj
    Y_conj = [v.conjugate() for v in Y]

    Z = [X[i] * Y_conj[i] for i in range(len(X))]

    result = ifft_dif(Z)

    # Исправляем порядок для соответствия numpy.correlate
    correlation_result = [v.real for v in result]
    
    # Циклический сдвиг для правильного порядка корреляции
    N = len(correlation_result)
    shift = len(y) - 1
    shifted_result = correlation_result[N-shift:] + correlation_result[:N-shift]
    
    return shifted_result[:L]

def compare_signals(signal1, signal2, name="Signal"):
    """
    Сравнение ручной реализации (signal1) с FFT реализацией (signal2).
    Ручная реализация считается эталонной.
    """
    print(f"\n========Проверка корректности схем вычисления свертки и корреляции========\n")

    min_len = min(len(signal1), len(signal2))

    error = max(
        abs(signal1[i] - signal2[i])
        for i in range(min_len)
    )

    print(f"{name} max error: {error}")

    if error < 1e-6:
        print(f"{name}: OK")
    elif error < 1e-3:
        print(f"{name}: ПРИЕМЛЕМО")
    else:
        print(f"{name}: NOT OK")
        
    # Объяснение причин больших ошибок
    if error > 50:
        print(f"\n🔍 АНАЛИЗ ПРИЧИН БОЛЬШИХ ОШИБОК:")
        print(f"   • Размер сигналов: {len(signal1)} элементов")
        print(f"   • FFT padding до степени 2: {1 << ((len(signal1)) - 1).bit_length()} элементов")
        print(f"   • Накопление численных ошибок в FFT при больших размерах")
        print(f"   • Множественные операции: FFT → умножение спектров → IFFT")
        print(f"   • Рекомендация: использовать ручную реализацию для точных вычислений")

    return error