import math

def bit_reverse(n, bits):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

def bit_reverse_array(arr):
    n = len(arr)
    bits = n.bit_length() - 1
    result = [0] * n
    for i in range(n):
        j = bit_reverse(i, bits)
        result[j] = arr[i]
    return result

def _fft_dif_recursive(a, direction=1):
    n = len(a)

    if n == 1:
        return a

    half = n // 2

    # Присвоить ωN значение главного комплексного корня N-й степени из единицы:
    angle = -2 * math.pi * direction / n
    omega_n = complex(math.cos(angle), math.sin(angle))

    omega = 1 + 0j

    # Операция бабочки для DIF FFT
    for j in range(half):
        temp1 = a[j]
        temp2 = a[j + half]

        a[j] = temp1 + temp2
        a[j + half] = (temp1 - temp2) * omega

        omega *= omega_n

    # Рекурсивный вызов для каждой части
    first_half = _fft_dif_recursive(a[:half], direction)
    second_half = _fft_dif_recursive(a[half:], direction)

    # Объединение двух частей
    return first_half + second_half

# Прямое преобразование
def fft_dif(a, direction=1):
    a_copy = a.copy()
    
    result = _fft_dif_recursive(a_copy, direction)
    
    return bit_reverse_array(result)

# Обратное преобразование
def ifft_dif(x):
    n = len(x)
    # Для обратного FFT: сопряжение -> FFT -> сопряжение -> деление на n
    x_conj = [_x.conjugate() for _x in x]
    X = fft_dif(x_conj, direction=1)
    x_restored = [X[i].conjugate() / n for i in range(n)]
    return x_restored

def pad_to_power_of_2(signal):
    n = len(signal)
    size = 1 << (n - 1).bit_length()
    return signal + [0] * (size - n)