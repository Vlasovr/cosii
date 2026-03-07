def convolution(x, y):
    n = len(x)
    m = len(y)

    length = n + m - 1

    # 1. дополняется нулями слева сигнал x до длины N + M − 1;
    padded_x = [0] * (length - n) + x

    # 2. инвертируется во времени сигнал y и дополняется нулями справа до длины N + M − 1;
    y_reversed = y[::-1]
    padded_y = y_reversed + [0] * (length - m)

    result = []
    # 3. в цикле от 0 до N + M − 2 сигнал y сдвигается вправо;
    for shift in range(length):
        if shift % 100 == 0 or shift == length - 1:
            if length > 1:
                percent = (shift / (length - 1)) * 100
            else:
                percent = 100.0
            print(f"\rСвёртка выполнена на: {percent:.2f}%", end="")

        s = 0
        # 4. на каждом шаге цикла вычисляется произведение элементов и подсчитывается сумма произведений.
        for i in range(length - shift):
            s += padded_x[i + shift] * padded_y[i]
        result.append(s)

    print("\nЛинейная свертка сигналов произведена.")

    return result