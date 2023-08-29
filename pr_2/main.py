import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

def z_1():
    a = int(input('Введите число: '))
    b = a
    c = 0 + abs(a ** 2)

    while b != 0:
        a = int(input('Введите следующее число: '))
        b = b + a
        print('Сумма ваших чисел: %d' % b)
        c = 0 + abs(a) ** 2
        if b == 0:
            break
    print('Квадрат всех считанных чисел: ', c)


def z_2():
    list, n = [], int(input('Введите число: '))
    for i in range(n):
        count = 0
        if n == 1:
            print(n)
            break
        while count < i + 1:
            list.append(i + 1)
            count += 1
            if len(list) == n:
                print(*list)
                break


def z_3():
    columns = int(input('Введите количество столбцов матрицы: '))
    rows = int(input('Введите количество строк матрицы: '))
    RC = columns * rows

    A = np.arange(RC).reshape(rows, columns)
    B = []

    for i in range(rows):
        for j in range(columns):
            B.append(A[i, j])

    print('Исходная матрица\n', A)
    print('Результат\n', B)


def z_4():
    A = [1, 2, 3, 4, 2, 1, 3, 4, 5, 6, 5, 4, 3, 2]
    B = ['a', 'b', 'c', 'c', 'c', 'b', 'a', 'c', 'a', 'a', 'b', 'c', 'b', 'a']
    T1 = {}


    for i in range(len(A)):
        if i > 3:
            C = T1.get(B[i]) + A[i]
            T1.update([(B[i], C)])
        else:
            T1.update([(B[i], A[i])])
    print(T1)

def z_5_11():
    data = fetch_california_housing(as_frame=True)
    print("---TYPE---\n", type(data))
    print("---KEYS---\n", data.keys())
    print("---DATA---\n", data.data)
    print("---CONCAT---\n", pd.concat([data["data"], data['target']], axis=1))
    print("---INFO---")
    data["data"].info()
    print("---ISNA.SUM---\n", data.data.isna().sum())
    print("---LOC---\n", data.data.loc[(data.data.HouseAge > 50) & (data.data.Population > 2500)])
    print("---MEDINC.MAX---\n", data.data.MedInc.max)
    print("---MEDINC.MIN---\n", data.data.MedInc.min)
    print("---APPLY---\n", data.data.apply(np.mean, axis=0))


if __name__ == '__main__':
    z = int(input('Выберите задание: '))
    if z == 1:
        z_1()
    elif z == 2:
        z_2()
    elif z == 3:
        z_3()
    elif z == 4:
        z_4()
    elif z == 5 or z == 6 or z == 7 or z == 8 or z == 9 or z == 10 or z==11:
        z_5_11()
