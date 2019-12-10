import math
import numpy as np
import matplotlib.pyplot as plt

x_a = -1  # крайняя левая точка интервала
x_b = 1   # крайняя правая точка интервала
k = 2     # правая точка интервала
n = 1     # определяем частей для рабития интервала [a;b]


def main():
    # a_f = lambda x: math.sin(k * x) # третье задание
    # b_f = lambda x: math.cos(k * x) # третье задание
    # a_f = lambda x: math.sin(k) # второе задание
    # b_f = lambda x: math.cos(k) # второе задание
    # a_f = lambda x: 1 # первое задание
    # b_f = lambda x: 1 # первое задание
    system_solve(n, lambda x=1: 1, lambda x=1: 1)                          # Первое (для первого уравнениния а = 1. Так как нет а и b, то и чтобы избавиться от а (фотка 2: подчёркнуто))
    system_solve(n, lambda x: math.sin(k), lambda x: math.cos(k))          # Второе (объявление а=sin(k) и b=cos(k) 
    system_solve(n, lambda x: math.sin(k * x), lambda x: math.cos(k * x))  # Третье (объявление а=sin(k*х) и b=cos(k*х)
    plt.show()


def system_solve(n, a_f, b_f): # решение системы
    for i in range(1, 10):     # кол-во графиков, которые выводятся
        n = n + 2              # увеличивем n для более точного решение
        h = (x_b - x_a) / n    # вычисляем h
        matrix, vector, X1 = system_of_equations_builder(n, h, a_f, b_f) # строим матрицу
        Y1 = solve(matrix, vector, n)                                    # решаем матицу
        plt.plot(X1, Y1)                                                 # вывод графиков
        #print(matrix)
        #print(vector)
    plt.show()
    # matrix, vector, X2 = system_of_equations_builder(n * 2, h / 2, a_f, b_f)
    # Y2 = solve(matrix, vector, n * 2)

    # diff = 0
    # for i in range(n + 1):
    #    diff = max(abs(Y1[i] - Y2[2*i]), diff)

    # print(diff)



def system_of_equations_builder(n, h, a_f, b_f):    # функция построения системы
    xi_list = [x_a + (h * i) for i in range(n + 1)] # список x (на которые мы разбивали интервал [a,b]
    matrix = np.zeros((n - 1, n - 1))               # Матрица с системой (инициализация пустой трёхдиагональной матрицы)
    vector = np.linspace(-h * h, -h * h, n - 1)     # Вектор ответов (вектор столбец, значения которыен находятся за знаком равно)(свободный вектор)

    for i in range(n - 1): # проходим по матрице, исключая известные y0 и yn (фотка №1)
        line = np.linspace(0, 0, n - 1) # создаём вектор (для строки матрицы)
        a = a_f(xi_list[i + 1])
        b = b_f(xi_list[i + 1])
        if i == 0:
            line[i:i + 2] = np.array([-(2 * a - (1 + b * (xi_list[i + 1] ** 2)) * h ** 2), a]) # частный случай (когда мы не учитываем первый(y0) (фотка 1)
        elif i == (n - 2):
            line[i - 1:i + 1] = np.array([a, -(2 * a - (1 + b * (xi_list[i + 1] ** 2)) * h ** 2)]) # частный случай (когда мы не учитываем последний столбец (yn) (фотка 1)
        else:
            line[i - 1:i + 2] = np.array([a, -(2 * a - (1 + b * (xi_list[i + 1] ** 2)) * h ** 2), a]) # (фотка 2:общий случай - подчёркнуто)
        matrix[i, :] = line

    return matrix, vector, xi_list


def solve(matrix, vector, n): # решение системы уравнения (матрицы)
    result = np.zeros(n + 1)  # библиотека numpy (создаём вектор из нулей - строку матрицы)
    result[1: n] = np.linalg.solve(matrix, vector) # заполняем решение системы (от 1, а не от 0 так как потому что y0 и yn равны 0 (фотка 1)
    return result


if __name__ == "__main__":
    main()
