# Лабораторная работа №3. Построение дискретных оптимальных планов эксперимента
# Вариант 6:
# Двухфакторная кубическая модель. [-1;1]. Множество Х - сетки 30х30 и 40х40
# D-оптимальные планы. Алгоритм Митчелла. Повторные наблюдения не допускаются

# %% Импортируем нужные библиотеки

import numpy as np
import matplotlib.pyplot as plt

# %% Обозначим функцию модели, функцию вычисления информационной матрицы


def model(x1, x2):
    return np.array([1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2, x1 ** 2 * x2, x1 * x2 ** 2, x1 ** 3, x2 ** 3])


def calculate_variance(x, info_mat):
    return model(x[0], x[1]) @ np.linalg.inv(info_mat) @ model(x[0], x[1]).T


def calculate_info_mat(factors, weights):
    info_mat_tmp = np.array(
        [np.dot(p, np.vstack(model(x[0], x[1])) @ np.vstack(model(x[0], x[1])).T) for x, p in zip(factors, weights)]
    )
    return np.sum(info_mat_tmp, axis=0)


def D_functional(plan):
    return np.linalg.det(calculate_info_mat(plan['x'], plan['p']))


def draw_plan(x, title):
    plt.scatter(x[0], x[1])
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# %% Строим сетку по заданным характеристикам
N = 40  # число наблюдений
size_web = 40  # размерность сетки
factors_interval_min, factors_interval_max = -1, 1  # область определения модели

# Дискретное множество Х
x1 = np.linspace(factors_interval_min, factors_interval_max, size_web)
x2 = np.linspace(factors_interval_min, factors_interval_max, size_web)

# %% Составим начальный план

weight = 1 / N * np.ones(N)  # Вектор весов

# Реализуем рандомную выборку точек из множества Х для плана размерности N
spectrum = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
random_indexes = np.random.randint(0, size_web**2 - 1, N)

start_plan = {
    'x': np.array([spectrum[ind] for ind in random_indexes]),
    'p': weight,
    'N': N
}

draw_plan(start_plan['x'].T, 'Начальный план')

# %% Реализация алгоритма Митчела
cur_plan = start_plan.copy()
cur_info_mat = calculate_info_mat(cur_plan['x'], cur_plan['p'])

iteration = 0
while True:
    # Выберем точки, не содержащиеся в плане
    x_s = np.array(
        [elem for elem in np.random.permutation(spectrum) if elem not in cur_plan['x']]
    )

    # Найдем значение дисперсии в точках вне плана
    cur_variance_s = np.array(
        [calculate_variance(x, cur_info_mat)
         for x in x_s]
    )

    # Добавим точку с максимальной дисперсией в план и изменим его
    max_variance = np.max(cur_variance_s)
    picked_x_s_index = np.where(cur_variance_s == max_variance)[0]
    picked_x_s_index = picked_x_s_index[:1][0]
    # Перестраиваем план
    tmp = x_s[picked_x_s_index].reshape(1,2)
    cur_plan['x'] = np.append(cur_plan['x'], x_s[picked_x_s_index].reshape(1, 2), axis=0)
    cur_plan['N'] += 1
    cur_plan['p'] = 1 / cur_plan['N'] * np.ones(cur_plan['N'])
    cur_info_mat = calculate_info_mat(cur_plan['x'], cur_plan['p'])

    # Удалим точку с минимальной дисперсией из текущего плана
    x_j = cur_plan['x']
    cur_variance_j = np.array(
        [calculate_variance(x, cur_info_mat)
         for x in x_j]
    )

    # Нашли точки с минимальной дисперсией
    min_variance = np.min(cur_variance_j)
    picked_x_j_index = np.where(cur_variance_j == min_variance)[0]
    picked_x_j_index = picked_x_j_index[:1][0]
    # Перестраиваем план, удаляя точку
    cur_plan['x'] = np.delete(cur_plan['x'], (picked_x_j_index), axis=0)
    cur_plan['N'] -= 1
    cur_plan['p'] = 1 / cur_plan['N'] * np.ones(cur_plan['N'])
    cur_info_mat = calculate_info_mat(
        cur_plan['x'], cur_plan['p'])

    if picked_x_s_index == picked_x_j_index or iteration == 2000:
        break
    iteration += 1

# %% Выведем характеристики полученного плана
end_plan = cur_plan.copy()
draw_plan(end_plan['x'].T, 'Оптимальный план')
print(f'Значение D-функционала начального плана = {D_functional(start_plan)}')
print(f'Значение D-функционала конечного плана = {D_functional(end_plan)}')
print(f'Количество итераций = {iteration}')
