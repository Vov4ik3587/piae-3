# Лабораторная работа №3. Построение дискретных оптимальных планов эксперимента
# Вариант 6:
# Двухфакторная кубическая модель. [-1;1]. Множество Х - сетки 30х30 и 40х40
# D-оптимальные планы. Алгоритм Митчелла. Повторные наблюдения не допускаются

# %% Импортируем нужные библиотеки

import numpy as np


# %% Обозначим функцию модели, функцию вычисления информационной матрицы
def model(x1, x2):
    return np.array([1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2, x1 ** 2 * x2, x1 * x2 ** 2, x1 ** 3, x2 ** 3])


def calculate_variance(x1, x2, info_mat):
    return model(x1, x2) @ np.linalg.inv(info_mat) @ model(x1, x2).T


def calculate_info_mat(factor_1, factor_2 , weights):
    info_mat_tmp = np.array(
        [np.dot(p, np.vstack(model(x1, x2)) @ np.vstack(model(x1, x2)).T)
         for x1, x2, p in zip(factor_1, factor_2, weights)
         ]
    )
    return np.sum(info_mat_tmp, axis=0)


# %% Строим сетку по заданным характеристикам
N = 20  # число наблюдений
size_web = 30  # размерность сетки
factors_interval_min, factors_interval_max = -1, 1  # область определения модели

# Дискретное множество Х
x1 = np.linspace(factors_interval_min, factors_interval_max, size_web)
x2 = np.linspace(factors_interval_min, factors_interval_max, size_web)



# %% Составим начальный план

weight = 1/N * np.ones(N)  # Вектор весов

# Реализуем рандомную выборку точек из множества Х для плана размерности N

spectrum_x1 = np.random.choice(x1, size=N, replace=False)
spectrum_x2 = np.random.choice(x2, size=N, replace=False)


start_plan = {
    'x1': spectrum_x1,
    'x2': spectrum_x2,
    'p': weight,
    'N': N
}

# %% Реализация алгоритма Митчела
cur_plan = start_plan.copy()
cur_info_mat = calculate_info_mat(cur_plan['x1'], cur_plan['x2'], cur_plan['p'])
cur_x1 = x1
cur_x2 = x2
# Выберем точки, не содержащиеся в плане
x1_s = np.array(
    [elem for elem in cur_x1 if elem not in cur_plan['x1']]
)
x2_s = np.array(
    [elem for elem in cur_x2 if elem not in cur_plan['x2']]
)

# Найдем значение дисперсии в точках вне плана
cur_variance = np.array(
    [calculate_variance(x1_s[i], x2_s[i], cur_info_mat) for i in range(len(x1_s))]
)

# Добавим точку с максимальной дисперсией в план
max_variance = np.max(cur_variance)
picked_x1_s_index, picked_x2_s_index = np.where(cur_variance == max_variance), np.where(cur_variance == max_variance)
cur_plan['x1'] = np.append(cur_plan['x1'], x1_s[picked_x1_s_index][0])
cur_plan['x2'] = np.append(cur_plan['x2'], x2_s[picked_x2_s_index][0])
cur_plan['N'] += 1
cur_plan['p'] = 1 / cur_plan['N'] * np.ones(cur_plan['N'])


# %%
