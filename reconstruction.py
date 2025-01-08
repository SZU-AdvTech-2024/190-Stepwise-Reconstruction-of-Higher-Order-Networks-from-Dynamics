from accelerate.commands.menu import cursor

import numpy as np

from numpy import sin
from sklearn.linear_model import Ridge
from sympy.external.tests.test_numpy import array
from tomlkit.items import Array
import models

def g1(xi , xj):
    return sin(xj - xi)

def g2(xi , xj , xk):
    return sin(xj + xk - 2 * xi)

#得到节点动力学
def get_frequencies():
    frequencies = []
    with open('Data/frequencies.dat') as file:
        index = 0
        for line in file:
            frequencies.append(float(line))
            index += 1
    return frequencies, index

def fx(i, frequencies):  #
    omega = frequencies[i]
    return omega

def load_time_series(file_path):
    try:
        # 加载数据文件并存储到二维 NumPy 数组中
        data = np.loadtxt(file_path, delimiter='\t')
        # 检查列数是否符合节点数要求
        print("Successfully loaded time series data.")
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

    return data


def calculate_derivative(data, h):
    """
    计算每列时间序列的导数
    """
    num_rows, num_cols = data.shape
    x_dot = np.zeros_like(data)  # 用于存储导数

    # 对每个节点的时间序列计算五点法导数
    for j in range(num_cols):
        # 对每个时间点 i 应用五点法公式，忽略边界的两个点
        for i in range(2, num_rows - 2):
            x_dot[i, j] = (-data[i + 2, j] + 8 * data[i + 1, j] - 8 * data[i - 1, j] + data[i - 2, j]) / (12 * h)

        # 边界处理，使用简单差分方法计算前两个和后两个点
        x_dot[0, j] = (data[1, j] - data[0, j]) / h
        x_dot[1, j] = (data[2, j] - data[1, j]) / h
        x_dot[-2, j] = (data[-2, j] - data[-3, j]) / h
        x_dot[-1, j] = (data[-1, j] - data[-2, j]) / h

    return x_dot


def infer_three_body_interactions(a2_matrix, sigma2=1, sigma3=0.2, epsilon=0.2):
    """
    根据二体交互矩阵推断三体交互，并考虑二体和三体交互的强度。

    参数:
    - a2_matrix: 二体交互矩阵 (n, n)，其中 n 是节点数。
    - sigma2: 二体交互强度，默认为 1。
    - sigma3: 三体交互强度，默认为 0.2。
    - epsilon: 判断交互强度的阈值，默认为 1e-3。

    返回:
    - three_body_interactions: 三体交互的节点三元组列表，每个元素为 (i, j, k)。
    """
    n = a2_matrix.shape[0]
    three_body_interactions = []

    # 遍历每个节点，推断可能的三体交互
    for i in range(n):
        interacting_nodes = [j for j in range(n) if j != i and abs(a2_matrix[i, j]) > epsilon]  # 强二体交互节点

        # 如果与当前节点 i 有足够多的强交互节点，则考虑三体交互
        if len(interacting_nodes) >= 2:
            for j in range(len(interacting_nodes)):
                for k in range(j + 1, len(interacting_nodes)):
                    # 计算二体交互强度
                    a_ij = abs(a2_matrix[i, interacting_nodes[j]])
                    a_ik = abs(a2_matrix[i, interacting_nodes[k]])
                    a_jk = abs(a2_matrix[interacting_nodes[j], interacting_nodes[k]])

                    # 判断是否存在有效的二体交互和三体交互
                    if a_ij > epsilon and a_ik > epsilon:
                        # 如果 sigma3 为 0，则三体交互必须与二体交互相关
                        if sigma3 == 0:
                            if a_jk > epsilon:  # 只有当三体交互强度足够时才考虑
                                three_body_interactions.append((i, interacting_nodes[j], interacting_nodes[k]))
                        # 当 sigma3 > 0 时，放宽条件，允许一些三体交互独立
                        else:
                            if a_jk > sigma3 * epsilon:  # 弱的三体交互也能被推断
                                three_body_interactions.append((i, interacting_nodes[j], interacting_nodes[k]))

    return three_body_interactions


def calculate_three_body_interactions_per_node(data, three_body_interactions, sigma3=0.2):
    """
    计算每个节点的三体交互矩阵，并应用三体交互强度 σ3。

    参数:
        data - 原始时间序列数据, 尺寸为 (n, m), n 是时间序列长度, m 是节点数
        three_body_interactions - 三体交互的节点三元组列表，每个元素为 (i, j, k)
        sigma3 - 三体交互强度系数，默认值为 0.2

    返回:
        g_list_3 - 每个节点的三体交互矩阵列表，类似于二体交互矩阵列表 g_list_1
    """
    n, m = data.shape
    g_list_3 = [[] for _ in range(m)]  # 初始化每个节点的三体交互矩阵列表

    # 为每个节点生成三体交互矩阵
    for i in range(m):
        # 找到与节点 i 相关的所有三体交互项 (i, j, k)
        node_interactions = [(i, j, k) for (i_, j, k) in three_body_interactions if i_ == i]

        # 初始化节点 i 的三体交互矩阵，大小为 (n, len(node_interactions))
        g_matrix = np.zeros((n, len(node_interactions)))

        # 填入三体交互项的值
        for idx, (i, j, k) in enumerate(node_interactions):
            for t in range(n):
                # 使用 g2 函数计算三体交互项并乘以 sigma3 强度系数
                g_matrix[t, idx] = sigma3 * g2(data[t, i], data[t, j], data[t, k])

        g_list_3[i] = g_matrix  # 将矩阵存入对应的节点位置

    return g_list_3


def merge_interaction_metrices(g_list_1 , g_list_3):
    combined_matrix = np.hstack((g_list_1, g_list_3))
    return combined_matrix


def solve_interaction_matrix(y_dot, g_list_1, alpha=1.0):
    """
    通过求解正则化方程 y_dot = G_list * A_i 构建交互矩阵 A，其中对角线元素为零。

    参数：
    - y_dot: 时间序列导数矩阵，大小为 (n, m)，其中 n 为时间点数，m 为节点数。
    - g_list_1: 二体交互矩阵列表，包含每个节点的 G 矩阵 (n, m-1) 的列表。
    - alpha: 正则化系数（默认值为1.0），用于控制正则化程度。

    返回：
    - a2_matrix: 交互矩阵，大小为 (m, m)，对角线元素为零。
    """
    n, m = y_dot.shape
    a2_matrix = np.zeros((m, m))

    for i in range(m):
        ridge_model = Ridge(alpha=alpha, fit_intercept=False)
        ridge_model.fit(g_list_1[i], y_dot[:, i])
        A_i = ridge_model.coef_

        col = 0
        for j in range(m):
            if i != j:
                a2_matrix[i, j] = A_i[col]
                col += 1

    return a2_matrix


def solve_interaction_matrices(y_dot, G_list, num_nodes, three_body_interactions, alpha=1.0):
    """
    使用正则化从 G_list 和 y_dot 中分别求解二阶和三阶交互矩阵。

    参数：
    - y_dot: 时间序列导数矩阵，大小为 (n, m)，其中 n 是时间点数，m 是节点数。
    - G_list: 包含二阶和三阶交互项的 G 矩阵列表。
    - num_nodes: 节点数 m。
    - three_body_interactions: 三体交互组合的列表。
    - alpha: 正则化系数（默认值为1.0），用于控制正则化程度。

    返回：
    - a2_matrix: 二阶交互矩阵，大小为 (m, m)。
    - a3_matrix: 三阶交互矩阵，大小为 (m, m, m)。
    """
    a2_matrix = np.zeros((num_nodes, num_nodes))
    a3_matrix = np.zeros((num_nodes, num_nodes, num_nodes))

    for i in range(num_nodes):
        ridge_model = Ridge(alpha=alpha, fit_intercept=False)
        ridge_model.fit(G_list[i], y_dot[:, i])
        A_i = ridge_model.coef_

        # 填充二阶交互矩阵
        col = 0
        for j in range(num_nodes):
            if i != j:
                a2_matrix[i, j] = A_i[col]
                col += 1

        # 填充三阶交互矩阵
        for interaction in three_body_interactions:
            if interaction[0] == i:
                _, j, k = interaction
                a3_matrix[i, j, k] = A_i[col]
                a3_matrix[i, k, j] = A_i[col]
                col += 1

    return a2_matrix, a3_matrix

