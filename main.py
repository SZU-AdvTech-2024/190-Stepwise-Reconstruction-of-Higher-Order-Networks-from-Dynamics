import numpy as np
from accelerate.commands.menu import cursor
from matplotlib import pyplot as plt

import reconstruction
import numpy as np

from simulate import simulate


def calculate_f1(A, A_true, epsilon=1e-5):
    """
    计算给定矩阵 A 和 A_true 之间的 F1 分数、精度和召回率。
    """
    A_binary = (np.abs(A) > epsilon).astype(int)
    A_true_binary = (np.abs(A_true) > epsilon).astype(int)

    TP = np.sum((A_binary == 1) & (A_true_binary == 1))
    FP = np.sum((A_binary == 1) & (A_true_binary == 0))
    FN = np.sum((A_binary == 0) & (A_true_binary == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score, precision, recall


def plot_fig():
    node_counts = [20, 30, 40, 50, 60]
    stepwise_f1_scores = []
    direct_f1_scores = []

    for n in node_counts:
        # Stepwise Reconstruction
        simulate("kuramoto1", n, (int)(n * 0.8), 30, 1)
        A_stepwise, _ = stepwise_reconstruction()
        A_true = np.loadtxt('Data/connectivity.dat')
        f1_stepwise, _, _ = calculate_f1(A_stepwise, A_true)
        stepwise_f1_scores.append(f1_stepwise)

        # Direct Reconstruction
        simulate("kuramoto1", n, 2, 30, 1)
        A_direct = direct_reconstruction()
        A_true = np.loadtxt('Data/connectivity.dat')
        f1_direct, _, _ = calculate_f1(A_direct, A_true)
        direct_f1_scores.append(f1_direct)

    plt.figure(figsize=(8, 6))
    plt.plot(node_counts, stepwise_f1_scores, linestyle='-', color='b', label="Stepwise Reconstruction F1", linewidth=2)
    plt.scatter(node_counts, stepwise_f1_scores, color='b', s=100, zorder=5)

    plt.plot(node_counts, direct_f1_scores, linestyle='-', color='r', label="Direct Reconstruction F1", linewidth=2)
    plt.scatter(node_counts, direct_f1_scores, color='r', s=100, zorder=5)

    plt.xlabel('Node Count')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Node Count')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()



#


# 执行绘图函数

def direct_reconstruction():
    # Step 1: 获取频率数组和节点数量
    frequency_array, node_count = reconstruction.get_frequencies()

    # Step 2: 计算所有节点的自动动力学项 f
    dynamics_list = [reconstruction.fx(i, frequency_array) for i in range(node_count)]

    # Step 3: 加载时间序列数据并计算每个节点的导数 x_dot
    data = reconstruction.load_time_series('Data/data.dat')
    x_dot = reconstruction.calculate_derivative(data, 1)  # 计算导数

    # Step 4: 计算 y_dot
    y_dot = x_dot - np.array(dynamics_list).reshape(1, -1)

    # Step 5: 初始化二阶交互矩阵，并确保是一个方阵
    interaction_matrices = np.zeros((node_count, node_count))  # 初始化为方阵

    for i in range(node_count):
        # 初始化节点 i 的 G 矩阵，用于存储二阶交互项
        G_matrix = np.zeros((data.shape[0], node_count - 1))  # 二阶交互项列数为 node_count - 1

        # 填充二体交互项
        col = 0
        for j in range(node_count):
            if i != j:
                G_matrix[:, col] = [reconstruction.g1(data[t][i], data[t][j]) for t in range(data.shape[0])]
                col += 1

        # 求解二阶交互系数矩阵 A_i
        A_i = np.linalg.lstsq(G_matrix, y_dot[:, i], rcond=None)[0]

        # 将 A_i 的二阶交互项存储在 interaction_matrices 的对应位置
        col = 0
        for j in range(node_count):
            if i != j:
                interaction_matrices[i, j] = A_i[col]
                col += 1

    # 将对角线元素设置为零
    np.fill_diagonal(interaction_matrices, 0)

    print("\nTwo-body interaction matrix A with zero diagonal:")
    print(interaction_matrices)

    return interaction_matrices

"""
def stepwise_reconstruction(sigma2=1, sigma3=0.2):
    # Step 1: 初始化自动动力学项和导数的空列表
    dynamics_list = []

    # Step 2: 获取频率数组和节点数量
    frequency_array, node_count = reconstruction.get_frequencies()  # 获取频率数组和节点数

    for i in range(node_count):
        dynamics_list.append(reconstruction.fx(i, frequency_array))  # 计算每个节点的自动力学项

    # Step 3: 加载时间序列数据并计算每个节点的导数
    data = reconstruction.load_time_series('Data/data.dat')
    x_dot = reconstruction.calculate_derivative(data, 1)  # 计算每个节点的导数

    # Step 4: 初始化 y_dot
    y_dot = x_dot - np.array(dynamics_list).reshape(1, -1)

    # 初始化 G 矩阵列表，用于存储二体交互项
    g_list_1 = []
    for i in range(node_count):
        g_matrix = np.zeros((data.shape[0], node_count - 1))
        col = 0
        for j in range(node_count):
            if i != j:
                # 计算二体交互项并乘以 sigma2
                g_matrix[:, col] = [sigma2 * reconstruction.g1(data[t][i], data[t][j]) for t in range(data.shape[0])]
                col += 1
        g_list_1.append(g_matrix)

    # Step 5: 计算初始二体交互矩阵 a2_matrix
    a2_matrix = reconstruction.solve_interaction_matrix(y_dot, g_list_1)

    for t in range(30):
        # 获取三体交互列表，根据 sigma3 调节阈值
        three_body_interactions = reconstruction.infer_three_body_interactions(a2_matrix, epsilon=sigma3 * 1e-1)

        # 计算每个节点的三体交互矩阵，并乘以 sigma3 调节强度
        g_list_3 = reconstruction.calculate_three_body_interactions_per_node(data, three_body_interactions, sigma3=sigma3)

        # 初始化最终交互矩阵列表，包含二体和三体交互
        combined_g_list = [np.hstack((g_list_1[i], g_list_3[i])) for i in range(node_count)]

        # 更新 a2_matrix 和 a3_matrix，得到二体和三体交互矩阵
        a2_matrix, a3_matrix = reconstruction.solve_interaction_matrices(y_dot, combined_g_list, node_count, three_body_interactions)

    #print("\nInteraction matrix A (with zeros on the diagonal):")
    #print("\na2_matrix_updated:\n", a2_matrix)
    #print("\na3_matrix:\n", a3_matrix)

    return a2_matrix, a3_matrix


"""
def stepwise_reconstruction():
    # Step 1: 初始化自动力学项 f 和导数 y_dot 的空列表
    f = []  # 存储每个节点的自动力学

    # Step 2: 获取频率数组和节点数量 index
    array, index = reconstruction.get_frequencies()  # 这里假设有函数获取频率和节点数量

    for i in range(index):
        f.append(reconstruction.fx(i, array))  # 计算每个节点 i 的自动力学项并存储到 f 中
    #print("动力学f：" , f)

    # Step 3: 加载时间序列数据并计算每个节点的导数
    data = reconstruction.load_time_series('Data/data.dat')  # 加载时间序列数据
    #print("\n时间序列:" ,data)
    x_dot = reconstruction.calculate_derivative(data, 1)  # 计算每个节点的导数 x_dot
    #print("\nx导数：" , x_dot)

    #Step 4: 初始化G矩阵和y_dot，逐步重构网络
    y_dot = np.zeros((len(data), index))
    for i in range(len(data)):
        y_dot = x_dot - f
    #print("\nY:",y_dot)

    n, m = data.shape  # 获取时间序列长度 n 和节点数 m
    g_list_1 = []  # 初始化耦合矩阵列表

    for i in range(m):
        # 为节点 i 创建一个 n × (m-1) 的耦合矩阵
        g_matrix = np.zeros((n, m - 1))  # m-1 表示与其他 m-1 个节点的耦合关系

        for t in range(n):
            col = 0  # 列索引，用于存储节点 i 与其他节点的耦合
            for j in range(m):
                if i != j:
                    g_matrix[t, col] = reconstruction.g1(data[t][i], data[t][j])  # 计算节点 i 与 j 的耦合
                    col += 1

        g_list_1.append(g_matrix)
    #print("\nG:" , g_list_1)

    a2_matrix = reconstruction.solve_interaction_matrix(y_dot, g_list_1)

    #print("\nA_hat:\n",a2_matrix)

    for t in range(30):
        three_body_interactions = reconstruction.infer_three_body_interactions(a2_matrix) #三体交互列表
        #print(three_body_interactions)
        #print(three_body_interactions)

        g_list_3 = reconstruction.calculate_three_body_interactions_per_node(data, three_body_interactions)

        # 初始化最终交互矩阵列表
        combined_g_list = []

        # 拼接每个节点的二体和三体交互矩阵
        for i in range(m):
            combined_matrix = np.hstack((g_list_1[i], g_list_3[i]))  # 拼接二体和三体交互矩阵
            combined_g_list.append(combined_matrix)

        #g_list_1 = combined_g_list
        # print("Combined interaction matrices for each node:")
        # print(combined_g_list)
        a2_matrix , a3_matrix = reconstruction.solve_interaction_matrices(y_dot, combined_g_list, index, three_body_interactions)
    print("\nInteraction matrix A (with zeros on the diagonal):")
    print("\na2_matrix_updated:\n" , a2_matrix)
    return a2_matrix , a3_matrix


if __name__ == "__main__":
    plot_fig()
