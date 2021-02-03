import numpy as np
import matplotlib.pyplot as plt
from hyper_parameter_configuration import *
from math import floor
from random import choices, sample, random, randint, uniform, choice


# 用于绘制
def plot_results(predicted_data, true_data, picture_name):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    ax.plot(predicted_data, label='Prediction')
    plt.xlabel('data number')
    plt.ylabel('traffic flow')
    plt.legend()
    plt.savefig(picture_name)
    plt.show()


# 预测性能指标计算
def predict_error(predict_data: np.array,
                  true_data: np.array) -> tuple:
    """
    :param predict_data: 预测结果
    :param true_data: 真实结果
    :return: MAPE, MAE, RMSE, EC, predict_q_2, true_q_2 等指标：tuple格式
    """
    n = len(true_data)
    MAPE = sum(abs(predict_data - true_data) / true_data) / n
    MAE = sum(abs(predict_data - true_data)) / n
    RMSE = sum((predict_data - true_data) ** 2)
    predict_q_2 = sum(predict_data ** 2)
    true_q_2 = sum(true_data ** 2)
    EC = 1 - (RMSE ** 0.5) / (predict_q_2 ** 0.5 + true_q_2 ** 0.5)
    RMSE = (RMSE / n) ** 0.5

    return MAPE, MAE, EC, RMSE


# 子种群选择
def sub_group_selector(individuals: list,
                       individuals_fitness: list,
                       pop_size: int,
                       reverse_parameter: float = 10.0) -> tuple:
    """
    :param reverse_parameter: 将individuals_fitness映射成选择概率时的参数
    :param individuals: 每个个体染色体编码矩阵
    :param individuals_fitness: 每个个体的fitness度，越小表示fitness度越好
    :param pop_size: 希望选取的子种群的大小
    :return: sub_individuals, sub_individuals_fitness
    """
    n = len(individuals_fitness)
    select_prob = [reverse_parameter/x for x in individuals_fitness]

    sum_sp = sum(select_prob)
    norm_select_prob = [x/sum_sp for x in select_prob]

    selected_index = choices(range(n), weights=norm_select_prob, k=pop_size)

    selected_individuals = [individuals[x] for x in selected_index]
    selected_individuals_fitness = [individuals_fitness[x] for x in selected_index]

    return selected_individuals, selected_individuals_fitness


def individual_cross(chrom_len: int,
                     pop_size: int,
                     chrom: list,
                     cross_prob: float,
                     model: str) -> np.array:
    """
    :param model: 模型选择从 ['svr', 'lstm']
    :param chrom_len: 单条染色体长度
    :param pop_size: 染色体条数
    :param chrom: 染色体数据矩阵
    :param cross_prob: 发生交叉概率
    :return: 交叉完成后的染色体数据矩阵
    """
    for _ in range(pop_size):
        if random() > cross_prob:
            continue
        idx1, idx2 = sample(range(pop_size), 2)
        c1, c2 = chrom[idx1], chrom[idx2]
        while True:
            exchange_pos = randint(0, chrom_len-1)
            exchange_portion = random()

            v1, v2 = c1[exchange_pos], c2[exchange_pos]

            chrom[idx1][exchange_pos] = exchange_portion * v2 + (1 - exchange_portion) * v1
            chrom[idx2][exchange_pos] = exchange_portion * v1 + (1 - exchange_portion) * v2

            if param_bound[model][exchange_pos][-1] == 'int':
                chrom[idx1][exchange_pos] = floor(chrom[idx1][exchange_pos])
                chrom[idx2][exchange_pos] = floor(chrom[idx2][exchange_pos])

            f1 = range_test(chrom, idx1, exchange_pos, model)
            f2 = range_test(chrom, idx2, exchange_pos, model)

            if f1 and f2:
                break

    return chrom


def individual_mutate(chrom_len: int,
                      pop_size: int,
                      chrom: list,
                      mutate_prob: float,
                      model: str,
                      max_gen: int,
                      curnt_gen: int) -> np.array:
    """
    :param mp: 是否自定义变异程度，默认否，自动采样于0-1均匀分布
    :param model: 模型选择从 ['svr', 'lstm']
    :param chrom_len: 单条染色体长度
    :param pop_size: 染色体条数
    :param chrom: 染色体数据矩阵
    :param mutate_prob: 发生交叉概率
    :return: 交叉完成后的染色体数据矩阵
    :param max_gen: 最大迭代次数
    :param curnt_gen: 当前迭代次数
    :return: 变异后的染色体数据矩阵
    """
    for _ in range(pop_size):
        if random() > mutate_prob:
            continue
        m_idx = choice(range(pop_size))
        while True:
            mutate_pos = randint(0, chrom_len-1)
            mutate_portion = (1 - curnt_gen / max_gen) ** 2
            v = chrom[m_idx][mutate_pos]
            v_min, v_max, v_type = param_bound[model][mutate_pos]
            l = [v + mutate_portion * (v_max - v) * random(), v - mutate_portion * (v - v_min) * random()][randint(0, 1)]
            if v_type == 'int':
                chrom[m_idx][mutate_pos] = floor(l)
            f = range_test(chrom, m_idx, mutate_pos, model)
            if f:
                break

    return chrom


def range_test(chrom: list,
               r: int,
               c: int,
               model: str) -> bool:
    """
    :param model: 模型选择从 ['svr', 'lstm']
    :param chrom: 染色体数据矩阵
    :param r: 行
    :param c: 列
    :return: 是否属于正确区间
    """
    v = chrom[r][c]
    v_min, v_max, v_type = param_bound[model][c]
    if v_type == 'int':
        return v in range(v_min, v_max + 2)
    elif v_type == 'float':
        return (v - v_min) * (v - v_max) < 0
    else:
        print("Unidentified Data Type from ['int', 'float'].")
        return False


def gen_chrom(pop_size: int,
              model: str) -> list:
    """
    :param pop_size: 染色体条数
    :param model: 模型选择从 ['lstm', 'svr']
    :return: 生成的染色体数据矩阵：二维list格式
    """
    chrom = []
    for _ in range(pop_size):
        chrom_n = []
        for k in param_bound[model]:
            l ,u, t = param_bound[model][k]
            if t == 'int':
                # if k == 1 and model == 'lstm':
                #     chrom_n.append([randint(1, u)] * chrom_n[0])
                # else:
                chrom_n.append(randint(l, u))
            else:
                chrom_n.append(uniform(l, u))
        chrom.append(chrom_n)
    return chrom


