import numpy as np
import pickle


# 重构数据集结构
def data_reconstruct(data,
                     q: int,
                     s: int,
                     target_col: int = -1,
                     panel: bool = False) -> tuple:
    """
    :param panel: 是否构建成面板数据 [m x n x t]
    :param target_col: 原始数据中，y列所在位置
    :param data: 原始数据，数据格式为 numpy.array
    :param q: 延迟阶数，即对未来数据的预测与以前多长时间的交通流数据有关
    :param s: 预测步长，即预测未来第几个时刻的交通流数据
    :return: 整理完后的数据
    """
    data_x, data_y = [], []  # dataX表示输入变量，dataY表示输出变量
    if panel:
        for i in range(len(data) - q - s + 1):
            data_x.append(data[i:(i + q), :].tolist())
            data_y.append(data[i + q + s - 1, target_col])
        data_x = np.array(data_x)
        data_y = np.array(data_y)
    else:
        for i in range(len(data) - q - s + 1):
            data_x.append(data[i:(i + q), target_col].tolist())
            data_y.append(data[i + q + s - 1, target_col])
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data = np.delete(data[-(len(data) - q - s + 1):, :], target_col, axis=1)
        data_x = np.concatenate((data, data_x), axis=1)
    return data_x, data_y


def data_scaler(data: np.array) -> np.array:
    """
    :param col_to_keep: 需要保留不变的列
    :param data: 原始数据
    :return: 标准化以后的数据
    """
    # 标准化公式参考 https://blog.csdn.net/bbbeoy/article/details/70185798
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
