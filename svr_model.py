from sklearn.svm import SVR
from data_processing import *
from sklearn.model_selection import train_test_split
import numpy as np


# This file implements the construction of a SVR model
class SVRr(object):
    def __init__(self,
                 params: list):
        """
        :param params: 生成的染色体
        """
        self.params_transform = {0: 'C', 1: 'epsilon', 2: 'gamma', 3: 'q', 4: 's'}
        self.svr_params_idx = [0, 1, 2]
        self.data_params_idx = [3, 4]
        self.params = params
        self.svr_params = dict(zip([self.params_transform[x] for x in self.svr_params_idx],
                                   [self.params[x] for x in self.svr_params_idx]))
        self.data_params = dict(zip([self.params_transform[x] for x in self.data_params_idx],
                                    [self.params[x] for x in self.data_params_idx]))
        self.m = SVR(**self.svr_params)

    def get_xy(self,
               data: np.array,
               test_size: float = 0.25,
               norm: bool = False) -> tuple:
        """
        :param test_size: 测试集占比，默认1/4
        :param norm: 是否标准化，默认True
        :param data: 原始数据
        :return: 训练集以及测试集
        """
        if norm:
            data = data_scaler(data)
        q, s = self.data_params['q'], self.data_params['s']
        x, y = data_reconstruct(data, q, s, -1, panel=False)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, x_test, y_train, y_test

    def train(self,
              x: np.array,
              y: np.array):
        """
        :param x: 用于训练的x数据
        :param y: 用于训练的y数据
        """
        self.m.fit(x, y)

    def predict(self,
                x: np.array) -> np.array:
        """
        :param x: 用于预测的x数据
        :return: 预测结果
        """
        y_predict = self.m.predict(x)
        return y_predict





