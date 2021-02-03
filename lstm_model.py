from data_processing import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


# This file implements the construction of a LSTM model
class LSTMr(object):
    def __init__(self,
                 params: list):
        """
        :param params: 生成的某条染色体
        """
        self.params_transform = {0: 'num_layer', 1: 'num_hunit', 2: 'dropout', 3: 'q', 4: 's'}
        self.lstm_params_idx = [0, 1, 2]
        self.data_params_idx = [3, 4]
        self.params = params
        self.lstm_params = dict(zip([self.params_transform[x] for x in self.lstm_params_idx],
                                    [self.params[x] for x in self.lstm_params_idx]))
        self.data_params = dict(zip([self.params_transform[x] for x in self.data_params_idx],
                                    [self.params[x] for x in self.data_params_idx]))
        self.m = self.build_model()

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
        x, y = data_reconstruct(data, q, s, -1, panel=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, x_test, y_train, y_test

    def build_model(self):
        layer_num, units, dropout = [self.lstm_params[x] for x in ['num_layer', 'num_hunit', 'dropout']]

        model = Sequential()

        if layer_num - 1:
            for i in range(layer_num - 1):
                model.add(LSTM(units=units, return_sequences=True))
                model.add(Dropout(dropout))

        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout))

        model.add(Dense(units=1))
        model.add(Activation("linear"))

        model.compile(loss='mse', optimizer='rmsprop')

        return model

    def train(self,
              x: np.array,
              y: np.array,
              epochs: int = 20,
              batch_size: int = 128,
              verbose: int = 1):
        """
        :param x: 训练数据x
        :param y: 训练数据y
        :param epochs: 迭代次数
        :param batch_size: mini batch大小
        :param verbose: 输出进度条记录
        """
        self.m.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self,
                x: np.array) -> np.array:
        """
        :param x: 预测集x
        :return: 预测的y
        """
        y_predict = self.m.predict(x)
        y_predict = np.reshape(y_predict, (1, -1))[0]
        return y_predict
