# HYPER-PARAMETERS FOR SVR AND LSTM MODEL:
svr_param_bound = {0: [10**(-5), 10, 'float'],  # C
                   1: [10**(-5), 1, 'float'],  # epsilon
                   2: [10**(-5), 1, 'float'],  # gamma
                   3: [1, 50, 'int'],  # q
                   4: [1, 10, 'int']}  # s

lstm_param_bound = {0: [1, 5, 'int'],  # 隐藏层数
                    1: [20, 50, 'int'],  # 隐藏节点个数
                    2: [0.01, 0.3, 'float'],  # 暂停率
                    3: [10, 50, 'int'],  # 输入长度
                    4: [1, 10, 'int']}  # 预测步长

param_bound = {'svr': svr_param_bound,
               'lstm': lstm_param_bound}