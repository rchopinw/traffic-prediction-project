这是交通流时间序列预测模型的说明文档：
一、环境配置
	1、python版本：python 3.7.6
	2、IDE以及版本：pyCharm Community Edition 19.3
	3、pandas版本：1.0.1
	4、numpy版本：1.18.1
	5、tensorflow版本：tensorflow-gpu 2.0.0
	6、keras版本：同tensorflow
	7、sklearn版本：0.22.1
二、代码文件说明
	1、data.csv是数据源文件。
	2、data_processing.py是用于数据处理的函数集合。
	3、genetic_core.py是遗传算法函数集合（包含了 淘汰、交叉、变异、生成、可视化、fitness计算）。
	4、hyper_parameter_configuration.py包含了LSTM和SVR模型的参数以及参数边界。
	5、lstm_main和svr_main是主要的运行文件。
	6、lstm_model和svr_model分别实现了LSTM模型和SVR模型。
三、运行说明
	在lstm_main和svr_main中调整参数运行即可，运行结果会以LSTM_result和SVR_result的pickle文件保存至根目录。