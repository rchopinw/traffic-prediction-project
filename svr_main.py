from hyper_parameter_configuration import *
from svr_model import *
from genetic_core import *
from data_processing import *
import pandas as pd
from math import log

data_raw = np.array(pd.read_csv('data.csv', encoding='utf_8_sig'))
data_raw = np.reshape(data_raw[:, -1], (-1, 1)).astype('float64')
# For SVR model
# 默认的 fitness 决定方法是 RMSE，若想修改，可以在代码的第28行进行修改。
POP_SIZE = 1000  # 生成的染色体总条数
SUB_POP_RATIO = 0.5  # 每次遗传过程中保留至下一代的比例
CROSS_PROB = 0.5  # 交叉的概率
MUTATE_PROB = 0.5  # 变异的概率
MAX_GEN = floor(log(1 / POP_SIZE) / log(SUB_POP_RATIO))  # 最大遗传代数：由 POP_SIZE 和 SUB_POP_RATIO 决定

if __name__ == '__main__':
    chrom = gen_chrom(pop_size=POP_SIZE, model='svr')
    gen_min_acc = []
    for curnt_gen in range(MAX_GEN):
        chrom_acc = []
        for i, c in enumerate(chrom):
            svr = SVRr(params=c)
            x, x_t, y, y_t = svr.get_xy(data=data_raw, norm=True)
            svr.train(x=x, y=y)
            p = svr.predict(x_t)
            mape, mae, ec, rmse = predict_error(p, y_t)
            chrom_acc.append(mape)
            if i % 100 == 0:
                print('Executing %d of %d' % (i, POP_SIZE))
        print('Current Optimal RMSE: %f' % min(chrom_acc))
        gen_min_acc.append((min(chrom_acc), chrom[np.argmin(chrom_acc)]))
        POP_SIZE = floor(POP_SIZE * SUB_POP_RATIO)
        if POP_SIZE <= 1:
            break
        sub_individuals, sub_individuals_fitness = sub_group_selector(individuals=chrom,
                                                                      individuals_fitness=chrom_acc,
                                                                      pop_size=POP_SIZE)
        l1, l2 = np.array(sub_individuals).shape
        c_sub_individuals = individual_cross(chrom_len=l2,
                                             pop_size=l1,
                                             chrom=sub_individuals,
                                             cross_prob=CROSS_PROB,
                                             model='svr')
        m_c_sub_individuals = individual_mutate(chrom_len=l2,
                                                pop_size=l1,
                                                chrom=c_sub_individuals,
                                                mutate_prob=MUTATE_PROB,
                                                model='svr',
                                                max_gen=MAX_GEN,
                                                curnt_gen=curnt_gen)
        chrom = m_c_sub_individuals
    optimal_args = gen_min_acc[-1][-1]
    optimal_model = SVRr(params=optimal_args)
    x, x_t, y, y_t = optimal_model.get_xy(data=data_raw)
    optimal_model.train(x, y)
    p = optimal_model.predict(x_t)
    # 绘制收敛图
    fig1 = plt.figure(facecolor='white')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    bx = fig1.add_subplot(111)
    bx.plot([x[0] for x in gen_min_acc], color='black', label='MAPE收敛曲线')
    plt.legend()
    plt.savefig('MAPE收敛图')
    plt.show()
    # 预测误差
    print(predict_error(p, y_t))
    # 绘制预测图
    plot_results(p, y_t, '网格搜索预测图')
    pd.DataFrame({'y_test': y_t.tolist(), 'y_predict': p.tolist()}).to_csv('predict_data_svr.csv',
                                                                           encoding='utf_8_sig', index=False)





assert two_number_sum([1, 3], 5) == []
