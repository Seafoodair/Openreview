import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
# print(boston.target.shape)
# print(boston.data.shape)
# print(boston.feature_names)
import pandas as pd
from numpy import genfromtxt
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\reason.csv', delimiter=',')#这个是自己改的
#bos = pd.DataFrame(boston.data)
#X = bos1[1:,1:-1]#不断的修改这个东西就可以0换成1等等 这个评审人level的
#X = bos1[1:,0:-1]#总体的
#X = bos1[1:,0:-2]#写作水平
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\arxiv.csv', delimiter=',')
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\lacklevel.csv', delimiter=',')#这个缺level
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\lacknovel.csv', delimiter=',')#这个缺创新
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\lackmotivation.csv', delimiter=',')#这个缺动机
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\lackexperiment.csv', delimiter=',')
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\lackreadable.csv', delimiter=',')
#bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\lackrelatework.csv', delimiter=',')
# X=bos2[1:,0:-1]
bos1= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\Rebuttal2.csv', delimiter=',')
# X=bos2[1:,0:-1]
# bos2= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\因果4.csv', delimiter=',')#这个实验
# X=bos2[1:,0:-1]
#bos2= genfromtxt('E:\\超级账本教程\\Openreview论文修改\\因果5.csv', delimiter=',')#这个相关工作
X=bos1[1:,0:-1]
# print(X)
#Y = pd.DataFrame(boston.target)
Y=bos1[1:,-1]
# print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)
from sklearn.preprocessing import StandardScaler
X_test= StandardScaler().fit_transform(X_test)
from sklearn.preprocessing import StandardScaler
X_train= StandardScaler().fit_transform(X_train)
#这快是主干--------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
t1=0
pp=2500
for bignum in range(pp):
    #f = open('116.txt', encoding='utf-8', mode='w')
    lm = SGDRegressor(eta0=0.01,learning_rate='constant',shuffle=False,max_iter=5000,tol=0.001)
    lm.partial_fit(X_train, Y_train)
    Y_pred = lm.predict(X_test)
    #print("Actual Y's: \n {}".format(np.array(Y_test[0])))
    #print("Predicted Y's: \n {}".format(np.array(Y_pred)))

    delta_y = np.array(Y_test[0]) - np.array(Y_pred)#这里是误差
    mse=0
    for i in delta_y:
        mse+=i*i
    #print(delta_y.shape[0])
    t=mse/delta_y.shape[0]
    #print(t)
    t1+=t

    #print("测评标准MSE from SGDRegressor: {}".format(mse/delta_y.shape[0]))#均方误差（这里考虑均方误差越小越好）
    # import seaborn as sns
    # sns.set_style('whitegrid')
    # sns.kdeplot(np.array(Y_test[0]), bw=0.5)
    # plt.show()
    # sns.set_style('whitegrid')
    # sns.kdeplot(Y_pred, bw=0.5)
    # plt.show()
    #print("权重大小为SGDRegressor weights: \n{}".format(lm.coef_))
    #print("b为",lm.intercept_)
    #print(mse)
print("均值误差为：",t1/pp)
