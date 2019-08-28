#DAY 2 SIMPLE LINEAR REGRESSION

#Step1: 导入数据

#Step1.1 导入模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step1.2 导入数据
df = pd.read_csv('../datasets/studentscores.csv')
#print(df)

#Step1.3 拆分数据集和训练集
from sklearn.model_selection import train_test_split
X = df.iloc[:, :1].values
Y = df.iloc[:, 1].values
#print(X)
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Step2: 利用回归模型训练数据集
#Step2.1 导入线性训练器
from sklearn.linear_model import LinearRegression

#Step2.2 创建训练器
reg = LinearRegression()


#Step2.3 训练数据
reg = reg.fit(X_train, Y_train)

#Step3: 预测数据
Y_predict = reg.predict(X_test)

#Step4: 可视化
#Step4.1 显示训练集数据
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_test, reg.predict(X_test), color = 'blue')
#plt.show()

#Step4.2 显示测试集数据
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, Y_predict, color = 'blue')
plt.show()