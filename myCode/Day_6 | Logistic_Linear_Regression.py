#DAY 4 LOGISTIC LINEAR REGRESSION

#Step 1: 数据预处理
import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv('../datasets/Social_Network_Ads.csv')
#print(df)

X = df.iloc[:, [2,3]].values
Y = df.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Step 2: 训练模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train, Y_train)


#Step 3: 预测数据
Y_predict = lr.predict(X_test)

#Step 4: 可视化

#Step 5: 计算准确性
print(lr.score(X_test, Y_test))