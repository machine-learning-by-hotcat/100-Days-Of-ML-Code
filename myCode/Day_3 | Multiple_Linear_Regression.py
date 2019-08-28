#DAY 3 MULTIPLE LINEAR REGRESSION

#Step1: 数据预处理
import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv('../datasets/50_Startups.csv')
#print(df)

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lEncoder = LabelEncoder()
oEncoder = OneHotEncoder(categorical_features=[3])

X[:,3] = lEncoder.fit_transform(X[:, 3])
X = oEncoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#print(X)

#Step2: 训练模型
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg = reg.fit(X_train, Y_train)



#Step3: 预测结果
Y_pred = reg.predict(X_test)
print(Y_pred)

#Step4: 可视化