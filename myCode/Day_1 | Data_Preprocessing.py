# DAY 1 DATA PREPROCESSING

#Step 1: import modules
import numpy as np
import pandas as pd

#Step 2: import dataset
df = pd.read_csv('../datasets/Data.csv')
#print(df)

#Step 3: 处理丢失数据
#Step 3.1: 导入sklearn的数据丢失模块
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)

#Step 3.2: 处理df的数据
X_fit = df.iloc[: , :-1].values
Y_fit = df.iloc[: , 3].values

#Step 3.3: 将df数据中的空值剔除
#imputer = imputer.fit(X_fit[: , 1:3])
#X_fit = imputer.transform(X_fit[: , 1:3])
X_fit[:,1:3] = imputer.fit_transform(X_fit[: , 1:3])
#print(X_fit)
#print(Y_fit)

#Step 4: 解析分类数据
#Step 4.1: 导入模组
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
lEncoder = LabelEncoder()
oEncoder = OneHotEncoder(categorical_features=[0])

#Step 4.2: Y数据接入laberEncoder
Y_lEncoder = LabelEncoder()
Y_fit = Y_lEncoder.fit_transform(Y_fit)
#print(Y_fit)

#Step 4.3: X数据写入OneHotEncoder
X_fit[:,0] = lEncoder.fit_transform(X_fit[:,0])
X_fit = oEncoder.fit_transform(X_fit).toarray()
#print(X_fit)

#Step 5: 拆分数据集为测试集合和训练集合
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_fit, Y_fit, test_size = 0.2, random_state = 0)
print('X_train Data:')
print(X_train)

print('X_test Data:')
print(X_test)

print('Y_train Data:')
print(Y_train)

print('Y_test Data:')
print(Y_test)

#Step 6: 特征缩放
#Step 6.1: 导入模块
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Step 6.2: 训练X_train
X_train = scaler.fit_transform(X_train)
print('X_train Data:')
print(X_train)

#Step 6.3: 训练X_test
X_test = scaler.transform(X_test)
print('X_test Data:')
print(X_test)

'''
#Step 6.4: 训练Y_train
Y_train = scaler.fit_transform(Y_train)
print('Y_train Data:')
print(Y_train)

#Step 6.5: 训练Y_test
Y_test = scaler.fit_transform(Y_test)
print('Y_test Data:')
print(Y_test)
'''