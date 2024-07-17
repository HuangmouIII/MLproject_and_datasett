import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
#
# Glass1 = pd.read_excel(r"C:\Users\hyn\Desktop\ML\K线导出_JM2409_1分钟数据的副本.xls")
# Glass = pd.DataFrame(Glass1)
#
# scaler = MinMaxScaler()
# Glass['成交额'] = scaler.fit_transform(Glass[['成交额']])
# Glass['成交量'] = scaler.fit_transform(Glass[['成交量']])
# print(Glass.corr())
# X = Glass.iloc[:,:-1]
# y = Glass.iloc[:,-1]
# print(X)
# print(y)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
# model = LogisticRegression(solver='newton-cg',C = 1e10,max_iter=1000)
# model.fit(X_train,y_train)
#
#
# # 迭代次数
# print(model.n_iter_)
# # 截距
# print(model.intercept_)
# # 系数
# print(model.coef_)
# # 准确率
# print(model.score(X_test, y_test))
#
#

A = [1,3,4,5,6,0,8,9,0]
index_min = np.argmin(A)
print(index_min,A[index_min])

