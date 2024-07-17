import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import  load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

diabetes = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\PimaIndiansDiabetes.csv')

# 1.形状和5个观测值
# print(diabetes.shape)
# print(diabetes.head(5))

# 2.展示统计特征，考察相应变量分布
# print(diabetes.describe())
# print(diabetes.diabetes.value_counts())

# 3.根据diabetes，画变量mass(body mass index)的箱型图
# sns.boxplot(x='mass',y='diabetes',data=diabetes)
# plt.show()

# 4.分层抽样随机抽取200个观测值作为测试集
# 这里是把neg变成0，把pos变成1
d = {'neg':0,'pos':1}
diabetes['diabetes'] = diabetes['diabetes'].map(d)

X = diabetes.iloc[:,:-1]
y = diabetes.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=200,shuffle=True,random_state=0)

# 5.标准化特征变量
scaler = StandardScaler()
scaler.fit(X_train)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# print(np.mean(X_train_s,axis=0))    # 测试是否符合
# print(np.std(X_train_s,axis=0))
# print(np.mean(X_test_s,axis=0))
# print(np.std(X_test_s,axis=0))

# 6.使用KNN(K=10)
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train_s,y_train)

# 7.在测试集中预测且展示混淆矩阵，并计算预测准确率
pred = model.predict(X_test_s)
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
# print(table)
# print(table.values.sum())
# print(table.iloc[0,1])
# print('预测准确率:',model.score(X_test_s,y_test))

# 8.从1到20找到预测准确率最高的K值
# 9.展示K与测试集预测准确率的关系
# 10.展示K与测试集错分率的关系
scores = []
Accuracy = []
Error_Rate = []

for K in range(1,21):
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
    Accuracy.append((table.iloc[0,0]+table.iloc[1,1])/table.values.sum())
    Error_Rate.append((table.iloc[0,1]+table.iloc[1,0])/table.values.sum())
    scores.append(model.score(X_test_s, y_test))

# print('预测位置:',np.argmax(scores))    # 15
# print('预测准确率:',max(scores))    # 0.795
# # print(scores)    # Score就是准确率 (TP+TN)/sum
# # print(Accuracy)
# plt.plot(list(range(1,21)),Accuracy,'o-')
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.title('Accuracy and K')
# plt.tight_layout()
# plt.show()

# plt.plot(list(range(1,21)),Error_Rate,'o-')
# plt.xlabel('K')
# plt.ylabel('Error_Rate')
# plt.title('Error_Rate and K')
# plt.tight_layout()
# plt.show()

# 11.展示1/K与测试集预测错分率的关系
# K = np.array(list(range(1,21)))
# K_inverse = np.divide(1,K)
# plt.plot(K_inverse,Error_Rate,'o-')
# plt.xlabel('1/K')
# plt.ylabel('Error_Rate')
# plt.title('Error_Rate and K')
# plt.tight_layout()
# plt.show()

# 12.训练集做十折交叉验证，选择最优K并在测试集中计算预测准确率
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
param_grid = {'n_neighbors':range(1,21)}
model = GridSearchCV(KNeighborsClassifier(),param_grid,cv=kfold)
model.fit(X_train_s,y_train)

print('最优K:',model.best_params_)
print('最优准确率:',model.score(X_test_s,y_test))
# KNeighborsClassifier(n_neighbors=13)    # 确认13的准确率是对的
# model.fit(X_train_s,y_train)
# print('最优K:',model.score(X_test_s,y_test))














