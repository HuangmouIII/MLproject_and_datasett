import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings

land_train = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\land_train.csv')
land_test = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\land_test.csv')
# pd.set_option('display.max_columns', None)

# 1.载入并考察形状
print(land_train.shape)
print(land_test.shape)
# print(land_test)
# 可以拼接两个dataframe
Overalldata = pd.concat([land_train,land_test],axis=0)
print(Overalldata.shape)
# 2.考察相应变量分布
X = land_train.iloc[:,1:]
y = land_train.iloc[:,0]
X1 = land_test.iloc[:,1:]
y1 = land_test.iloc[:,0]
# print(y.value_counts())
# sns.catplot(x='Class',kind='count',data=land_train) # 可以直接用
# plt.show()

# 3.进行高斯朴素贝叶斯估计
model = GaussianNB()
model.fit(X,y)
# print(model.score(X,y))

# 4.测试集预测概率并展示前5
prob = model.predict_proba(X1)
# print(prob[:5])

# 5.测试集中预测结果并展示前5
pred = model.predict(X1)
# print(pred[:5])
print('训练集:',model.score(X, y))
print('测试集:',model.score(X1, y1))

# 6.混淆矩阵
table = pd.crosstab(y1,pred,rownames=['Actual'],colnames=['Predicted'])
# print(table)

# 7.Kappa
kappa = cohen_kappa_score(y1,pred)
# print(kappa)

# 8.使用交叉验证评估模型
scores = cross_val_score(GaussianNB(), X, y, cv=10)
# print("Cross-Validation Scores:", scores)
# print("Mean Cross-Validation Score:", scores.mean())

# 与本身划分好的差距不大，故模型较为稳定


