import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

Glass = pd.read_csv(r"C:\Users\hyn\Desktop\ML\datasett\Glass.csv")
# 大小
# print(Glass.shape)
# 前10个数据
# print(Glass.head(10))
# 每种的个数
# print(Glass.Type.value_counts())
# 画柱状图
# sns.displot(Glass.Type,kde=False)
# plt.show()
# 箱型图
# sns.boxplot(x = 'Type',y='Mg',data=Glass,palette = "Blues")
# plt.show()
# 数据矩阵和响应向量
X = Glass.iloc[:,:-1]
y = Glass.iloc[:,-1]
# print(X)
# print(y)
# 分集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)

model = LogisticRegression(solver='newton-cg',C = 1e10,max_iter=1000)
model.fit(X_train,y_train)

# 迭代次数
# print(model.n_iter_)

# 截距
# print(model.intercept_)

# 系数
# print(model.coef_)

# 准确率
# print(model.score(X_test, y_test))

# 每种的分类概率
prob = model.predict_proba(X_test)
# print(prob[:3])

# 前几个的预测
pred = model.predict(X_test)
# print(pred[:5])

# 混淆矩阵
# table = confusion_matrix(y_test,pred)
# print(table)

# Crosstable可以更加好展示
table = pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['Predicted'])
# print(table)

# 热图
sns.heatmap(table,cmap='Blues',annot=True)
plt.show()

# 详细的预测效果指标
print(classification_report(y_test,pred))

# kappa指标
print(cohen_kappa_score(y_test, pred))



