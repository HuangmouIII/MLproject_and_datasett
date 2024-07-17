import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import enet_path
# 1.读取数据
student = pd.read_table(r'C:\Users\hyn\Desktop\ML\datasett\student-mat.csv',sep=';')

# 2.删除G1，G2
student = student.drop(['G1','G2'],axis=1)
X_raw = student.iloc[:,:-1]
X = X_raw
y = student.iloc[:,-1]

# 3.画直方图关于G3
# student.iloc[:,-1].plot.hist(subplots=True)
# plt.show()

# 4.把分类变量变成虚拟变量
X_raw = pd.get_dummies(X_raw)
X = pd.get_dummies(X)
# print(student)

# 5.标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# print(student.shape)
# print(np.std(student, axis=0))

# 6.画岭回归的系数路径
alphas = np.logspace(-3,6,100)
coefs = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X,y)
    coefs.append(model.coef_)

# ax = plt.gca()
# ax.plot(alphas,coefs)
# ax.set_xscale('log')
# plt.xlabel('alpha(log scale)')
# plt.ylabel('Coefficients')
# plt.title('Ridge Cofficient Path')
# plt.axhline(0,linestyle='--',linewidth=1,color='k')
# plt.legend(X_raw.columns)
# plt.show()

# 7.十折交叉检验
kfold = KFold(n_splits=10,shuffle=True,random_state=1)
model = RidgeCV(alphas=alphas,cv = kfold)
model.fit(X,y)
# print(model.alpha_) # 284.8035868435805
# print(model.coef_)
# print(pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient']))

# 8.使用lasso_path()函数，并画出路径
# alphas,coefs,_ = lasso_path(X,y,eps=1e-4)
# ax = plt.gca()
# ax.plot(alphas,coefs.T)
# ax.set_xscale('log')
# plt.xlabel('alpha(log scale)')
# plt.ylabel('Coefficients')
# plt.title('Lasso Coefficient Path')
# plt.axhline(0,linestyle='--',linewidth=1,color='k')
# plt.legend(X_raw.columns)
# plt.show()

# 9.通过10折交叉验证选择最优alpha进行Lasso回归，并展示系数
# alphas = np.logspace(-3,1,100)
# kfold = KFold(n_splits=10,shuffle=True,random_state=1)
# model = LassoCV(alphas=alphas,cv=kfold)
# model.fit(X,y)
# print(model.alpha_)    # 0.1668100537200059
#
# # 这个经常出现问题，原因在于model.mse_path_的程序有误
# # mse = np.mean(model.mse_path_,axis=1)
# # index_min = np.argmin(mse)
# # 面手动验证
# scores = []
# for alpha in alphas:
#     model = Lasso(alpha=alpha)
#     kfold = KFold(n_splits=10,shuffle=True,random_state=1)
#     score_val = -cross_val_score(model,X,y,cv=kfold,scoring='neg_mean_squared_error')
#     score = np.mean(score_val)
#     scores.append(score)
#
# mse = np.array(scores)
# index_min = np.argmin(mse)
# print(alphas[index_min])    # 0.1668100537200059
# print(pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient']))

# 10.使用弹性网回归
# alphas = np.logspace(-3,1,100)
# l1_ratio = [0.001,0.01,0.1,0.5,1]
# kfold = KFold(n_splits=10,shuffle=True,random_state=1)
#
# model = ElasticNetCV(cv=kfold,alphas=alphas,l1_ratio=l1_ratio)
# model.fit(X,y)
#
# print(model.score(X, y))
# print(model.alpha_)    # 0.7390722033525783
# print(model.l1_ratio_)    # 0.001
# print(pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient']))

# 11.预留100个数据作为测试集，并使用最优的弹性网回归，还有岭回归
n_samples = len(X)
test_size = 100 / n_samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

alphas = np.logspace(-3,1,100)
l1_ratio = [0.001,0.01,0.1,0.5,1]
kfold = KFold(n_splits=10,shuffle=True,random_state=0)

Elasticmodel = ElasticNetCV(cv=kfold,alphas=alphas,l1_ratio=l1_ratio)
Elasticmodel.fit(X_train,y_train)

print("Elastic:",Elasticmodel.alpha_)
print("Elastic:",Elasticmodel.l1_ratio_)
print("Elastic:",Elasticmodel.score(X_train, y_train))
print("Elastic:",Elasticmodel.score(X_test, y_test))

model = RidgeCV(cv=kfold,alphas=alphas)
model.fit(X_train,y_train)

# model1 = Ridge(alpha=10)
# model1.fit(X_train,y_train)

# print("Ridge:",model1.score(X_train, y_train))
# print("Ridge:",model1.score(X_test, y_test))

print("Ridge:",model.alpha_)
print("Ridge:",model.score(X_train, y_train))
print("Ridge:",model.score(X_test, y_test))
