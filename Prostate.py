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
# 读取数据
prostate = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\prostate.csv')
# print(prostate.shape)

# 由于mean和std相差较大，这里标准化数据，lpsa为相应变量，位于最后一列
X_raw = prostate.iloc[:,:-1]
y = prostate.iloc[:,-1]

# 这里进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
# print(np.mean(X,axis=0))
# print(np.std(X,axis=0))

# 进行岭回归
model = Ridge()
model.fit(X,y)
# print(model.score(X, y))

# 展示系数
A = pd.DataFrame(model.coef_,index=X_raw.columns,columns=['Coefficient'])
# print(A)

# # 对于不同alpha
# alphas = np.logspace(-3,6,100)
# coefs = []
# for alpha in alphas:
#     model = Ridge(alpha = alpha)
#     model.fit(X,y)
#     coefs.append(model.coef_)

# # 定义ax为当前画轴
# ax = plt.gca()
# ax.plot(alphas,coefs)
# ax.set_xscale('log')
# plt.xlabel('alpha(log scale)')
# plt.ylabel('Coefficients')
# plt.title('Ridge Cofficient Path')
# plt.axhline(0,linestyle='--',linewidth=1,color='k')
# plt.legend(X_raw.columns)
# plt.show()

# 这里发现alpha在6.6左右，所以后面取1到10，找到精确最优位置
# model = RidgeCV(alphas=alphas)
# model.fit(X,y)
# print(model.alpha_)

# 1到10
alphas = np.linspace(1,10,1000)
model = RidgeCV(alphas = alphas,store_cv_results=True)
model.fit(X,y)
# print(model.alpha_)
# print(model.cv_results_.shape)

# 找到最小mse
mse = np.mean(model.cv_results_,axis=0)
# print(np.min(mse))
# print(mse.shape)

# 用argmin找到最小值的索引
index_min = np.argmin(mse)
# print(index_min)

# 用索引找到其alpha和最小的mse
# print(alphas[index_min],mse[index_min])   # 6.018018018018018 0.5363247513368606

# 画图显示是最低位置
# plt.plot(alphas,mse)
# plt.axvline(alphas[index_min],linestyle='--',linewidth=1,color='k')
# plt.xlabel('alpha')
# plt.ylabel('Mean Squared Error')
# plt.title('CV Error for Ridge Regression')
# plt.tight_layout()
# plt.show()

# 展示系数
# print(model.coef_)

# B = pd.DataFrame(model.coef_,index=X_raw.columns,columns=['Coefficient'])
# print(B)

# 下面用10折交叉验证
# kfold = KFold(n_splits=10,shuffle=True,random_state=1)
# model = RidgeCV(alphas=np.linspace(1,10,1000),cv=kfold)
# model.fit(X,y)
# print(model.alpha_)  # 3.2522522522522523 跟前面不同是因为10折里只有9折参与训练

# Lasso L1
model= Lasso(alpha=0.1)
model.fit(X,y)
# print(model.score(X, y))
# print(pd.DataFrame(model.coef_, index=X_raw.columns, columns=['Coefficient']))

alphas,coefs,_=lasso_path(X,y,eps=1e-4)
# print(alphas.shape)
# print(coefs.shape)

# 从图中可以看出Lasso的变量依次变为0，可以由此筛选变量
# ax = plt.gca()
# ax.plot(alphas,coefs.T)
# ax.set_xscale('log')
# plt.xlabel('alpha(log scale)')
# plt.ylabel('Coefficients')
# plt.title('Lasso Coefficient Path')
# plt.axhline(0,linestyle='--',linewidth=1,color='k')
# plt.legend(X_raw.columns)
# plt.show()

# 交叉验证取最优alpha
# kfold = KFold(n_splits=10,shuffle=True,random_state=1)
# alphas = np.logspace(-4,-2,100)
# model=LassoCV(alphas=alphas,cv=kfold)
# model.fit(X,y)
# print(model.alpha_)

# 找到最优的mse   100*10 ，100是alpha的尝试个数，10是10折内每一次得到的mse
# mse = np.mean(model.mse_path_,axis=1)
# index_min = np.argmin(mse)
# print(alphas[index_min],index_min)

# alphas = np.logspace(-4,-2,100)
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
# print(alphas[index_min])
# print(mse[index_min])

# model = ElasticNet(alpha=0.1,l1_ratio=0.5)
# model.fit(X,y)
# print(model.score(X, y))

# 用enet_path直接输出，不需要model
# alphas,coefs,_ = enet_path(X,y,eps=1e-4,l1_ratio=0.5)
# print(alphas.shape,coefs.shape)
# ax = plt.gca()
# ax.plot(alphas,coefs.T)
# ax.set_xscale('log')
# plt.xlabel('alpha(log scale)')
# plt.ylabel('Coefficients')
# plt.title('Elastic Net Coefficient Path')
# plt.axhline(0,linestyle='--',linewidth=1,color='k')
# plt.legend(X_raw.columns)
# plt.show()

# 这里使用弹性网查看参数  alpha和l1_ratio_
# alphas = np.logspace(-4,0,100)
# kfold = KFold(n_splits=10,shuffle=True,random_state=1)
# model = ElasticNetCV(cv=kfold,alphas=alphas,l1_ratio=[0.0001,0.001,0.01,0.1,0.5,1])
# model.fit(X,y)
# print(model.alpha_)
# print(model.l1_ratio_)

# 这里用73随机抽样
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
model = RidgeCV(alphas=np.linspace(1,20,1000))
model.fit(X_train,y_train)
print(model.alpha_)
print(model.score(X_train, y_train)) # 0.6760210841041492
print(model.score(X_test, y_test)) # 0.45586215215050363
# 所以测试集表现并不好 R^2在测试集中只有0.456









