import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

data_url = "http://lib.stat.cmu.edu/datasets/boston"
#region
# 读取数据
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=21, header=None)

# 重新整理数据，每两行合并成一行
data = []
for i in range(0, len(raw_df), 2):
    row = pd.concat([raw_df.iloc[i], raw_df.iloc[i+1]], ignore_index=True)
    data.append(row)
# 创建新的DataFrame
processed_df = pd.DataFrame(data)
# 删除完全是NaN的列
processed_df = processed_df.dropna(axis=1, how='all')
# 列名
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
processed_df.columns = column_names[:processed_df.shape[1]]
boston = processed_df
#endregion

X,y =make_blobs(n_samples=40,centers=2,n_features=2,random_state=6)
y = 2*y-1
data = pd.DataFrame(X,columns=['x1','x2'])
# sns.scatterplot(x='x1',y='x2',data=data,hue=y,palette=['blue','black'])
# plt.show()
# 惩罚力度很大
model = LinearSVC(C=1000,loss='hinge',random_state=123)
model.fit(X,y)
# print(model.get_params())
# dist = model.decision_function(X)
# index = np.where(y * dist <= (1+1e-10))
# print(index)

def support_vectors(model,X,y):
    dist = model.decision_function(X)
    index = np.where(y*dist<=(1+1e-10))
    return X[index]

def svm_plot(model,X,y):
    data = pd.DataFrame(X,columns=['x1','x2'])
    data['y']=y
    sns.scatterplot(x='x1',y='x2',data=data,s=30,hue=y,palette=['blue','black'])
    s_vectors = support_vectors(model,X,y)
    plt.scatter(s_vectors[:,0],s_vectors[:,1],s=100,linewidths=1,facecolors='none',edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx,yy = np.meshgrid(np.linspace(xlim[0],xlim[1],50),np.linspace(ylim[0],ylim[1],50))
    Z = model.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx,yy,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
    plt.show()

# model = LinearSVC(C=0.1,loss="hinge",random_state=123,max_iter=1000)
# model.fit(X,y)
#
# # print(svm_plot(model, X, y))
#
# param_grid = {'C':[0.01,0.01,0.1,1,10,100]}
# kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
# model = GridSearchCV(LinearSVC(loss="hinge",random_state=123,max_iter=10000),param_grid,cv=kfold)
# model.fit(X,y)
# print(model.best_params_)
# model = model.best_estimator_
# svm_plot(model,X,y)
#
# X_test,y_test = make_blobs(n_samples=1040,centers=2,n_features=2,random_state=6)
# y_test=2*y_test-1
# X_test = X_test[40:,:]
# y_test = y_test[40:]
# print(model.score(X_test, y_test))
# pred = model.predict(X_test)
# print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))
#
# model = SVC(kernal='linear',C=0.01,random_state=123)
# model.fit(X,y)

np.random.seed(1)
X=np.random.randn(200,2)
y = np.logical_xor(X[:,0]>0,X[:,1]>0)
y = np.where(y,1,-1)
data = pd.DataFrame(X,columns=['x1','x2'])

# sns.scatterplot(x = 'x1',y='x2',data=data,hue=y,palette=['blue','black'])
# model = SVC(kernel='rbf',C=1,gamma=0.5,random_state=123)
# model.fit(X,y)

# plot_decision_regions(X,y,model,hide_spines=False)
# plt.title('SVM (C=1,gamma=0.5)')
# print(model.score(X, y))
# plt.show()
#
# model = SVC(kernel='rbf',C=10000,gamma=0.5,random_state=123)
# model.fit(X,y)
# plot_decision_regions(X,y,model,hide_spines=False)
# plt.show()
#
# model = SVC(kernel='rbf',C=0.01,gamma=0.5,random_state=123)
# model.fit(X,y)
# plot_decision_regions(X,y,model,hide_spines=False)
# plt.show()

param_grid = {'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100]}
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
model = GridSearchCV(SVC(kernel='rbf',random_state=123),param_grid,cv=kfold)
model.fit(X,y)
print(model.best_params_)
plot_decision_regions(X,y,model,hide_spines=False)
plt.show()

np.random.seed(369)
X_test = np.random.randn(1000,2)
y_test = np.logical_xor(X_test[:,0]>0,X_test[:,1]>0)
y_test = np.where(y_test,1,-1)
print(model.score(X_test, y_test))
pred = model.predict(X_test)
print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))








