import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor,export_text
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LinearRegression

# 数据链接
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
# 由于没法直接在sklearn下载，则需要分割响应变量
X = boston.iloc[:,:-1]
y = boston.iloc[:,-1]
column_names_data = column_names[:-1]
# print(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(y.value_counts())
# 对于一个简单深度为2的无惩罚决策树
model = DecisionTreeRegressor(max_depth=2,random_state=123)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
# print(export_text(model,feature_names=list(column_names_data)))

plot_tree(model,feature_names=column_names_data,node_ids=True,rounded=True,precision=2)
plt.show()

# 根据path.ccp_alphas的增加，由于对模型的惩罚，逐渐不纯度也在增加
model = DecisionTreeRegressor(random_state=123)
path = model.cost_complexity_pruning_path(X_train,y_train)
#
# plt.plot(path.ccp_alphas,path.impurities,marker = 'o',drawstyle = 'steps-post')
# plt.xlabel('alpha (ccp)')
# plt.ylabel('Total Leaf MSE')
# plt.title('Total Leaf MSE vs alpha for Training Set')
# plt.show()

param_grid = {'ccp_alpha':path.ccp_alphas}
kfold = KFold(n_splits=10,shuffle=True,random_state=1)

model = GridSearchCV(DecisionTreeRegressor(random_state=123),param_grid,cv=kfold)
model.fit(X_train,y_train)
# print(model.best_params_)
model = model.best_estimator_
# print(model.score(X_test, y_test))
# plot_tree(model,feature_names=column_names,node_ids=True,rounded=True,precision=2)
# plt.show()
# print('深度:',model.get_depth())
# print('叶节点数目:',model.get_n_leaves())

# sorted_index = model.feature_importances_.argsort()
# plt.barh(range(13),model.feature_importances_[sorted_index])
# plt.yticks(np.arange(13),X.columns[sorted_index])
# plt.ylabel('Feature')
# plt.xlabel('Decision Tree')
# plt.tight_layout()
# plt.show()
#
# pred = model.predict(X_test)
# plt.scatter(pred,y_test,alpha=0.6)
# w =np.linspace(min(pred),max(pred),100)
# plt.plot(w,w)
# plt.show()

# # 决策树略优于线性回归
# model = LinearRegression().fit(X_train,y_train)    # 0.673382550640016
# print(model.score(X_test, y_test))







