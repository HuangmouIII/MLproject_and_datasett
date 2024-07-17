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
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions

bank = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\bank-additional.csv',sep=';')
# pd.set_option('display.max_columns', None)
bank = bank.drop('duration',axis=1)

# X_raw = bank.iloc[:,:-1]
# X = pd.get_dummies(X_raw)
# y = bank.iloc[:,-1]
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=1000,random_state=1)
#
# # model = DecisionTreeClassifier(max_depth=2,random_state=123)
# # model.fit(X_train,y_train)
# # print(model.score(X_test, y_test))
# # plot_tree(model,feature_names=X.columns,node_ids=True,rounded=True,precision=2)
# # plt.show()
#
# model = DecisionTreeClassifier(random_state=123)
# path = model.cost_complexity_pruning_path(X_train,y_train)
# param_grid = {'ccp_alpha':path.ccp_alphas}
# kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
# model = GridSearchCV(DecisionTreeClassifier(random_state=123),param_grid,cv=kfold)
# model.fit(X_train,y_train)
# print(model.best_params_)
# model = model.best_estimator_
# print(model.score(X_test, y_test))
#
# # plot_tree(model,feature_names=X.columns,node_ids=True,proportion=True,rounded=True,precision=2)
# # plt.show()
#
# sorted_index = model.feature_importances_.argsort()
# plt.barh(range(X_train.shape[1]),model.feature_importances_[sorted_index])
# plt.yticks(np.arange(X_train.shape[1]),X.columns[sorted_index])
# plt.ylabel('Feature')
# plt.xlabel('Decision Tree')
# plt.tight_layout()
# plt.show()
#
# pred = model.predict(X_test)
# table = pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['Predicted'])
# print(table)
#
# table = np.array(table)
# # 灵敏度太低了，只能识别到22%的有意愿购买的人
# Sensitivity = table[1,1]/(table[1,0]+table[1,1])
# print(Sensitivity)

# 最后使用iris的数据做分类树
# 加载Iris数据集
X, y = load_iris(return_X_y=True)
X2 = X[:, 2:4]
# print(y.shape)
# 初始化决策树分类器
model = DecisionTreeClassifier(random_state=123)

# 获取代价复杂度剪枝路径
path = model.cost_complexity_pruning_path(X2, y)
param_grid = {'ccp_alpha': path.ccp_alphas}

# 设置k折交叉验证
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# 设置GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=kfold)
grid_search.fit(X2, y)

# 输出模型得分
print(grid_search.score(X2, y))    # 0.9933333333333333

# 绘制决策树
plt.figure(figsize=(10, 8))
plot_tree(grid_search.best_estimator_, feature_names=['petal_length', 'petal_width'], node_ids=True, proportion=True, rounded=True, precision=2)
plt.show()

plot_decision_regions(X2,y,grid_search)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for Decision Tree')
plt.show()






