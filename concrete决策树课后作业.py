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
from sklearn.metrics import mean_squared_error
concrete = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\concrete.csv')
feature_number=len(concrete.columns)-1
# 1.
# print(concrete)
# print(concrete.shape)
# print(concrete.head(5))
# print(concrete.value_counts(['CompressiveStrength']))

# 2.
X = concrete.iloc[:,:-1]
y = concrete.iloc[:,-1]
# print(y.value_counts())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=300,random_state=0)    # 如果后面需要用decisiontree就不要加随机抽样

# 3.
model = DecisionTreeRegressor(max_depth=2,random_state=123)
model.fit(X_train,y_train)
# plot_tree(model, feature_names=X.columns, node_ids=True, proportion=True, rounded=True, precision=2)
# plt.show()
# print(model.score(X_test, y_test))

# 4.
model = DecisionTreeRegressor(random_state=123)
path = model.cost_complexity_pruning_path(X_train,y_train)
kfold = KFold(n_splits=10,shuffle=True,random_state=123)
param_grid = {'ccp_alpha':path.ccp_alphas}
model = GridSearchCV(DecisionTreeRegressor(random_state=123),param_grid,cv=kfold)
model.fit(X_train,y_train)
print('最优alpha:',model.best_params_)
print(model.score(X_test, y_test))

# 5.
model = model.best_estimator_
sorted_index = model.feature_importances_.argsort()
# plt.barh(range(feature_number),model.feature_importances_[sorted_index])
# plt.yticks(np.arange(feature_number),X.columns[sorted_index])
# plt.ylabel('Feature')
# plt.xlabel('Decision Tree')
# plt.tight_layout()
# plt.show()

# 6.
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('决策树MSE:', mse)

# 7.
lmodel = LinearRegression()
lmodel.fit(X_train,y_train)
print('线性R^2',lmodel.score(X_test, y_test))

y_predl = lmodel.predict(X_test)
lmse = mean_squared_error(y_test, y_predl)
print('线性MSE:', lmse)

# 决策树：0.774781251308691，线性回归：0.6278167298105268



