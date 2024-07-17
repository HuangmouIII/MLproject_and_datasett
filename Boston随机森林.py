import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor,export_text
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from colorama import Fore, Style, init
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

bar_format = f"{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}"

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
X = boston.iloc[:,:-1]
y = boston.iloc[:,-1]
column_names_data = column_names[:-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
# model = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=123),n_estimators=2000,oob_score=True,random_state=0,n_jobs=-1)
# model.fit(X_train,y_train)
# pred_oob = model.oob_prediction_
# print('池外均方误差：',mean_squared_error(y_train, pred_oob))
# print('池外准确率：',model.oob_score_)
# print('测试集拟合优度：',model.score(X_test, y_test))
#
# model = LinearRegression().fit(X_train,y_train)
# print('线性测试集拟合优度：',model.score(X_test, y_test))

# oob_errors = []
# for n_estimators in tqdm(range(100,301),desc="Processing", bar_format=bar_format):
#     model = BaggingRegressor(estimator=D ecisionTreeRegressor(random_state=123),n_estimators=n_estimators,n_jobs=-1,oob_score=True,random_state=0)
#     model.fit(X_train,y_train)
#     pred_oob = model.oob_prediction_
#     oob_errors.append(mean_squared_error(y_train,pred_oob))
#
# plt.plot(range(100,301),oob_errors)
# plt.xlabel('Number of Trees')
# plt.ylabel('OOB Error')
# plt.title('Bagging OOB Errors')
# plt.show()

# max_features = int(X_train.shape[1]/3)
# model = RandomForestRegressor(n_estimators=500,max_features = max_features,random_state=0)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))
#
# pred = model.predict(X_test)
# plt.scatter(pred,y_test,alpha=0.6)
# w = np.linspace(min(pred),max(pred),100)
# plt.plot(w,w)
# plt.xlabel('pred')
# plt.ylabel('y_test')
# plt.title('Random Forest Prediction')
# plt.show()

# sorted_index = model.feature_importances_.argsort()
# print(model.feature_importances_[sorted_index])
# print(X.columns[sorted_index])
# plt.barh(range(X.shape[1]),model.feature_importances_[sorted_index])
# plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
# plt.ylabel('Feature')
# plt.xlabel('Decision Tree')
# plt.tight_layout()
# plt.show()

# 查看前两个重要特征对于房价的影响
# fig, ax = plt.subplots(figsize=(12, 6))
# PartialDependenceDisplay.from_estimator(model, X_train,features=['LSTAT','RM'], ax=ax, n_jobs=-1)
# plt.suptitle('Partial Dependence Plots')
# plt.show()

# scores = []
# for max_features in tqdm(range(1,X.shape[1]+1),desc='进程',bar_format=bar_format):
#     model = RandomForestRegressor(max_features=max_features,n_estimators=500,random_state=123)
#     model.fit(X_train,y_train)
#     scores.append(model.score(X_test,y_test))
# index = np.argmax(scores)
# print(range(1, X.shape[1] + 1)[index])
#
# plt.plot(range(1,X.shape[1]+1),scores,'o-')
# plt.axvline(range(1,X.shape[1]+1)[index],linestyle = '--',color='k',linewidth = 1)
# plt.xlabel('max_features')
# plt.ylabel('R2')
# plt.title('Choose max_features via Test Set')
# plt.show()

scores_rf = []
for n_estimators in tqdm(range(1,301),desc="RF", bar_format=bar_format):
    model = RandomForestRegressor(max_features=9,n_estimators=n_estimators,random_state=123,n_jobs=-1)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    scores_rf.append(mse)
scores_bag = []
for n_estimators in tqdm(range(1,301),desc="Bagging", bar_format=bar_format):
    model = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=123), n_estimators=n_estimators, random_state=0,n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    scores_bag.append(mse)

model = DecisionTreeRegressor()
path = model.cost_complexity_pruning_path(X_train,y_train)
param_grid = {'ccp_alpha':path.ccp_alphas}
kfold = KFold(n_splits=10,shuffle=True,random_state=1)
model = GridSearchCV(DecisionTreeRegressor(random_state=123),param_grid,cv=kfold,scoring='neg_mean_squared_error')
model.fit(X_train,y_train)
score_tree = -model.score(X_test,y_test)
print(score_tree)
scores_tree = [score_tree for i in range(1,301)]

plt.plot(range(1,301),scores_tree,'k--',label='Single Tree')
plt.plot(range(1,301),scores_bag,'k-',label='Bagging')
plt.plot(range(1,301),scores_rf,'b-',label='Random Forest')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')
plt.title('Test Error')
plt.legend()
plt.show()



