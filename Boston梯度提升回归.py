import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import PartialDependenceDisplay
from tqdm import tqdm
from colorama import Fore, Style, init

data_url = "http://lib.stat.cmu.edu/datasets/boston"
bar_format = f"{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}"
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

# model = GradientBoostingRegressor(random_state=123)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))    # 0.9214598819200858

param_distributions = {'n_estimators':range(1, 300), 'max_depth':range(1, 10), 'subsample':np.linspace(0.1, 1, 10), 'learning_rate':np.linspace(0.1, 1, 10)}
kfold = KFold(n_splits=10,shuffle=True,random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=123),param_distributions = param_distributions,n_iter=100,random_state=0,n_jobs=-1)
model.fit(X_train,y_train)
print('model.best_params_:',model.best_params_)
model = model.best_estimator_
print('Test_best:',model.score(X_test, y_test))



# PartialDependenceDisplay.from_estimator(model,X_train,['LSTAT','RM'])
# plt.show()

scores = []
for n_estimators in tqdm(range(1,301),desc="Processing", bar_format=bar_format):
    model = GradientBoostingRegressor(n_estimators=n_estimators,subsample=0.5,max_depth=5,learning_rate=0.1,random_state=123)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    scores.append(mse)
index = np.argmin(scores)
print(index)    # 136




