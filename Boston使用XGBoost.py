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
import xgboost as xgb

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
# model = xgb.XGBRegressor(objective = 'reg:squarederror',n_estimators = 300,max_depth=6,subsample = 0.6,colsample_bytree=0.8,learning_rate=0.1,random_state=0)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))    # 0.9045137226664561
# pred = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test,pred))
# print(rmse)

params = {'objective' : 'reg:squarederror','max_depth':6,'subsample' :0.6,'colsample_bytree':0.8,'learning_rate':0.1}
dtrain = xgb.DMatrix(data=X_train,label=y_train)
results = xgb.cv(dtrain = dtrain,params=params,nfold=10,metrics="rmse",num_boost_round=300,as_pandas=True,seed=123)
plt.plot(range(1,301),results['train-rmse-mean'],'k',label = 'Training Error')
plt.plot(range(1,301),results['test-rmse-mean'],'b',label = 'Test Error')
plt.show()

