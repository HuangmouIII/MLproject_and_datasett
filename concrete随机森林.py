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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_iris

concrete = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\concrete.csv')
print(concrete)
# print(concrete.shape)

# 1.
X = concrete.iloc[:,:-1]
y = concrete.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=300,random_state=3)
# print(y.value_counts())
# 2.
model = RandomForestRegressor(n_estimators= 100,max_features=3,random_state=123)
model.fit(X_train,y_train)
print(model.score(X_test, y_test))

# 3.
sorted_index = model.feature_importances_.argsort()
# plt.barh(range(X.shape[1]),model.feature_importances_[sorted_index])
# plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
# plt.ylabel('Feature')
# plt.xlabel('Decision Tree')
# plt.tight_layout()
# plt.show()

# 4.
# PartialDependenceDisplay.from_estimator(model,X_train,['Age','Cement'],n_jobs=-1)
# plt.suptitle('Partial Dependence Plots')
# plt.show()


# 5.
pred = model.predict(X_test)
print('MSE:',mean_squared_error(y_test, pred))

# 6.
max_features = range(1,X.shape[1]+1)
param_grid = {'max_features':max_features}
kfold = KFold(n_splits=10,shuffle=True,random_state=123)
model = GridSearchCV(RandomForestRegressor(n_estimators=100,random_state=123),param_grid,cv=kfold,return_train_score=True)
model.fit(X_train,y_train)
print(model.best_params_)
answer = model.cv_results_['mean_test_score']
plt.plot(max_features,answer,'o-')
plt.title('Score vs Max_features')
plt.show()














