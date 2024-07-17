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

glass = pd.read_csv(r"C:\Users\hyn\Desktop\ML\datasett\Glass.csv")

X = glass.iloc[:,:-1]
y = glass.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
# model = GradientBoostingClassifier(random_state=123)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))

param_distributions = {'n_estimators':range(1, 300), 'max_depth':range(1, 10), 'subsample':np.linspace(0.1, 1, 10), 'learning_rate':np.linspace(0.1, 1, 10)}
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=123),param_distributions=param_distributions,cv=kfold,random_state=66)
model.fit(X_train,y_train)
print(model.best_params_)
model = model.best_estimator_
print(model.score(X_test, y_test))     # 0.7384615384615385


# model = GradientBoostingClassifier(random_state=123,subsample=0.6,n_estimators=20,max_depth=4,learning_rate=0.1)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))

sorted_index = model.feature_importances_.argsort()
plt.barh(range(X.shape[1]),model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
plt.ylabel('Feature')
plt.xlabel('Decision Tree')
plt.tight_layout()
plt.show()






