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

bar_format = f"{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}"

sonar = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\Sonar.csv')

# print(sonar.shape)
# print(sonar)

X = sonar.iloc[:,:-1]
y = sonar.iloc[:,-1]

# sns.heatmap(X.corr(),cmap='Blues')
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=50,random_state=1)

# # 先考察单棵树
# model = DecisionTreeClassifier()
# path = model.cost_complexity_pruning_path(X_train,y_train)
# param_grid = {'ccp_alpha':path.ccp_alphas}
# kfold = KFold(n_splits=10,shuffle=True,random_state=1)
# model = GridSearchCV(DecisionTreeClassifier(random_state=123),param_grid,cv=kfold)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))    # 0.74
#
# model = LogisticRegression(C = 1e10,max_iter=500)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))    # 0.7
#
# model = RandomForestClassifier(n_estimators=500,max_features='sqrt',random_state=123)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))    # 0.78

# y_train_dummy = pd.get_dummies(y_train)
# y_train_dummy = y_train_dummy.iloc[:,1]
# # print(y_train_dummy)
# param_grid = {'max_features':range(1,11)}
# kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
# model = GridSearchCV(RandomForestClassifier(n_estimators=500,random_state=123),param_grid,cv=kfold)
# model.fit(X_train,y_train_dummy)
# print(model.best_params_)

# model= RandomForestClassifier(n_estimators=500,max_features=9,random_state=123)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))
#
# PartialDependenceDisplay.from_estimator(model,X_train,features = ['V11'])
# plt.show()
# pred = model.predict(X_test)
# table = pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['Predicted'])
# print(table)
#
# RocCurveDisplay.from_estimator(model,X_test,y_test)
# x = np.linspace(0,1,100)
# plt.plot(x,x,'k--',linewidth=1)
# plt.title('ROC Curve for Random Forest')
# plt.show()

X , y = load_iris(return_X_y=True)
X2 = X[:,2:4]

model = RandomForestClassifier(n_estimators=500,max_features=1,random_state=1)
model.fit(X2,y)
print(model.score(X2, y))
plot_decision_regions(X2,y,model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for Random Forest')
plt.show()











