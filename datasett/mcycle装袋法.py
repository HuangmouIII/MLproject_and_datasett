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
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor


mcycle = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\mcycle.csv')
X = mcycle.iloc[:,0].values.reshape(-1,1)
y = mcycle.iloc[:,1]
# X = np.array(mcycle.times).reshape(-1,1)
# y = mcycle.accel

model = DecisionTreeRegressor(random_state=123)
path = model.cost_complexity_pruning_path(X,y)
param_grid = {'ccp_alpha':path.ccp_alphas}
kfold = KFold(n_splits=10,shuffle=True,random_state=1)
model = GridSearchCV(DecisionTreeRegressor(random_state=123),param_grid,cv=kfold)
pred_tree = model.fit(X,y).predict(X)

sns.scatterplot(x='times',y='accel', data=mcycle)
plt.plot(X,pred_tree,'b')
plt.title('Single Tree Estimation')
plt.show()

model = BaggingRegressor(estimator = DecisionTreeRegressor(random_state=123),n_estimators=500,random_state=0)
pred_bag = model.fit(X,y).predict(X)

sns.scatterplot(x='times',y='accel', data=mcycle,alpha = 0.6)
plt.plot(X,pred_bag,'b')
plt.title('Single Tree Estimation')
plt.show()




