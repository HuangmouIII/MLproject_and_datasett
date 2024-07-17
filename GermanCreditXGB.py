import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
from sklearn.metrics import mean_squared_error
#
# german = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\GermanCredit.csv')
#
# # 1.
# print(german.head(5))
# print(german.shape)
#
# # 2.
# X = german.iloc[:,:-1]
# y = german.iloc[:,-1]
#
# X = pd.get_dummies(X)
# d = {'bad':1,'good':0}
# y = y.map(d)
# print(y)
# # 3.
# X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=300,random_state=0)
#
# # 4.
# model = xgb.XGBClassifier(random_state=123)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, mean_squared_error
import xgboost as xgb

# 读取数据
german = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\GermanCredit.csv')

# 数据预览
print(german.head(5))
print(german.shape)

# 特征和标签
X = german.iloc[:, :-1]
y = german.iloc[:, -1]


X = pd.get_dummies(X)

# 清理特征名称
X.columns = [col.replace(' ', '_').replace('[', '').replace(']', '').replace('<', '') for col in X.columns]

# 标签编码
d = {'bad': 1, 'good': 0}
y = y.map(d)
print(y)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=300, random_state=0)

# 模型训练
model = xgb.XGBClassifier(random_state=123)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Cohen Kappa Score:", cohen_kappa_score(y_test, y_pred))

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(random_state=123), param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
print("Best Model Accuracy:", best_model.score(X_test, y_test))












