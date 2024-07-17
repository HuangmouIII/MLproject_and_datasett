import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

spam = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\spam.csv')
X = spam.iloc[:,:-1]
y = spam.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1000,stratify=y,random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# model = SVC(kernel='linear',random_state=123)
# model.fit(X_train_s,y_train)
# print("线性核:",model.score(X_test_s, y_test))
#
# model = SVC(kernel='poly',degree=2,random_state=123)
# model.fit(X_train_s,y_train)
# print("多项式核2:",model.score(X_test_s, y_test))
#
# model = SVC(kernel='poly',degree=3,random_state=123)
# model.fit(X_train_s,y_train)
# print("多项式核3:",model.score(X_test_s, y_test))
#
# model = SVC(kernel='rbf',random_state=123)
# model.fit(X_train_s,y_train)
# print("径向核:",model.score(X_test_s, y_test))
#
# model = SVC(kernel='sigmoid',random_state=123)
# model.fit(X_train_s,y_train)
# print("S型核:",model.score(X_test_s, y_test))

param_grid = {'C':[0.1,1,10],'gamma':[0.01,0.1,1]}
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
model = GridSearchCV(SVC(kernel='rbf',random_state=123),param_grid,cv=kfold)
model.fit(X_train_s,y_train)
print(model.score(X_test_s, y_test))
pred = model.predict(X_test_s)
print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))








