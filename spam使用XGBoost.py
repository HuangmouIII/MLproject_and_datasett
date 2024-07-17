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
import xgboost as xgb

spam = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\spam.csv')

X = spam.iloc[:,:-1]
y = spam.iloc[:,-1]
d = {'spam':1,'email':0}
y = y.map(d)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)

model = xgb.XGBClassifier(objective = 'binary:logistic',n_estimators=300,max_depth=6,subsample = 0.6,colsample_bytree = 0.8,learning_rate = 0.1,random_state=0)
model.fit(X_train,y_train)
# print(model.score(X_test, y_test))

pred = model.predict(X_test)
print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))























