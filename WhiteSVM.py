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
from sklearn.linear_model import LogisticRegression

white = pd.read_table(r'C:\Users\hyn\Desktop\ML\datasett\winequality-white.csv',sep=';')

# 1.
# print(white.head(5))
# print(white.shape)

# 2.
X = white.iloc[:,:-1]
y = white.iloc[:,-1]
# white.iloc[:,-1].plot.hist()
# plt.show()

# 3.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1000,stratify=y,random_state=0)

# 4.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 5.
model = SVC(kernel='rbf',C=1,gamma=1,random_state=123)
model.fit(X_train_s,y_train)
print("径向核:",model.score(X_test_s, y_test))    # 径向核: 0.663

# 6.
pred = model.predict(X_test_s)
print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))

# 7.
model = LogisticRegression(solver='newton-cg',C=1e10, max_iter=1000)
model.fit(X_train_s,y_train)
print("多项逻辑回归:",model.score(X_test_s, y_test))    # 多项逻辑回归: 0.546









