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

meter = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\Meter_D.csv')

# 1.
# print(meter.head(5))
# print(meter.shape)

# 2. 3.
X = meter.iloc[:,:-1]
y = meter.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=100,stratify=y,random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# print(np.mean(X_train_s, axis=0))
# print(np.mean(X_test_s, axis=0))
# print(np.std(X_train_s, axis=0))
# print(np.std(X_test_s, axis=0))

# 4.
model = SVC(kernel='linear',random_state=123)
model.fit(X_train_s,y_train)
print("线性核:",model.score(X_test_s, y_test))
# 5.
pred = model.predict(X_test_s)
print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))
# 6.
model = SVC(kernel='poly',degree=2,random_state=123)
model.fit(X_train_s,y_train)
print("多项式核2:",model.score(X_test_s, y_test))
# 7.
model = SVC(kernel='poly',degree=3,random_state=123)
model.fit(X_train_s,y_train)
print("多项式核3:",model.score(X_test_s, y_test))
# 8.
model = SVC(kernel='rbf',random_state=123)
model.fit(X_train_s,y_train)
print("径向核:",model.score(X_test_s, y_test))

# 9.
model = SVC(kernel='sigmoid',random_state=123)
model.fit(X_train_s,y_train)
print("S型核:",model.score(X_test_s, y_test))

# 10.
scores = []
for c in range(1,500):
    model = SVC(kernel='linear',C=c, random_state=123)
    model.fit(X_train_s, y_train)
    scores.append(model.score(X_test_s, y_test))
scores_error = [1-a for a in scores]
# print(scores_error)
plt.plot(range(1,500),scores_error)
plt.xlabel('C_value')
plt.ylabel('error')
plt.show()
best_c = range(1,500)[np.argmin(scores_error)]
print(best_c)

# 11.
model = SVC(kernel='linear',C=350, random_state=123)
model.fit(X_train_s, y_train)
print(model.score(X_test_s, y_test))










