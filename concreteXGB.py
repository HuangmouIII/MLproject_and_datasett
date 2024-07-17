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

concrete = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\concrete.csv')

X = concrete.iloc[:,:-1]
y = concrete.iloc[:,-1]
# print(y.value_counts())

# 1.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=300,random_state=0)

# 2.
model = xgb.XGBRegressor(random_state=123)
model.fit(X_train, y_train)
# print(model.score(X_train, y_train))    # 0.9950603762686686
# print(model.score(X_test, y_test))    # 0.9101108879253677

# 3.
sorted_index = model.feature_importances_.argsort()
# plt.barh(range(X.shape[1]),model.feature_importances_[sorted_index])
# plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
# plt.ylabel('Feature')
# plt.xlabel('Decision Tree')
# plt.tight_layout()
# plt.show()

# 4.
PartialDependenceDisplay.from_estimator(model,X_train,['Age','Cement'])
plt.show()

# 5.
pred = model.predict(X_test)
print(mean_squared_error(y_test, pred))

# 6.
params = {'objective':'reg:squarederror','max_depth': 6,'subsample':0.6,'colsample_bytree':0.8,'learning_rate':0.1}
dtrain = xgb.DMatrix(data=X_train,label=y_train)
results = xgb.cv(dtrain=dtrain,params=params,nfold=10,metrics="rmse",num_boost_round=200,as_pandas=True,seed=123)
plt.plot(range(1,201),results['train-rmse-mean'],'k',label = 'Training Error')
plt.plot(range(1,201),results['test-rmse-mean'],'b',label = 'Test Error')
plt.show()

# model = xgb.XGBRegressor(random_state=123,n_estimators = 200)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

