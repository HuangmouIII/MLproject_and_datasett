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

mushroom = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\mushroom.csv')

# print(mushroom)
# print(mushroom.Class.value_counts(normalize= True))    # edible       0.517971
# #                                                        poisonous    0.482029
#
# 1.
X = mushroom.iloc[:,1:]
y = mushroom.iloc[:,0]
# print(X)
# print(y)

X = pd.get_dummies(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=1000,random_state=0)

# 2.
model = RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=123)
model.fit(X_train,y_train)
print(model.score(X_test, y_test))

# 3.
sorted_index = model.feature_importances_.argsort()
plt.barh(range(X.shape[1]),model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
plt.ylabel('Feature')
plt.xlabel('Decision Tree')
plt.tight_layout()
plt.show()
print(X.columns[sorted_index])

# 4.
pred = model.predict(X_test)
print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))

sns.heatmap(X.corr(),cmap='Blues')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()







