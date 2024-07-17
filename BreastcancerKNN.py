import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import  load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
# pd.options.display.max_columns = 40
df['diagnosis']= cancer.target   # 0是恶性肿瘤，1是良性肿瘤


d = {0:'malignant',1:'benign'}
df['diagnosis'] = df['diagnosis'].map(d)

# 先检查均值和标准差的差异程度来确定是否需要标准化
# print(df.iloc[:, :3].describe())
# print(df.diagnosis.value_counts())
# sns.boxplot(x='diagnosis',y='mean radius',data=df)  # 发现恶性的肿瘤大小比良性大小要大很多
# plt.show()

X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X, y,stratify=y,test_size=100,random_state=1)

scaler = StandardScaler()
# 先fit再transform
scaler.fit(X_train)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# print(np.mean(X_train_s,axis=0))
# print(np.mean(X_test_s,axis=0))   # 这里test的mean没有这么接近0的原因是因为用train来fit的

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_s,y_train)

pred = model.predict(X_test_s)
# print(pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted']))
# print(model.score(X_test_s, y_test)) # 0.97

# 我们发现我们的正确率很高，下面测试不同k的影响
scores = []
ks = range(1,51)
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s,y_train)
    score = model.score(X_test_s,y_test)
    scores.append(score)

# print(np.argmax(scores),max(scores))    # 2 0.97
# index_max = np.argmax(scores)
# plt.plot(ks,scores,'o-')
# plt.xlabel('K')
# plt.axvline(ks[index_max],linewidth = 1,linestyle='--',color='k')
# plt.ylabel('Accuracy')
# plt.title('KNN')
# plt.tight_layout()
# plt.show()

param_grid = {'n_neighbors':range(1,51)}
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
model = GridSearchCV(KNeighborsClassifier(),param_grid,cv=kfold)       # 这里需要用字典形式
model.fit(X_train_s,y_train)
print(model.best_params_)    # {'n_neighbors': 12}
print(model.score(X_test_s, y_test))    # 0.96









