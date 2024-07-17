import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

pd.set_option('display.max_columns', None)

iris = load_iris()
X = pd.DataFrame(iris.data,columns=iris.feature_names)
# sns.heatmap(X.corr(),cmap='Blues',annot=True)
# plt.show()
# print(X.corr())
# print(X)

y = iris.target
model = LinearDiscriminantAnalysis()
model.fit(X, y)
# print(model.score(X, y ))
# print(model.explained_variance_ratio_)

lda_loadings = pd.DataFrame(model.scalings_,index=iris.feature_names,columns=['LD1','LD2'])
# print(lda_loadings)
lda_scores = model.fit(X,y).transform(X)
# print(lda_scores.shape)
LDA_scores = pd.DataFrame(lda_scores,columns=['LD1','LD2'])
LDA_scores['Species']= iris.target
d = {0:'setosa',1:'versicolor',2:'virginica'}
LDA_scores['Species']=LDA_scores['Species'].map(d)
# print(LDA_scores)
#
# sns.scatterplot(x='LD1',y='LD2',data=LDA_scores,hue='Species')
# plt.show()

# 上面是线性判元，下面用画决策边界，只用最后两个特征

X2 = X.iloc[:,2:4]
model = LinearDiscriminantAnalysis()
model.fit(X2,y)
# print(model.score(X2, y))            # 0.96
# 第一个线性判元可以解释0.9947的组间方差
# print(model.explained_variance_ratio_)

# # 画决策边界
# plot_decision_regions(np.array(X2),y,model)
# plt.xlabel('petal_length')
# plt.ylabel('petal_width')
# plt.title('Decision Boundary for LDA')
# plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=123)

model = LinearDiscriminantAnalysis()
model.fit(X_train,y_train)

# print(model.score(X_test,y_test))

pred = model.predict(X_test)

# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred))

# 下面用二次判别分析QDA

model = QuadraticDiscriminantAnalysis()
model.fit(X_train,y_train)
print(model.score(X_test, y_test))

X2 = X.iloc[:,2:4]
model = QuadraticDiscriminantAnalysis()
model.fit(X2,y)
print(model.score(X2,y))

plot_decision_regions(np.array(X2),y,model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for QDA')
plt.show()

