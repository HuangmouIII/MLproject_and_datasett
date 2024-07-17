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

warnings.filterwarnings("ignore", category=RuntimeWarning)

spam = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\spam.csv')

# print(spam.shape)
# print(spam.spam.value_counts(normalize = True))

# 直方图观察发现大部分邮件都很接近0
# spam.iloc[:,:5].plot.hist(subplots=True,bins=100)
# plt.show()

X = spam.iloc[:,:-1]
y = spam.iloc[:,-1]

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)

# # 高斯贝叶斯
# model = GaussianNB()
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))
#
# # 多项贝叶斯对于连续数据效果没有高斯贝叶斯效果好
# model = MultinomialNB(alpha=0)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))
#
# # 互补贝叶斯
# model = ComplementNB(alpha=1)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))
#
# # 二项贝叶斯   小于等于0是0，大于0是1
# model = BernoulliNB(alpha=1)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))
# # 目前效果最好的是二项贝叶斯， 由于大部分值是略大于0且不严格遵守正态分布
#
# # 新参数二项贝叶斯，0.1为新的门槛值
# model = BernoulliNB(binarize=0.1,alpha=1)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))

# 对于这两个超参数可以套loop来寻找最优值
# best_score = 0
# for binarize in np.arange(0,1.1,0.1):
#     for alpha in np.arange(0,1.1,0.1):
#         model = BernoulliNB(binarize=binarize, alpha=alpha)
#         model.fit(X_train, y_train)
#         score = model.score(X_test,y_test)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'binarize':binarize,'alpha':alpha}
# print(best_score)
# print(best_parameters)

# 将样本分为3份进行训练，验证，测试
# 验证为确定超参数，测试来测试其误差大小

# 下面分为3份,先用trainval找到好的参数，然后再用这个参数fit trainval，最后score test部分
X_trainval,X_test,y_trainval,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)
X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,test_size=0.25,stratify=y_trainval,random_state=123)

print(y_train.shape,y_val.shape,y_test.shape)

# best_score = 0
# for binarize in np.arange(0,1.1,0.1):
#     for alpha in np.arange(0,1.1,0.1):
#         model = BernoulliNB(binarize=binarize, alpha=alpha)
#         model.fit(X_train, y_train)
#         score = model.score(X_val,y_val)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'binarize':binarize,'alpha':alpha}
# print(best_score)
# print(best_parameters)

# model = BernoulliNB(**best_parameters)
# model.fit(X_trainval, y_trainval)
# print(model.score(X_test, y_test))

# 如果样本容量大可以考虑用上面的，如果不大可以用交叉验证，比如10折交叉验证

# best_score = 0
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
# for binarize in np.arange(0,1.1,0.1):
#     for alpha in np.arange(0,1.1,0.1):
#         model = BernoulliNB(binarize=binarize, alpha=alpha)
#         scores = cross_val_score(model,X_trainval,y_trainval,cv=kfold)  # 如果直接写cv = 10 数据则不是打乱后的，除非提前打乱
#         score = np.mean(scores)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'binarize':binarize,'alpha':alpha}
# print(best_score)
# print(best_parameters)
# best_parameters={'binarize': 0.1, 'alpha': 0.1}
#
# model = BernoulliNB(**best_parameters)
# model.fit(X_trainval, y_trainval)
# print(model.score(X_test, y_test))

# 交叉验证可以用更快的办法而不是双loop，使用GridSearchCV
param_grid = {'binarize': np.arange(0,1.1,0.1), 'alpha': np.arange(0,1.1,0.1)}
model = GridSearchCV(BernoulliNB(),param_grid,cv=kfold)
model.fit(X_trainval,y_trainval)

# print(model.score(X_test, y_test))
# print(model.best_score_)
# print(model.best_params_)
# print(model.best_estimator_)  # 一般直接用这个就可以

# 以下是详细参数，包括每一折的精确率
results = pd.DataFrame(model.cv_results_)
# results.head(2)

# 下面转换成热图
scores = np.array(results.mean_test_score).reshape(11,11)
fig,ax = plt.subplots(figsize=(10,5))
ax = sns.heatmap(scores,cmap='Blues',annot=True,fmt='.3f')
ax.set_xlabel('binarize')
ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ax.set_ylabel('binarize')
ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.tight_layout()
plt.show()

pred = model.predict(X_test)
print(pred[:3])

table = pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['Predicted'])
print(table)

RocCurveDisplay.from_estimator(model,X_test,y_test)
x = np.linspace(0,1,100)
plt.plot(x,x,'k--',linewidth=1)
plt.title('ROC Curve for Bernoulli Naive Bayes')
plt.show()
