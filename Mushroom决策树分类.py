import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 读取数据
mushroom = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\Mushroom.csv')

# 检查数据
print(mushroom.head(5))
print(mushroom.shape)

# 预处理数据
X = mushroom.iloc[:, 1:]
y = mushroom.iloc[:, 0]
print(y.value_counts())

# 将类别变量转换为虚拟变量
X = pd.get_dummies(X)

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

# 初始模型训练
model = DecisionTreeClassifier(max_depth=2, random_state=123)
model.fit(X_train, y_train)
plot_tree(model, feature_names=X.columns, node_ids=True, proportion=True, rounded=True, precision=2)
plt.show()
print("Initial model score:", model.score(X_test, y_test))

# 使用GridSearchCV进行模型优化
model = DecisionTreeClassifier(random_state=123)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
path = model.cost_complexity_pruning_path(X_train, y_train)
param_grid = {'ccp_alpha': path.ccp_alphas}
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)
print("Optimized model score:", model.score(X_test, y_test))
print("Best parameters:", model.best_params_)

# 提取最佳模型
best_model = model.best_estimator_

# 绘制最佳模型的决策树
plot_tree(best_model, feature_names=X.columns, node_ids=True, proportion=True, rounded=True, precision=2)
plt.show()

# 特征重要性分析
sorted_index = best_model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), best_model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X.columns[sorted_index])
plt.ylabel('Feature')
plt.xlabel('Decision Tree')
plt.tight_layout()
plt.show()

# 预测结果
pred = best_model.predict(X_test)
print(pd.crosstab(y_test, pred, colnames=['Actual'], rownames=['Predict']))
prob = best_model.predict_proba(X_test)
print(prob)
