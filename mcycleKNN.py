import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import  load_iris
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import  plot_decision_regions

# # KNN 中的回归问题
# mcycle = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\mcycle.csv')
#
# X = np.array(mcycle['times']).reshape(-1, 1)
# y = mcycle['accel']
#
# fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
# for i, k in enumerate([1, 10, 25, 50]):
#     model = KNeighborsRegressor(n_neighbors=k)
#     model.fit(X, y)
#     pred = model.predict(np.arange(60).reshape(-1, 1))
#
#     ax_idx = (i // 2, i % 2)
#     ax[ax_idx].scatter(mcycle['times'], mcycle['accel'], s=20, facecolor='none', edgecolor='k')
#     ax[ax_idx].plot(np.arange(60), pred, 'b')
#     ax[ax_idx].text(0, 55, f'K={k}')
#
# plt.tight_layout()
# plt.show()

# KNN中的分类问题

X,y = load_iris(return_X_y=True)
X2 = X[:,2:4]

fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.1,wspace=0.1)

for i, k in enumerate([1, 10, 25, 50]):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X2,y)
    ax_idx = (i // 2, i % 2)  # 定位到具体的子图
    plot_decision_regions(X2, y, clf=model, ax=ax[ax_idx], legend=2)
    ax[ax_idx].set_xlabel('Petal Length')
    ax[ax_idx].set_ylabel('Petal Width')
    ax[ax_idx].text(0.3, 3, f'K={k}')
plt.tight_layout()
plt.show()















