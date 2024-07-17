import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_decision_regions
from tqdm import tqdm
from colorama import Fore, Style, init

bar_format = f"{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}"
data_url = "http://lib.stat.cmu.edu/datasets/boston"

#region
# 读取数据
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=21, header=None)

# 重新整理数据，每两行合并成一行
data = []
for i in range(0, len(raw_df), 2):
    row = pd.concat([raw_df.iloc[i], raw_df.iloc[i+1]], ignore_index=True)
    data.append(row)
# 创建新的DataFrame
processed_df = pd.DataFrame(data)
# 删除完全是NaN的列
processed_df = processed_df.dropna(axis=1, how='all')
# 列名
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
processed_df.columns = column_names[:processed_df.shape[1]]
boston = processed_df
#endregion

X = boston.iloc[:,:-1]
y = boston.iloc[:,-1]
column_names_data = column_names[:-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(5,),random_state=123,max_iter=10000)
# model.fit(X_train_s,y_train)
# print(model.score(X_test_s, y_test))
#
# print(model.n_iter_)
# table = pd.DataFrame(model.coefs_[0],index=column_names_data,columns=[1,2,3,4,5])
# sns.heatmap(table,cmap='Blues',annot=True)
# plt.xlabel('Neuron')
# plt.ylabel('Neural Network Weights')
# plt.tight_layout()
# plt.show()
#
# result = permutation_importance(model,X_test_s,y_test,n_repeats=20,random_state=42)
# # 如果不知道出来的值有什么属性可以用dir
#
# index = result.importances_mean.argsort()
# plt.boxplot(result.importances[index].T,vert=False,labels=column_names_data)
# plt.title('Permutation Importances')
# plt.show()
# scores = []
# for n_nodes in range(1,41):
#     model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(n_nodes,),random_state=123,max_iter=10000)
#     model.fit(X_train_s,y_train)
#     scores.append(model.score(X_test_s,y_test))
#
# index = np.argmax(scores)
# print(range(1,41)[index])
# plt.plot(range(1,41),scores,'o-')
# plt.axvline(range(1,41)[index],linestyle='--',color='k',linewidth=1)
# plt.xlabel('Number of Nodes')
# plt.ylabel('R2')
# plt.title('Test Set R2 vs Number of Nodes')
# plt.show()

model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(8,),random_state=123,max_iter=10000)
model.fit(X_train_s,y_train)
print(model.score(X_test_s, y_test))

best_score = 0
best_size = (1,1)
for i in tqdm(range(1,11),desc="Processing", bar_format=bar_format):
    for j in range(1,11):
        model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(i,j),random_state=123,max_iter=10000)
        model.fit(X_train_s,y_train)
        score = model.score(X_test_s,y_test)
        if best_score < score:
            best_score = score
            best_size = (i,j)

print(best_score)
print(best_size)












