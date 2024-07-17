import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

pd.set_option('display.max_columns', None)

wine = pd.read_csv(r"C:\Users\hyn\Desktop\ML\datasett\\wine.csv")
X = wine.iloc[:,1:]
y = wine.iloc[:,0]

sns.displot(wine.Type,kde=False)
plt.show()
# print(X)
# print(wine.Type.value_counts())
# print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
model = LogisticRegression(solver='newton-cg',C=1e10, max_iter=1000)
model.fit(X_train,y_train)
# print(model.n_iter_)
# print(model.intercept_)
# print(model.coef_)
print(model.score(X_test, y_test))

pred = model.predict(X_test)

table = pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['Predicted'])

sns.heatmap(table,cmap = 'Blues',annot= True)
plt.tight_layout()
plt.show()
# print(table)
# print(classification_report(y_test,pred))

print(cohen_kappa_score(y_test, pred))

# # 创建一个新的数据行，确保所有列都与原始数据一致
# new_data_values = [13.5, 2.1, 2.3, 18.0, 105, 2.5, 2.7, 0.3, 1.6, 5.0, 1.2, 3.2, 850]
#
# # 将新数据转化为 DataFrame
# new_data = pd.DataFrame([new_data_values], columns=X.columns)
#
# # 使用训练好的模型对新数据进行预测
# prediction = model.predict(new_data)
# print("预测的类别:", prediction)


