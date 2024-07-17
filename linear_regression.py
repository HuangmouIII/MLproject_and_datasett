import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt  # 确保导入 pyplot
import pandas as pd
# data = sm.datasets.engel.load_pandas().data
# model = smf.ols('foodexp ~ income', data=data)
# results = model.fit()
#
# a = sns.regplot(x='income', y='foodexp', data=data)
# plt.show()  # 显示图形


path = r"C:\Users\hyn\Desktop\ML\datasett\cobb_douglas.csv"
data = pd.read_csv(path)
model = smf.ols('lny ~ lnk + lnl', data=data)
results = model.fit()
print(results.params)