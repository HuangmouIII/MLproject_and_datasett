import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from patsy import dmatrices

titanic = pd.read_csv(r"C:\Users\hyn\Desktop\ML\datasett\titanic.csv")
freq = titanic.Freq.to_numpy()
index = np.repeat(np.arange(32),freq)
titanic=titanic.iloc[index,:]
titanic = titanic.drop('Freq',axis=1)

# print(titanic.describe())
# print(pd.crosstab(titanic.Sex,titanic.Survived,normalize='index'))
# print(pd.crosstab(titanic.Age,titanic.Survived,normalize='index'))
# print(pd.crosstab(titanic.Class,titanic.Survived,normalize='index'))

train,test = train_test_split(titanic,test_size = 0.3,stratify=titanic.Survived,random_state=0)

y_train,X_train = dmatrices('Survived ~ Class + Sex + Age',data=train,return_type = 'dataframe')
# pd.options.display.max_columns = 10
# print(X_train.head())
# print(y_train.head())
y_train = y_train.iloc[:,1]
y_test,X_test = dmatrices('Survived ~ Class + Sex + Age',data=train,return_type = 'dataframe')
y_test = y_test.iloc[:,1]

model = sm.Logit(y_train,X_train)
results = model.fit()

print(results.summary())






