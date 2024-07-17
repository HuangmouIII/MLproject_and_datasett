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
spam = pd.read_csv(r"C:\Users\hyn\Desktop\ML\datasett\spam.csv")

X = spam.iloc[:,:-1]
y = spam.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = MLPClassifier(solver='adam',hidden_layer_sizes=(100,),random_state=123,max_iter=10000)
model.fit(X_train_s,y_train)
print(model.score(X_test_s, y_test))

model = MLPClassifier(solver='adam',hidden_layer_sizes=(500,500),random_state=123,max_iter=10000,early_stopping=True,validation_fraction=0.25)
model.fit(X_train_s,y_train)
print(model.score(X_test_s, y_test))

model = MLPClassifier(solver='adam',hidden_layer_sizes=(500,500),random_state=123,max_iter=10000,alpha=0.1)
model.fit(X_train_s,y_train)
print(model.score(X_test_s, y_test))










