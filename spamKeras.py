import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras import regularizers
from keras.datasets import mnist,reuters,boston_housing
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Conv2D,MaxPool2D
from keras.utils import to_categorical
from BostonHousingKeras import set_my_seed

spam = pd.read_csv(r'C:\Users\hyn\Desktop\ML\datasett\spam.csv')
X = spam.iloc[:,:-1]
y = spam.iloc[:,-1]
y = pd.get_dummies(y).iloc[:,1]

print(X.shape)
print(y.shape)

X_trainval,X_test,y_trainval,y_test = train_test_split(X,y,test_size=1000,stratify=y,random_state=0)
X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,test_size=1000,stratify=y_trainval,random_state=321)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_trainval_s = scaler.transform(X_trainval)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
X_train_s = scaler.transform(X_train)

# set_my_seed()
# def build_model():
#     model = Sequential()
#     model.add(Dense(units=256,activation='relu',input_shape=(X_train_s.shape[1],)))
#     model.add(Dense(units=256,activation='relu'))
#     model.add(Dense(units=1,activation='sigmoid'))
#     model.compile(optimizer = 'rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#     return model

# model = build_model()
# hist = model.fit(X_train_s,y_train,validation_data=(X_val_s,y_val),epochs=50,batch_size=64,shuffle=False)
#
# val_loss = hist.history['val_loss']
# index_min= np.argmin(val_loss)
# print(index_min)
#
# val_accuracy = hist.history['val_accuracy']
# np.max(val_accuracy)

set_my_seed()
def build_model1():
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_shape=(X_train_s.shape[1],),kernel_regularizer=regularizers.l2(0.0001)))
#   model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model1()
hist = model.fit(X_train_s,y_train,validation_data= (X_val_s,y_val),epochs=50,batch_size = 64,shuffle= False)
val_accuracy = hist.history['val_accuracy']
index = np.argmax(val_accuracy)
print(index)

set_my_seed()
model = build_model1()
hist = model.fit(X_trainval_s,y_trainval,epochs=index+1,batch_size = 64,verbose=0,shuffle= False)

test_loss , test_accuracy = model.evaluate(X_test_s,y_test)
print(test_loss , test_accuracy)











