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

(X_train,y_train),(X_test,y_test) = boston_housing.load_data(test_split=0.2,seed=113)
print(X_train.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

def set_my_seed():
    os.environ['PYTHONHASHSEED']='0'
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)

set_my_seed()

def build_model():
    model = Sequential()
    model.add(Dense(units=256,activation='relu',input_shape=(X_train_s.shape[1],)))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer = 'rmsprop',loss='mse',metrics=['mse'])
    return model

# model = build_model()
# print(model.summary())
# hist = model.fit(X_train_s,y_train,validation_split=0.25,epochs = 300, batch_size = 16,shuffle = False)
#
# Dict = hist.history
# print(Dict)
#
# val_mse = Dict['val_mse']
#
# index = np.argmin(val_mse)
#
# plt.plot(Dict['mse'],'k',label = 'Train')
# plt.plot(Dict['val_mse'],'b',label = 'Validation')
# plt.axvline(index+1,linestyle='--',color='k')
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.title('Mean Squared Error')
# plt.legend()
# plt.show()

set_my_seed()
model = build_model()
model.fit(X_train_s,y_train,epochs =294,batch_size = 16,verbose = 1)
pred = model.predict(X_test_s)
pred = np.squeeze(pred)
print(np.corrcoef(y_test, pred) ** 2)    # 0.86602909








