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

(X_trainval,y_trainval_original),(X_test,y_test_original) = reuters.load_data(num_words=1000)
Y_trainval_original = pd.DataFrame(y_trainval_original,columns=['topic'])








