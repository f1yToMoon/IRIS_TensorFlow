import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

data_path = os.path.join('data')
os.makedirs(data_path, exist_ok=True)

iris_data = pd.read_csv('C:/Users/IRIS.csv', header=0)

X = iris_data.loc[:, iris_data.columns != 'species']
y = iris_data.loc[:, ['species']]

y_enc = LabelEncoder().fit_transform(y)
y_label = tf.keras.utils.to_categorical(y_enc)

X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.3)

X_train.to_csv(os.path.join(data_path, "X_train.csv"))
X_test.to_csv(os.path.join(data_path, "X_test.csv"))
with open('data/y_train.npy', 'wb') as f:
    np.save(f, y_train)
with open('data/y_test.npy', 'wb') as f:
    np.save(f, y_test)
