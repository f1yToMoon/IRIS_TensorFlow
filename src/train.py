import pandas as pd
import numpy as np
import sys
import os
import yaml

from tensorflow import keras
from tensorflow.keras.models import Sequential
from dvclive.keras import DVCLiveCallback
from dvclive import Live

if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py features-dir-path model-filename\n'
    )
    sys.exit(1)

data_path = sys.argv[1]

params = yaml.safe_load(open('params.yaml'))['train']

epochs = params['epochs']

X_test_file = os.path.join(data_path, 'X_test.csv')
X_train_file  = os.path.join(data_path, 'X_train.csv')
y_test_file = os.path.join(data_path, 'y_test.npy')
y_train_file  = os.path.join(data_path, 'y_train.npy')

X_test = pd.read_csv(X_test_file)
X_train = pd.read_csv(X_train_file)
y_test = np.load(y_test_file)
y_train = np.load(y_train_file)

def get_model():
    model = Sequential([
        keras.layers.Input(shape=X_train.shape[1:]),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(500, activation='relu',),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    return model

model = get_model()

model.compile(optimizer='adam', 
              loss=keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])

from dvclive import Live
from dvclive.keras import DVCLiveCallback

with Live("custom_dir") as live:
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1, callbacks=[DVCLiveCallback(live=live)])

    # Log additional data after training
    test_loss, test_acc = model.evaluate(X_test, y_test)
    live.log_metric("test_loss", test_loss, plot=False)
    live.log_metric("test_acc", test_acc, plot=False)
