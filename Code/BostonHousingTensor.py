## Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# short name
layers = tf.keras.layers

##### Settings #####

## Parameters
num_epochs = 5000
num_hidden = 5

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')

##### End of Settings #####

## Load data
boston_housing = tf.keras.datasets.boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data ()
# properties
num_samples = x_train.shape[0]
num_inputs  = x_train.shape[1]

## Normalize data
x_full = np.append (x_train, x_test, axis=0)
y_full = np.append (y_train, y_test, axis=0)
for i in range (num_inputs):
    x_col  = x_full[:,i]
    x_mean = x_col.mean ()
    x_std  = x_col.std  ()
    x_train[:,i] = (x_train[:,i] - x_mean)/x_std
    x_test [:,i] = (x_test [:,i] - x_mean)/x_std
y_max = max (y_full)
y_min = min (y_full)
y_mean = (y_max + y_min)/2
y_std  = (y_max - y_min)/2
y_train = (y_train - y_mean)/y_std
y_test  = (y_test  - y_mean)/y_std

## Create model
model = tf.keras.models.Sequential (
[   layers.Dense (num_hidden, input_dim=num_inputs, activation='tanh')
,   layers.Dense (num_hidden, activation='tanh')
,   layers.Dense (1, activation='linear')
])
model.compile (optimizer='adam',
               loss='mean_squared_error',
               metrics=[])

## Train
result = model.fit (x_train, y_train,
                    epochs=num_epochs,
                    batch_size=num_samples,
                    use_multiprocessing=True,
                    workers=12,
                    verbose=0,
                    validation_data=(x_test, y_test))

## Evaluate
model.evaluate (x_test, y_test, verbose=2)

## Plot
plt.plot (range (num_epochs), result.history['loss'], label='loss')
plt.plot (range (num_epochs), result.history['val_loss'], label='validation')
plt.legend ()
plt.show ()

