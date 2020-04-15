## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# short name
layers = tf.keras.layers

##### Settings #####

## Parameters
num_epochs = 2000
num_hidden = 5
normalise = True

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')
plot_filename = "BostonHousingTensorResultsSGD.pgf"
tf.random.set_seed (12)

##### End of Settings #####

## Load data
boston_housing = tf.keras.datasets.boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data ()
# properties
num_samples = x_train.shape[0]
num_inputs  = x_train.shape[1]

## Normalize data
if normalise:
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
model.compile (optimizer='SGD',#'Adam',
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

print ("Initial loss:", result.history['loss'][0])
print ("Final loss:", result.history['loss'][-1])

print ("Initial validation loss:", result.history['val_loss'][0])
print ("Final validation loss:", result.history['val_loss'][-1])

## Plot
plt.plot (range (num_epochs), result.history['loss'], label='Loss')
plt.plot (range (num_epochs), result.history['val_loss'], label='Validation')
plt.legend ()
if plot_filename:
    plt.savefig (plot_filename)
plt.show ()

