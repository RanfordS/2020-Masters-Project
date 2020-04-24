## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## Settings

x_0 = 0.1        # starting value
num = 1000       # total number of samples
frame = 10       # number of samples in a timeframe
split_ft = 0.8   # fitting/testing split
neurons = 64     # number of LSTM neurons
epochs = 100     # number of epochs
batch = 16       # training batch size
split_vt = 0.1   # validation/training split
tf.random.set_seed (123456)

## End of settings

## Generate chaotic data
x_all = [x_0]
for i in range (num-1):
    x_0 = 4*x_0*(1 - x_0)
    x_all.append (x_0)
x_all = np.array (x_all)

#plt.plot (x)
#plt.show ()


## Prepare data
# need (S,T,C,) array, where
#  S = number of samples
#  T = timesteps per sample
#  C = channels (1)
# for x

# create separate x and y sets with batched frames
x = []
y = []
for i in range (num - frame):
    f = []
    for j in range (frame):
        f.append ([x_all[i + j]])
    x.append (f)
    y.append (x_all[i + frame])
x = np.array (x)
y = np.array (y)
num -= frame
# split x and y into fitting and testing
fitt_num = int (num*split_ft)
fitt_x = x[0:fitt_num]
fitt_y = y[0:fitt_num]
test_num = num - fitt_num
test_x = x[fitt_num:num]
test_y = y[fitt_num:num]

fitt_x += np.random.uniform (-0.10, 0.10, fitt_x.shape)

## Build model

layers = tf.keras.layers
model = tf.keras.Sequential ()

model.add (layers.LSTM (neurons))
#model.add (layers.Flatten (input_shape = fitt_x[0].shape))
#model.add (layers.Dense (neurons, activation = 'tanh'))
#model.add (layers.Dense (neurons, activation = 'tanh'))

model.add (layers.Dense (1))
model.compile (loss = 'mean_squared_error', optimizer = 'Adam')

## Train

results = model.fit (fitt_x, fitt_y,
                     epochs = epochs,
                     batch_size = batch,
                     validation_split = split_vt,
                     shuffle = False)

## Results

# plot error history
plt.plot (results.history['loss'], label = "Loss")
plt.plot (results.history['val_loss'], label = "Validation")
plt.legend ()
plt.show ()

# plot prediction graph
pred_y = model.predict (test_x)[:,0]
print (pred_y.shape)
print (test_y.shape)
pred_err = np.sqrt (sum ((pred_y - test_y)**2))
print ("Prediction error", pred_err)
plt.plot (pred_y, label = "Prediction")
plt.plot (test_y, label = "Actual")
plt.legend ()
plt.show ()
