## Imports
import numpy as np
import matplotlib.pyplot as plt
# import keras so that we can access the Boston housing data
from tensorflow import keras

##### Settings #####

## Parameters
num_epochs = 10
eta = 0.001
# which columns to use
# last column will be replaced with the bias
inputs = (5,8,12,1) 

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')
# set to false to diable figure save
plot_filename = "BostonHousingResult.pdf"

##### End of settings #####

## Load the data and crop it
num_inputs = len (inputs)
dataset = keras.datasets.boston_housing
(train_X, train_y), (test_X, test_y) = dataset.load_data (test_split = 0)
train_X = train_X[:,inputs]
train_X[:,-1] = 1
num_train = train_X.shape[0]

## Normalize data
for i in range (num_inputs - 1):
    train_X[:,i] = (train_X[:,i] - train_X[:,i].mean()) / train_X[:,i].std()
miny = min (train_y)
maxy = max (train_y)
mean = (maxy + miny)/2
std  = (maxy - miny)/2
train_y = (train_y - mean)/std

## Initialise weights with random values
w = np.array ([0.1*np.random.uniform () for i in inputs])

## Initial error value
y1 = np.tanh (np.array([np.dot (train_X[i,:], w) for i in range (num_train)]))
e1 = train_y - y1
mse_start = e1.var ()

## Main loop
W = [w]
k = 0
for _ in range (num_epochs):
    for n in range (num_train):
        yk = np.tanh (np.dot (train_X[n,:], w))
        err = yk - train_y[n]
        g = train_X[n,:] * ((1 - yk**2)*err)
        w = w - eta*g
        W.append (w)
        k += 1

## Final error value
y2 = np.tanh (np.array([np.dot (train_X[i,:], w) for i in range (num_train)]))
e2 = train_y - y2
mse_end = e2.var()

## Display results
# weights
print ("Initial weights")
print (W[0])
print ("Final weights:")
print (W[-1])
# error values
print ("Initial mean square err: {0}".format (mse_start))
print ("Final mean square err:   {0}".format (mse_end))
# plot of weights against iterations
W = np.array (W)
K = range (k+1)
for i in range (num_inputs):
    plt.plot (K, W[:,i].tolist(), label="$w_{{{0}}}$".format (i))

plt.legend ()
if plot_filename:
    plt.savefig (plot_filename)
plt.show ()

##### EOF #####
