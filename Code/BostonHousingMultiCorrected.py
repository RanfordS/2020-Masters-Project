## Imports
import numpy as np
import matplotlib.pyplot as plt
# import keras so that we can access the Boston housing data
from tensorflow import keras

##### Settings #####

## Parameters
num_epochs = 1000
max_num_hidden = 3
# base eta value
eta = 0.05

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')
# set to false to disable figure save
plot_filename = "BostonHousingMultiResult.pdf"

##### End of Settings #####

## Functions
def UniformRandomMatrix (rows, cols):
    res = [[np.random.uniform () for c in range (cols)] for r in range (rows)]
    return np.matrix (res)

## Load the data
dataset = keras.datasets.boston_housing
(train_X, train_y), (_,_) = dataset.load_data (test_split = 0)
train_X = np.matrix (train_X)
train_y = np.matrix (train_y).transpose()
(num_samples, num_inputs) = train_X.shape
# add bias
bias = np.ones (num_samples)
bias = np.matrix (bias)
X = np.append (train_X.transpose(), bias, axis=0)

print (train_X.shape)

## Normalize data
for i in range (num_inputs):
    row = X[i,:]
    X[i,:] = (row - row.mean()) / row.std()
miny = train_y.min ()
maxy = train_y.max ()
mean = (maxy + miny)/2
std  = (maxy - miny)/2
train_y = (train_y.transpose() - mean)/std
# adjust for bias column
num_inputs += 1
# adjust for sample size
eta /= num_samples

## Test various hidden node counts
for num_hidden in range (1, max_num_hidden + 1):
    print ("Hidden nodes: {}".format (num_hidden))

    ## Initialise weights
    np.random.seed (123456)
    W = 0.1*UniformRandomMatrix (num_hidden, num_inputs)
    w = 0.1*UniformRandomMatrix (1, num_hidden+1)

    first = True
    ## Iterate
    mse = []
    for _ in range (num_epochs):
        # output
        Phi = np.tanh (W*X)
        Psi = np.append (Phi, bias, axis=0)
        y = w*Psi
        # err1
        e = y - train_y
        g_out = e*Psi.transpose()
        # err2
        w_hat = w[0, range (num_hidden)]
        E = w_hat.transpose()*e
        Phi = np.array (Phi)
        E = np.array (E)
        Phi = (1 - Phi**2)*E
        Phi = np.matrix (Phi)
        G_hidden = Phi*X.transpose()
        # step
        w -= eta*g_out
        W -= eta*G_hidden
        mse.append (e.var())

    ## Plot
    plt.plot (range (num_epochs), mse, label="Hidden Neurons = {0}".format (num_hidden))

# save plot and display
plt.legend ()
if plot_filename:
    plt.savefig (plot_filename)
plt.show ()

##### EOF #####
