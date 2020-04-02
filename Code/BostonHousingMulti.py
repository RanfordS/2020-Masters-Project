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
train_y = np.matrix (train_y)
(num_samples, num_inputs) = train_X.shape
# add bias
bias = np.ones (num_samples)
bias = np.matrix (bias).transpose ()
train_X = np.append (train_X, bias, axis=1)

## Normalize data
for i in range (num_inputs):
    col = train_X[:,i]
    train_X[:,i] = (col - col.mean()) / col.std()
miny = train_y.min ()
maxy = train_y.max ()
mean = (maxy + miny)/2
std  = (maxy - miny)/2
train_y = (train_y - mean)/std
# adjust for bias column
num_inputs += 1
# adjust for sample size
eta /= num_samples

## Test various hidden node counts
for num_hidden in range (1, max_num_hidden + 1):
    print ("Hidden nodes: {}".format (num_hidden))

    ## Initialise weights
    np.random.seed (123456)
    w_hidden = 0.1*UniformRandomMatrix (num_inputs, num_hidden)
    w_output = 0.1*UniformRandomMatrix (num_hidden+1, 1)

    first = True
    ## Iterate
    mse = []
    for _ in range (num_epochs):
        # outputs
        phi = np.append (np.tanh (train_X*w_hidden), bias, axis=1)
        y = phi * w_output
        err = y - train_y.transpose ()
        # gradients
        g_output = phi.transpose() * err
        phi_range = np.array (phi[:, range(num_hidden)])
        w_output_range = w_output [range (num_hidden), 0].transpose ()
        err_term = np.array (err*w_output_range)
        g_hidden = train_X.transpose() * np.matrix((1 - phi_range**2)*err_term)
        if first:
            first = False
            print ("w_hidden ", w_hidden.shape)
            print ("w_output ", w_output.shape)
            print ()
            print ("phi      ", phi.shape)
            print ("y        ", y.shape)
            print ("err      ", err.shape)
            print ("g_output ", g_output.shape)
            print ()
            print ("err      ", err.shape)
            print ("phi_range", phi_range.shape)
            print ("(1-p^2)*e", ((1-phi_range**2)*err_term).shape)
            print ("out_range", w_output_range.shape)
            print ("err_term ", err_term.shape)
            print ("g_hidden ", g_hidden.shape)
            print ("X^T      ", train_X.transpose().shape)
            print ("g_hidden ", g_hidden.shape)
            print ()

        # descent
        w_output -= eta * g_output
        w_hidden -= eta * g_hidden#.transpose ()
        mse.append (err.var ())

    ## Plot
    plt.plot (range (num_epochs), mse, label="Hidden Neurons = {0}".format (num_hidden))

# save plot and display
plt.legend ()
if plot_filename:
    plt.savefig (plot_filename)
plt.show ()

##### EOF #####
