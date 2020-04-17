## Imports
import numpy as np
import matplotlib.pyplot as plt

##### Settings #####

## Parameters
num_epochs = 300
max_num_hidden = 3
# base eta value
eta = 0.2

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')
# set to False to disable figure save
plot_filename = False#"ResultPython.pgf"
data_filename = "DataPython{}.csv"
data_stride = 1

##### End of Settings #####

## Functions
def UniformRandomMatrix (rows, cols):
    res = [[np.random.uniform () for c in range (cols)] for r in range (rows)]
    return np.matrix (res)

## Load the data
X = np.matrix ([[0,1,0,1],
                [0,0,1,1]])
yt= np.matrix ([[0,1,1,0]])

(num_inputs, num_samples) = X.shape

bias = np.matrix (np.ones (num_samples))
X = np.append (X, bias, axis=0)
num_inputs += 1



## Test various hidden node counts
for num_hidden in range (1, max_num_hidden + 1):
    print ("Hidden nodes: {}".format (num_hidden))

    ## Initialise weights
    np.random.seed (123456)
    W = UniformRandomMatrix (num_hidden, num_inputs)
    w = UniformRandomMatrix (1, num_hidden+1)

    first = True
    ## Iterate
    mse = []
    pred = yt
    for _ in range (num_epochs):
        # output
        Phi = np.tanh (W*X)
        Psi = np.append (Phi, bias, axis=0)
        y = w*Psi
        # err1
        e = y - yt
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
        pred = y

    ## Plot
    if data_filename:
        with open (data_filename.format (num_hidden), "w") as f:
            I = [i for i in range (0, num_epochs, data_stride)]
            if I[-1] != num_epochs-1:
                I.append (num_epochs-1)
            for i in I:
                f.write ("{0:d},{1:f}\n".format (i, mse[i]))
    plt.plot (range (num_epochs), mse, label="Hidden Neurons = {0}".format (num_hidden))
    print ("initial error", mse[0])
    print ("final error", mse[-1])
    print ("final prediction", pred)

# save plot and display
plt.legend ()
if plot_filename:
    plt.savefig (plot_filename)
plt.show ()

##### EOF #####
