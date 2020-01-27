import numpy as np
# import keras so that we can access the Boston housing data
from tensorflow import keras

# parameters
num_epochs = 10000
eta = 0.001
inputs = (5,8,12,1) # end index is a dummy for the bais

# load the data and crop it
num_inputs = len (inputs)
dataset = keras.datasets.boston_housing
(train_X, train_y), (test_X, test_y) = dataset.load_data (test_split = 0)
train_X = train_X[:,inputs]
train_X[:,-1] = 1
#test_X  = test_X [:,inputs]
num_train = train_X.shape[0]

print (train_X)

# normalize data
for i in range (num_inputs - 1):
    train_X[:,i] = (train_X[:,i] - train_X[:,i].mean()) / train_X[:,i].std()
miny = min (train_y)
maxy = max (train_y)
mean = (maxy + miny)/2
std =  (maxy - miny)/2
train_y = (train_y - mean)/std

print (train_X)

# initialise weights with random values
w = np.array ([0.1*np.random.uniform () for i in inputs])

# single layer
y1 = np.tanh (np.array([np.dot (train_X[i,:], w) for i in range (num_train)]))
e1 = train_y - y1

# trackers
W = [w]
mse = e1.var()
print ("mean square err: {0}".format (mse))

k = 0
for m in range (num_epochs):
    k += 1
    #epoch_err = 0
    for n in range (num_inputs):
        yk = np.tanh (np.dot (train_X[n,:], w))
        err = yk - train_y[n]
        #epoch_err += abs (err)
        g = train_X[n,:] * ((1 - yk**2)*err)
        w = w - eta*g
        W.append (w)
        #print (err)
        #print (w)
    #print ("epocherr: {0}".format (epoch_err))

# single layer
y2 = np.tanh (np.array([np.dot (train_X[i,:], w) for i in range (num_train)]))
e2 = train_y - y2

print (0)
print (W[0])
print (k)
print (W[-1])
# trackers
mse = e2.var()
print ("mean square err: {0}".format (mse))


