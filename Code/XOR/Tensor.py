## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# short name
layers = tf.keras.layers

##### Settings #####

num_epochs = 2000
## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')

data_filename = "DataTensor{}.csv"
data_stride = 10

##### End of Settings #####

## Data
X = np.matrix ([[0,1,0,1],
                [0,0,1,1]]).transpose()
yt= np.matrix ([[0,1,1,0]]).transpose()

for num_hidden in [1,2,3]:
    print ("count:", num_hidden)
    tf.random.set_seed (12345)
    ## Create model
    model = tf.keras.models.Sequential (
    [   layers.Dense (num_hidden,
                      input_dim=2,
                      activation='tanh')
    ,   layers.Dense (1)
    ])
    model.compile (#optimizer='RMSprop',#'adam',
                   loss='mean_squared_error')#,
                   #metrics=[])

    ## Train
    result = model.fit (X, yt,
                        epochs=num_epochs,
                        verbose=0,
                        batch_size=4)

    ## Plot
    if data_filename:
        with open (data_filename.format (num_hidden), "w") as f:
            for i in range (0, num_epochs, data_stride):
                f.write ("{0:d},{1:f}\n".format (i, result.history['loss'][i]))
    plt.plot (range (num_epochs),
              result.history['loss'],
              label='Num Hidden = {0}'.format(num_hidden))
    print ("initial:", result.history['loss'][0])
    print ("final:", result.history['loss'][-1])
plt.legend ()
plt.show ()

