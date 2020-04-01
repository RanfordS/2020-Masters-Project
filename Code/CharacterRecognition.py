## Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# short name
layers = tf.keras.layers

## Parameters
num_epochs = 100

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')

## Load data
char_set = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = char_set.load_data ()

sample_size  = x_train.shape[0]
sample_shape = x_train[0].shape

## Normalize
x_train = x_train/255.0
x_test  = x_test /255.0

## Custom activation
def act_threshold (x):
    condition = tf.less (x, tf.constant (0.0))
    #return tf.where (condition, tf.constant (0.0), x)# + tf.constant (1.0))
    #return tf.where (condition, tf.constant (0.0), x*x)
    #return x*x
    #return tf.where (condition, tf.constant (0.0), tf.math.sqrt (x))
    #return tf.where (condition, tf.constant (0.1)*x, x)
    #return tf.where (condition, tf.constant (0.0), tf.constant (1.0))
    #return tf.where (condition, tf.constant (0.0), tf.constant (1.0) - tf.math.exp (-x))
    #return tf.where (condition, tf.constant (0.0), tf.math.exp (x))
    return tf.where (condition, tf.constant (0.0), x - x*tf.math.exp (-x))
    #return tf.math.exp (x)
    #return tf.where (condition, tf.constant (0.0), tf.math.acos (x))
    #return tf.where (condition, x*x, x)
    #return tf.where (condition, tf.constant (0.0), x*x*x)

## Create model
model = tf.keras.models.Sequential (
[   layers.Flatten (input_shape=sample_shape)
,   layers.Dense (128, activation='relu')#'relu' or act_threshhold
,   layers.Dense (10, activation="softmax")
])
model.compile (optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=[])

## Train
result = model.fit (x_train, y_train,
                    epochs=num_epochs,
                    batch_size=sample_size,
                    use_multiprocessing=True,
                    workers=12,
                    validation_data=(x_test, y_test))

## Plot
plt.plot (range (num_epochs), result.history['loss'], label='loss')
plt.plot (range (num_epochs), result.history['val_loss'], label='validation')
plt.legend ()
plt.show ()

predictions = model.predict (x_test)

## Show elements with predictions
plt.figure ()
for i in range (25):
    plt.subplot (5, 5, i+1)
    plt.xticks ([])
    plt.yticks ([])
    plt.grid (False)
    plt.imshow (x_test[i], cmap=plt.cm.binary)
    y_pred = np.argmax (predictions[i])
    y_cert = max (predictions[i])
    plt.xlabel ("NN:{0} ({1:.2f}), Is:{2}".format (y_pred, y_cert, y_test[i]))
plt.show ()
