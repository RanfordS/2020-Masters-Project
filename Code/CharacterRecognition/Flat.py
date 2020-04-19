## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# short name
layers = tf.keras.layers

## Parameters
num_epochs = 200
use_custom = False

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')
#Does not work well
plot_filename = False#"ResultFlat.pgf"
data_filename = "DataFlat{}.csv"
data_stride = 1

tf.random.set_seed (123456)

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
    c0 = tf.constant (0.0)
    c1 = tf.constant (1.0)
    condition = tf.less (x, c0)

    # Heavy Squared: $H(x)x^2$
    #return tf.where (condition, c0, x*x)

    # Squared: $x^2$
    #return x*x

    # Heavy Root: $H(x)\sqrt{x}$
    #return tf.where (condition, c0, tf.math.sqrt (x))

    # Leaky ReLU: $(0.1 + 0.9H(x))x$
    #return tf.where (condition, tf.constant (0.1)*x, x)

    # Damppened Heavy: $H(x)(1 - e^{-x})
    #return tf.where (condition, c0, c1 - tf.math.exp (-x))

    # Heavy Exponential: $H(x)e^x$
    #return tf.where (condition, c0, tf.math.exp (x))

    # Smoothed ReLU $H(x)(1 - e^{-x})x$
    return tf.where (condition, c0, x - x*tf.math.exp (-x))

    # Heavy Arccos $\arccos(x)$
    #return tf.where (condition, c0, tf.math.acos (x))

act = use_custom and act_threshold or 'relu'

## Create model
model = tf.keras.models.Sequential (
[   layers.Flatten (input_shape=sample_shape)
,   layers.Dense (128, activation=act)#'relu' or act_threshhold
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
print ("initial loss:", result.history['loss'][0])
print ("initial vali:", result.history['val_loss'][0])
print ("final loss:  ", result.history['loss'][-1])
print ("final loss:  ", result.history['val_loss'][-1])

## Plot
if data_filename:
    for att in [["loss","Loss"],["val_loss","Vali"]]:
        with open (data_filename.format (att[1]), "w") as f:
            prop = result.history[att[0]]
            for i in range (0, num_epochs, data_stride):
                f.write ("{0:d},{1:f}\n".format (i, prop[i]))
plt.plot (range (num_epochs), result.history['loss'], label='loss')
plt.plot (range (num_epochs), result.history['val_loss'], label='validation')
plt.legend ()
plt.show ()

pred_train = np.argmax (model.predict (x_train), axis=1)
pred_test  = np.argmax (model.predict (x_test),  axis=1)
diff_train = pred_train - y_train
diff_test  = pred_test  - y_test
errs_train = np.sum (np.absolute (diff_train))
errs_test  = np.sum (np.absolute (diff_test))
print ("training size:    ", x_train.shape[0])
print ("training errors:  ", errs_train)
print ("validation size:  ", x_test.shape[0])
print ("validation errors:", errs_test)

## Show elements with predictions
predictions = model.predict (x_test)
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
    print ("{0}/{1:.2f}/{2}".format (y_pred, y_cert, y_test[i]))

if plot_filename:
    plt.savefig (plot_filename)
plt.show ()
