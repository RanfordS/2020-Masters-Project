## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
layers = tf.keras.layers

num_epochs = 50

## Load data
char_set = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = char_set.load_data ()

shape_t = x_train.shape
x_train = x_train.reshape (shape_t[0],shape_t[1],shape_t[2],1)
shape_v = x_test.shape
x_test  = x_test.reshape  (shape_v[0],shape_v[1],shape_v[2],1)

print ("Reshaped")

sample_size  = x_train.shape[0]
sample_shape = x_train[0].shape

## Normalize
x_train = x_train/255.0
x_test  = x_test /255.0

## Create model
model = tf.keras.models.Sequential ()
model.add (layers.Conv2D (4, (7,7), activation='relu', input_shape=sample_shape))
model.add (layers.Conv2D (8, (7,7), activation='relu'))
model.add (layers.Flatten ())
model.add (layers.Dense (10, activation="softmax"))

model.compile (optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=[])

## Train
result = model.fit (x_train, y_train,
                    epochs=num_epochs,
                    batch_size=600,
                    verbose=1,
                    validation_data=(x_test, y_test))



print ("initial loss:", result.history['loss'][0])
print ("initial vali:", result.history['val_loss'][0])
print ("final loss:  ", result.history['loss'][-1])
print ("final loss:  ", result.history['val_loss'][-1])
pred_train = np.argmax (model.predict (x_train), axis=1)
pred_test  = np.argmax (model.predict (x_test),  axis=1)
diff_train = pred_train - y_train
diff_test  = pred_test  - y_test
errs_train = np.sum (np.absolute (diff_train))
errs_test  = np.sum (np.absolute (diff_test))
print ("training size:    ", x_train.shape[0])
print ("training errors:  ", errs_train)
print ("training %err:    ", 100*errs_train/x_train.shape[0])
print ("validation size:  ", x_test.shape[0])
print ("validation errors:", errs_test)
print ("validation %err:  ", 100*errs_test/x_test.shape[0])

## Plot
epoch = range (num_epochs)
plt.plot (epoch, result.history['loss'], label='loss')
plt.plot (epoch, result.history['val_loss'], label='validation')
plt.legend ()
plt.show ()


