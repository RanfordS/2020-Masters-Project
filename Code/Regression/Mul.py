## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num = 20
eps = 10000
upper = 2

x = []
y = []
for i in range (num):
    a = i*upper/(num-1)
    for j in range (num):
        b = j*upper/(num-1)
        x.append (a)
        x.append (b)
        y.append (a*b)
x = np.array (x).reshape ((num**2, 2))
y = np.array (y)

layers = tf.keras.layers
model = tf.keras.Sequential (
[   layers.Dense (1, input_dim=2)
])

model.compile (optimizer="Adam",
               loss='mean_squared_error',
               metrics=[])
res = model.fit (x, y,
                 batch_size=num,
                 epochs=eps,
                 use_multiprocessing=True,
                 workers=12,
                 verbose=0)

print ("Initial loss:", res.history['loss'][0])
print ("Final loss:  ", res.history['loss'][-1])
