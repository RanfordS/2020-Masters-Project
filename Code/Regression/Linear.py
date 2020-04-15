## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num = 100
lower = 0
upper = 2*np.pi
noise = 0.02

eps = 10000

x = np.array ([i/(num-1) for i in range (num)])*(upper-lower) + lower
y = np.sin (x)
rx = x + np.random.uniform (-noise, noise, num)
ry = y + np.random.uniform (-noise, noise, num)

layers = tf.keras.layers
model = tf.keras.Sequential (
[   layers.Dense (4, input_dim=1, activation='tanh')
,   layers.Dense (1)
])

model.compile (optimizer="Adam",
               loss='mean_squared_error',
               metrics=[])
res = model.fit (rx, ry,
                 batch_size=num,
                 epochs=eps,
                 use_multiprocessing=True,
                 workers=12,
                 verbose=0)

print ("Initial loss:", res.history['loss'][0])
print ("Final loss:  ", res.history['loss'][-1])

py = model.predict (x)[:,0]

plt.plot (rx, ry, '.')
plt.plot (x, py)
plt.show ()
