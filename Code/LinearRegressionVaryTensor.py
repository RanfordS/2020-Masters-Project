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

plt.plot (rx, ry, '.')
for i in range (1, 6):
  layers = tf.keras.layers
  model = tf.keras.Sequential (
  [   layers.Dense (i, input_dim=1, activation='tanh')
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

  print ("hidden:      ", i)
  print ("Initial loss:", res.history['loss'][0])
  print ("Final loss:  ", res.history['loss'][-1])

  py = model.predict (x)[:,0]
  plt.plot (x, py, label="hidden {0}".format (i))

plt.legend ()
plt.show ()
