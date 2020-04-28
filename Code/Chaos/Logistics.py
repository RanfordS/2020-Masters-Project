## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## Settings

x_0 = 0.1        # starting value
num = 1000       # total number of samples
neurons = 64     # number of LSTM neurons
epochs = 10     # number of epochs

frame = 100      # number of samples in a sequence
batch = 20       # training batch size

tf.random.set_seed (123456)

save_dir = "./SaveLogistics"

## End of settings

def build_model (batch_size):
    l = tf.keras.layers
    return tf.keras.Sequential (
    [   l.LSTM (neurons, return_sequences = True, stateful = True)
    ,   l.Dense (1)
    ])

# Generate chaotic data

print ("Generating data")
x_all = [x_0]
for i in range (num-1):
    x_0 = 4*x_0*(1 - x_0)
    x_all.append (x_0)
x_all = np.array (x_all)

print ("Forming dataset")
dataset_x = tf.data.Dataset.from_tensor_slices (x_all)
sequences = dataset_x.batch (frame, drop_remainder = True)
print (sequences)
def split (chunk):
    return chunk[:-1], chunk[1:]
dataset = sequences.map (split)
print (dataset)
dataset = dataset.shuffle (10000).batch (batch, drop_remainder = True)
print (dataset)

# Training model

def loss_scc_log (labels, logits):
    scc = tf.keras.losses.sparse_categorical_crossentropy
    return scc (labels, logits, from_logits = True)

#tlstm = tf.keras.layers.LSTM(16,return_sequences=True,stateful=True)
#tin = np.random.uniform(size=(1,2,1))
#print (tlstm (tin))

print ("Building model")
model = build_model (batch)
model.compile (optimizer = 'adam', loss = loss_scc_log)
model.build (tf.TensorShape ([batch, 1]))
model.summary ()

print ("Training")
save_pre = os.path.join (save_dir, "ckpt_{epoch}")
save_call = tf.keras.callbacks.ModelCheckpoint
save_call = save_call (filepath = save_pre, save_weights_only = True)
history = model.fit (dataset, epochs = epochs, callbacks = [save_call])
