import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import numpy as np

text_path = tf.keras.utils.get_file ('shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open (text_path, 'rb').read ().decode (encoding = 'utf-8')



# process
vocab = sorted (set (text))
vocab_size = len (vocab)
print ("Unique: {}".format (vocab_size))
char_to_index = {u:i for i, u in enumerate (vocab)}
index_to_char = np.array (vocab)
text_as_index = np.array ([char_to_index[c] for c in text])

sequence_length = 100
examples_per_epoch = len (text)//(sequence_length + 1)
char_dataset = tf.data.Dataset.from_tensor_slices (text_as_index)

sequences = char_dataset.batch (sequence_length + 1, drop_remainder = True)

def split_input_target (chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map (split_input_target)

batch_size = 64
buffer_size = 10000
embedding_dim = 256
rnn_units = 128#1024
epochs = 20
do_train = False
checkpoint_dir = "./checkpoints"

dataset = dataset.shuffle (buffer_size).batch (batch_size, drop_remainder = True)

def build_model (batch_size):
    return tf.keras.Sequential (
    [   tf.keras.layers.Embedding (vocab_size,
                                   embedding_dim,
                                   batch_input_shape = [batch_size, None])
    #,   tf.keras.layers.GRU (rnn_units,
    #                         return_sequences = True,
    #                         stateful = True,
    #                         recurrent_initializer = 'glorot_uniform')
    ,   tf.keras.layers.LSTM (rnn_units,
                              return_sequences = True,
                              stateful = True)
    ,   tf.keras.layers.Dense (vocab_size)
    ])

def logits_loss (labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy (labels, logits,
                                                            from_logits = True)

if do_train:
    model = build_model (batch_size)
    model.compile (optimizer = 'adam', loss = logits_loss)

    checkpoint_pre = os.path.join (checkpoint_dir, "ckpt_{epoch}")
    checkpoint_call = tf.keras.callbacks.ModelCheckpoint (filepath = checkpoint_pre,
                                                          save_weights_only = True)

    history = model.fit (dataset, epochs = epochs, callbacks = [checkpoint_call])

## Predict

model = build_model (1)
model.load_weights (tf.train.latest_checkpoint (checkpoint_dir))
model.build (tf.TensorShape ([1, None]))

start_string = u"RANFORD:\nWhere art thou!"
num_generate = 1000
input_eval = [char_to_index[c] for c in start_string]
input_eval = tf.expand_dims (input_eval, 0)
text_generated = []
temperature = 1.0

model.summary ()

model.reset_states ()
#for i in range (num_generate):
i = num_generate
last = "a"
while i > 0 or text_generated[-1] not in ".?!":
    i -= 1
    predictions = model (input_eval)
    predictions = tf.squeeze (predictions, 0)
    predictions = predictions / temperature
    predicted_id = tf.random.categorical (predictions, num_samples = 1)[-1,0].numpy ()
    input_eval = tf.expand_dims ([predicted_id], 0)
    text_generated.append (index_to_char[predicted_id])
print (start_string + ''.join (text_generated))
