# UNUSED/FAILED
# Uses International Phonetic Alphabet spellings of words
# to predict language of origin.
# Data extracted from https://www.wiktionary.org/
# Problem: Network learns to always predict English

# Import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

#crop = 20

# Load file into to arrays
print ("~@> Loading File")
language = []
phonetic = []
with open ("phonetics.csv", "r") as f:
    while True:
        line = f.readline ()
        if line == "":
            break
        lang, phon = line.split (",", 1)
        language.append (lang)
        phonetic.append (phon.strip ('\n'))
        #if len (language) >= crop:
        #    break

# Parse inputs
print ("~@> Parsing")
num_examples = len (phonetic)
longest_word = max ([len (s) for s in phonetic])

for i in range (num_examples):
    phonetic[i] = phonetic[i].ljust (longest_word)

#for i in range (crop):
#    print ("'" + phonetic[i] + "'")

languages = sorted (set (language))
num_languages = len (languages)
language_to_index = {u:i for i, u in enumerate (languages)}
index_to_language = np.array (languages)

phonemes = sorted (set (''.join (phonetic)))
num_phonemes = len (phonemes)
phoneme_to_index = {u:i for i, u in enumerate (phonemes)}
index_to_phoneme = np.array (phonemes)

# Make dataset
print ("~@> Converting to Index")
language_index = [language_to_index[lang] for lang in language]
phonetic_index = [[phoneme_to_index[phon] for phon in word] for word in phonetic]

#language_index = tf.constant (language_index,
#                              shape = (num_examples, longest_word+1))
#phonetic_index = tf.constant (phonetic_index,
#                              shape = (num_examples,))

#print (phonetic_index[:20])

print ("~@> Converting to Dataset")
dataset = tf.data.Dataset.from_tensor_slices ((phonetic_index, language_index))

#for element in dataset.as_numpy_iterator():
#    print(element)

# Parameters

embedding_dim = num_phonemes #32 #num_phonemes*16
rnn_units = 128
epochs = 4#4*256

print ("Examples: ", num_examples)
print ("Langauges:", num_languages)
print ("Phonemes: ", num_phonemes)
print ("Longest:  ", longest_word)
print ("Embed Dim:", embedding_dim)

#embed = tf.keras.layers.Embedding (num_phonemes,
#                                   embedding_dim,
#                                   batch_input_shape = [10, None])
#rshp = tf.keras.layers.Reshape ((1,embedding_dim,1))
#lstm = tf.keras.layers.LSTM (rnn_units)
#for element in dataset.as_numpy_iterator ():
#    print ("\nStart\n")
#    e = element[0]
#    print (e)
#    e = embed (e)
#    print (e)
#    #e = rshp (e)
#    e = lstm (e)
#    print (e)
#    print ("\nEnd\n")

# Define the model

def build_model (batch_size):
    l = tf.keras.layers
    return tf.keras.Sequential (
    [   l.Input (shape = (None,))
    ,   l.Embedding (num_phonemes,
                     embedding_dim)#,
                     #batch_input_shape = [batch_size, None])
    ,   l.LSTM (rnn_units)
    ,   l.Dense (num_languages, activation='softmax')
    ])
print ("~@> Building Model")
model = build_model (64)
print ("~@> Compiling Model")
model.compile (optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.summary ()
print ("~@> Fitting to Data")
history = model.fit (phonetic_index,
                     language_index,
                     batch_size = 4096,
                     epochs = epochs)
print ("~@> Training Finished")
prediction = model.predict (phonetic_index)
diff = language_index == np.argmax (prediction, axis=1)
diff = np.sum (diff)
print ("errors", diff)
for i in range (50):
    print ("\ntest phonetic", phonetic[i])
    print ("act lang", language[i])
    g = np.argmax (prediction[i])
    print ("pred idx", g)
    print ("predlang", index_to_language[g])
    if g == language_index[i]:
        print ("~ correct")
    else:
        print ("~ incorrect")
