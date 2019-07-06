# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Ignore warinings
import warnings
warnings.filterwarnings("ignore")

# Load preprocessed csv file using pandas.
file_path = './cleaned_reviews.csv'
data = pd.read_csv(file_path)

# Analyze the length of the reviews and the summary to get an overall idea about the distribution of length of the text.
text_word_count = []
summary_word_count = []

## Populate the lists with sentence lengths
#for i in range(0, len(data)):
#    text_word_count.append(len(data['text'][i].split()))
#    s = str(data['summary'][i])
#    summary_word_count.append(len(s.split()))
#
#length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
#length_df.hist(bins = 30)
#plt.show()

# 98% of the summaries have length below 8
# ~100% of the text have length below 20
max_text_len = 20
max_summary_len = 8

# Selecting the reviews and summaries whose length falls below or equal
# to max_text_len and max_summary_len
cleaned_text = np.array(data['text'])
cleaned_summary = np.array(data['summary'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if(len(str(cleaned_summary[i]).split()) <= max_summary_len
       and len(cleaned_text[i].split()) <= max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

# Dataset used for training and testing model.       
df=pd.DataFrame({'text': short_text, 'summary' : short_summary})

# Adding START(stoken) and END(etoken) special tokens at the beginning and
# end of the summary.
df['summary'] = df['summary'].apply(lambda x : 'stoken '+ str(x) + ' etoken')

# Spliting dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(df['text']), 
                                                    np.array(df['summary']),
                                                    test_size = 0.1,
                                                    random_state = 0,
                                                    shuffle = True)

# Preparing Tokenizer
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

# Prepare a tokenizer for text on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(X_train))

# Find rare words and its total coverage in the entire text.
threshold = 4 # means word whose count is below 4 is considered as a rare word.
cnt = 0       
total_cnt = 0 
freq = 0
total_freq = 0

for key,value in x_tokenizer.word_counts.items():
    total_cnt += 1
    total_freq = total_freq + value
    if(value < threshold):
        cnt += 1
        freq = freq + value
    
print("% of rare words in vocabulary:",(cnt / total_cnt) * 100)
print("Total Coverage of rare words:",(freq / total_freq) * 100)

# Prepare a tokenizer for text on training data (remove uncommon words)
x_tokenizer = Tokenizer(num_words = total_cnt - cnt)
x_tokenizer.fit_on_texts(list(X_train))

# Convert text sequences into integer sequences
X_train_seq =   x_tokenizer.texts_to_sequences(X_train) 
X_test_seq  =   x_tokenizer.texts_to_sequences(X_test)

# Padding zero upto maximum length
X_train =   pad_sequences(X_train_seq,  maxlen = max_text_len, padding = 'post')
X_test  =   pad_sequences(X_test_seq, maxlen = max_text_len, padding = 'post')

# Size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1



# Pepare a tokenizer for summary on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_train))

# Find Rarewords and its Coverage
threshold = 8 
cnt = 0
total_cnt = 0
freq = 0
total_freq = 0

for key,value in y_tokenizer.word_counts.items():
    total_cnt += 1
    total_freq += value
    if(value < threshold):
        cnt += 1
        freq += value
    
print("% of rare words in vocabulary:",(cnt / total_cnt) * 100)
print("Total Coverage of rare words:",(freq / total_freq) * 100)

# Define tokenizer with top most common words for summary. 
# Prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words = total_cnt - cnt) 
y_tokenizer.fit_on_texts(list(y_train))

# Convert text sequences into integer sequences
y_train_seq =   y_tokenizer.texts_to_sequences(y_train) 
y_test_seq  =   y_tokenizer.texts_to_sequences(y_test) 

# Padding zero upto maximum length
y_train =   pad_sequences(y_train_seq, maxlen = max_summary_len, padding = 'post')
y_test  =   pad_sequences(y_test_seq, maxlen = max_summary_len, padding = 'post')

# Size of test data vocabulary
y_voc  =   y_tokenizer.num_words + 1


# Deleting the rows that contain only START and END tokens.
ind = []
for i in range(len(y_train)):
    cnt = 0
    for j in y_train[i]:
        if j != 0:
            cnt += 1
    if(cnt == 2):
        ind.append(i)

y_train = np.delete(y_train, ind, axis = 0)
X_train = np.delete(X_train, ind, axis = 0)

# Deleting the rows that contain only START and END tokens.
ind = []
for i in range(len(y_test)):
    cnt = 0
    for j in y_test[i]:
        if j != 0:
            cnt += 1
    if(cnt == 2):
        ind.append(i)

y_test = np.delete(y_test, ind, axis = 0)
X_test = np.delete(X_test, ind, axis = 0)


# Build the dictionary to convert the index to word for target and source vocabulary
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index


# Library for building model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model

# Building a 3 stacked LSTM for the encoder:
from keras import backend as K 
K.clear_session()

# Config
latent_dim = 300
embedding_dim = 100

# Encoder
encoder_inputs = Input(shape = (max_text_len,))

# Embedding layer
encoder_embedding =  Embedding(x_voc, embedding_dim, trainable = True)(encoder_inputs)

# Encoder lstm 1
encoder_lstm1 = LSTM(latent_dim, return_sequences = True, return_state = True, dropout = 0.4,
                     recurrent_dropout = 0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)

# Encoder lstm 2
encoder_lstm2 = LSTM(latent_dim, return_sequences = True, return_state = True, dropout = 0.4,
                     recurrent_dropout =0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# Encoder lstm 3
encoder_lstm3 = LSTM(latent_dim, return_state = True, return_sequences = True,
                     dropout = 0.4, recurrent_dropout = 0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape = (None,))

# Embedding layer
#decoder_embedding = Embedding(y_voc, embedding_dim, trainable = True)(decoder_inputs)
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable = True)
decoder_embedding = dec_emb_layer(decoder_inputs)

# Decoder lstm
decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True,
                    dropout = 0.4, recurrent_dropout = 0.2)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(decoder_embedding,
                                                                      initial_state = [state_h, state_c])

# Third party implementation
from attention import AttentionLayer

# Attention layer
attn_layer = AttentionLayer(name = 'attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis = -1, name = 'concat_layer')([decoder_outputs, attn_out])

# Dense layer
decoder_dense =  TimeDistributed(Dense(y_voc, activation = 'softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()


# Compile
model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy')

# Use early stopping on validation loss start increasing.
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1,
                   patience = 2, restore_best_weights = True)

# Use checkpoint to save model after each epoch
from tensorflow.keras.callbacks import ModelCheckpoint
filepath = "weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose = 1,
                save_best_only = False, save_weights_only = False,
                mode ='min', period = 1)
callbacks_list = [checkpoint, es]

# Train the model on a batch size of 128 and
# validate it on the holdout set(which is 10% of our dataset):
hist = model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0],
                    y_train.shape[1], 1)[:,1:], epochs = 25, 
                    callbacks = callbacks_list, batch_size = 128,
                    validation_data = ([X_test, y_test[:,:-1]], 
                    y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))

# Save Model
model.save('text_summarization_model.hdf5')

#---------------------------------------------------------------------------------

# Load model
from tensorflow.keras.models import load_model
model = load_model('text_summarization_model.hdf5')

# Plot a few diagnostic plots to understand the behavior of the model over time
plt.plot(hist.history['loss'], label = 'train')
plt.plot(hist.history['val_loss'], label = 'test')
plt.legend()
plt.show()


# Set up the inference for the encoder and decoder

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs = encoder_inputs,
                      outputs = [encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

# Get the embeddings of the decoder sequence
decoder_emb2 = decoder_embedding
# To predict the next word in the sequence, set the initial states to the
#  states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_emb2,
                initial_state = [decoder_state_input_h, decoder_state_input_c])

# Attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input,
                                            decoder_outputs2])
decoder_inf_concat = Concatenate(axis = -1,
                                 name = 'concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h,
    decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])
 
    
# Defining a function below which is the implementation of the inference process
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['stoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='etoken'):
            decoded_sentence += ' '+ sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'etoken' or
            len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

# Defining the functions to convert an integer sequence to a word sequence for
# summary as well as the reviews
def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if((i != 0 and i != target_word_index['stoken']) and
           i != target_word_index['etoken']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString

def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if(i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString

# Here are a few summaries generated by the model:
for i in range(0, 100):
    print("Review:",seq2text(X_train[i]))
    print("Original summary:",seq2summary(y_train[i]))
    print("Predicted summary:",decode_sequence(X_train[i].reshape(1, max_text_len)))
    print("\n")
