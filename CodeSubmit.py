#   <--- This is row 001
#
#
# =========================================================================================
#     Content:
#
#        # import ------------------------------------------------------- row 040
#        # Loading data ------------------------------------------------- row 070
#        # F1-score function -------------------------------------------- row 076
#        # 1. Remove row with NAs --------------------------------------- row 117
#        # 2. Use index represent string words -------------------------- row 147
#        # 3. Embedding ------------------------------------------------- row 287
#        # 4. Compare RNN and LSTM -------------------------------------- row 314
#        # 5. Compare embedding vector length --------------------------- row 407
#        # 6. Model 1 (Chapter 3.5.2 from my paper) --------------------- row 465
#        # 7. Model 2 (Chapter 3.5.2 from my paper) --------------------- row 514
#        # 8. Model 3 (Chapter 3.5.2 from my paper) --------------------- row 564
#        # 9. Models with high hidden units in LSTM layer --------------- row 613
#        # 10. Plot ----------------------------------------------------- row 685
#
# ========================================================================================
#
#
#
#
#
#
#
#
#
#
#
#
#
#   <--- This is row 035




# import
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import time
import os
import keras
from keras import preprocessing
from keras import models
from keras import layers
from keras import backend
from keras import optimizers
from keras import Sequential
from keras.datasets import mnist
from keras.datasets import imdb
from keras.layers import SimpleRNN
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.metrics import precision_recall_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Loading data
dfquora_test = pd.read_csv("/mnt/PyCharm_Project_1/Data/quora_test.csv")
dfquora_train = pd.read_csv("/mnt/PyCharm_Project_1/Data/quora_train.csv")

glove_dir = '/mnt/PyCharm_Project_1/Embedding'

# F1-score function
list_prec, list_recall, list_f1 = [], [], []

def precision(current, predict):
    curr_positive = backend.sum(
        backend.round(
            backend.clip(current*predict, 0, 1)
        )
    )
    pred_positive = backend.sum(
        backend.round(
            backend.clip(predict, 0, 1)
        )
    )
    p = curr_positive/(pred_positive+backend.epsilon())
    list_prec.append(p)
    return p

def recall(current, predict):
    curr_positive = backend.sum(
        backend.round(
            backend.clip(current*predict, 0, 1)
        )
    )
    pred_positive = backend.sum(
        backend.round(
            backend.clip(current, 0, 1)
        )
    )
    r = curr_positive/(pred_positive + backend.epsilon())
    list_recall.append(r)
    return r

def f1_score(current, predict):
    p = precision(current, predict)
    r = recall(current, predict)
    score = 2 * ( (p*r) / (p+r+backend.epsilon()) )
    list_f1.append(score)
    return score


# 1. Remove row with NAs
## 1.1. For training data
train_num_na_01, train_num_na_02 = 0, 0
train_list_na_index_01, train_list_na_index_02 = [], []

for i in range(len(dfquora_train)):
    if pd.isna(dfquora_train.loc[i, 'question1']) == True:
        train_num_na_01 += 1
        train_list_na_index_01.append(i)
    if pd.isna(dfquora_train.loc[i, 'question2']) == True:
        train_num_na_02 += 1
        train_list_na_index_02.append(i)

dfquora_train_dropna = dfquora_train.dropna()

## 1.2. For test data
test_num_na_01, test_num_na_02 = 0, 0
test_list_na_index_01, test_list_na_index_02 = [], []

for i in range(len(dfquora_test)):
    if pd.isna(dfquora_test.loc[i, 'question1']) == True:
        test_num_na_01 += 1
        test_list_na_index_01.append(i)
    if pd.isna(dfquora_test.loc[i, 'question2']) == True:
        test_num_na_02 += 1
        test_list_na_index_02.append(i)

dfquora_test_dropna = dfquora_test.dropna()


# 2. Use index represent string words
train_ques_01, train_ques_02, train_labels = [], [], []

for i in range(len(dfquora_train_dropna)+1):
    if i not in train_list_na_index_01 and i not in train_list_na_index_02:
        train_ques_01.append(dfquora_train_dropna.loc[i, 'question1'])
        train_ques_02.append(dfquora_train_dropna.loc[i, 'question2'])
        train_labels.append(dfquora_train_dropna.loc[i, 'is_duplicate'])

test_ques_01, test_ques_02, test_labels = [], [], []
for i in range(len(dfquora_test_dropna)+1):
    if i not in test_list_na_index_01 and i not in test_list_na_index_02:
        test_ques_01.append(dfquora_test_dropna.loc[i, 'question1'])
        test_ques_02.append(dfquora_test_dropna.loc[i, 'question2'])
        test_labels.append(dfquora_test_dropna.loc[i, 'is_duplicate'])

## 2.1. Token
### 2.1.1 Overal
#### train
train_ques_all = train_ques_01+train_ques_02
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_ques_all)
train_seq_all = tokenizer.texts_to_sequences(train_ques_all)

train_word_index_all = tokenizer.word_index
print('Question train overally has %s unique tokens.' % len(train_word_index_all))

#### test
test_ques_all = test_ques_01+test_ques_02
tokenizer_test = Tokenizer()
tokenizer_test.fit_on_texts(test_ques_all)
test_seq_all = tokenizer_test.texts_to_sequences(test_ques_all)

test_word_index_all = tokenizer.word_index
print('Question test overally has %s unique tokens.' % len(test_word_index_all))

#### Overal
ques_all = train_ques_all+test_ques_all
tokenizer_all = Tokenizer()
tokenizer_all.fit_on_texts(ques_all)
seq_all = tokenizer.texts_to_sequences(ques_all)
all_word_index_all = tokenizer_all.word_index
print('Question all overally has %s unique tokens.' % len(all_word_index_all))

### 2.1.2. Question 01
## train
tokenizer01 = Tokenizer()
tokenizer01.fit_on_texts(train_ques_01)
train_seq_01 = tokenizer01.texts_to_sequences(train_ques_01)

train_word_index_01 = tokenizer01.word_index
print('Question 1 has %s unique tokens.' % len(train_word_index_01))

## test
tokenizer01_test = Tokenizer()
tokenizer01_test.fit_on_texts(test_ques_01)
test_seq_01 = tokenizer01_test.texts_to_sequences(test_ques_01)

test_word_index_01 = tokenizer01_test.word_index
print('Question test 1 has %s unique tokens.' % len(test_word_index_01))

### 2.1.3. Question 02
#### train
tokenizer02 = Tokenizer()
tokenizer02.fit_on_texts(train_ques_02)
train_seq_02 = tokenizer01.texts_to_sequences(train_ques_02)

train_word_index_02 = tokenizer02.word_index
print('Question 2 has %s unique tokens.' % len(train_word_index_02))

#### test
tokenizer02_test = Tokenizer()
tokenizer02_test.fit_on_texts(test_ques_02)
test_seq_02 = tokenizer02_test.texts_to_sequences(test_ques_02)

test_word_index_02 = tokenizer02_test.word_index
print('Question test 2 has %s unique tokens.' % len(test_word_index_02))

## 2.2. Distribution of sequence length
def distribution_of_sequence_length_4_ques(input_list):
    list_length = []
    for i in range(len(input_list)):
        list_length.append(len(input_list))
    return list_length

length_distribution = distribution_of_sequence_length_4_ques(train_seq_01+test_seq_01)

plt.hist(length_distribution, bins=80, histtype="stepfilled")
plt.xlabel('Length of sequences for question 01')
plt.axis([0, 300, 0, 130000])
plt.show()

plt.clf()

length_distribution = distribution_of_sequence_length_4_ques(train_seq_02+test_seq_02)

plt.hist(length_distribution, bins=80, histtype="stepfilled")
plt.xlabel('Length of sequences for question 02')
plt.axis([0, 300, 0, 130000])
plt.show()

## 2.3. Let sequence has same length
### train
train_data_01 = pad_sequences(train_seq_01, maxlen=40)
train_data_02 = pad_sequences(train_seq_02, maxlen=50)
train_labels = np.asarray(train_labels)

print('Shape of data 1 tensor:', train_data_01.shape,)
print('Shape of data 2 tensor:', train_data_02.shape,)
print('Shape of label tensor:', train_labels.shape)

### test
test_data_01 = pad_sequences(test_seq_01, maxlen=40)
test_data_02 = pad_sequences(test_seq_02, maxlen=50)
test_labels = np.asarray(test_labels)

print('Shape of test data 1 tensor:', test_data_01.shape,)
print('Shape of test data 2 tensor:', test_data_02.shape,)
print('Shape of test label tensor:', test_labels.shape)

## 2.4. Shuffle the data and diivide them into train set and validation set
indices = np.arange(train_data_01.shape[0])
np.random.shuffle(indices)

train_data_shuffle_01 = train_data_01[indices]
train_data_shuffle_02 = train_data_02[indices]
train_labels_shuffle = train_labels[indices]

train_samples_num = int(0.8*len(train_data_01)) - 1
val_samples_num = len(train_data_01) - train_samples_num -1

training_samples_text_01 = train_data_shuffle_01[:train_samples_num]
training_samples_text_02 = train_data_shuffle_02[:train_samples_num]
training_samples_labels = train_labels_shuffle[:train_samples_num]

val_samples_text_01 = train_data_shuffle_01[train_samples_num:]
val_samples_text_02 = train_data_shuffle_02[train_samples_num:]
val_samples_labels = train_labels_shuffle[train_samples_num:]


# 3. Embedding
embedding_dim = 200
# embedding_dim = 300

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding='UTF-8')
# f = open(os.path.join(glove_dir, 'glove.twitter.27B.100d.txt'), encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

## 3.1. Overal
no_shown_words = []

embedding_matrix_all = np.zeros((len(all_word_index_all)+1, embedding_dim))
for word, i in all_word_index_all.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix_all[i] = embedding_vector
    else:
        no_shown_words.append(word)


# 4. Compare RNN and LSTM
lstm100 = layers.LSTM(100)
rnn100 = layers.SimpleRNN(100)

## 4.1. RNN
### 4.1.1. Model branch 02
input_training_text_01 = Input(shape=(len(training_samples_text_01[0]),), name='text01')
embedding_text_01 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_01)
encoded_text_01 = lstm100(embedding_text_01)

### 4.1.2. Model branch 02
input_training_text_02 = Input(shape=(len(training_samples_text_02[0]),), name='text02')
embedding_text_02 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_02)
encoded_text_02 = lstm100(embedding_text_02)

### 4.3.3. Concatenate
concatenated = layers.concatenate([encoded_text_01, encoded_text_02], axis=-1)

### 4.1.4. Out put
dropout = layers.Dropout(0.5)(concatenated)
results = layers.Dense(1, activation='sigmoid')(dropout)

### 4.1.5. Final model
model = Model([input_training_text_01, input_training_text_02], results)
model.summary()

### 4.1.6. Embedding setting
model.layers[2].set_weights([embedding_matrix_all])
model.layers[2].trainable = False

model.layers[3].set_weights([embedding_matrix_all])
model.layers[3].trainable = False

## 4.1.7. Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[f1_score, 'acc'])

history_len_lstm100_em200 = model.fit([training_samples_text_01, training_samples_text_02],
                                       training_samples_labels,
                                       epochs=10,
                                       batch_size=256,
                                       validation_data=([val_samples_text_01, val_samples_text_02],
                                                        val_samples_labels))

model_len_lstm100_em200 = model
history_dict = history_len_lstm100_em200.history

## 4.2. LSTM
### 4.2.1. Model branch 02
input_training_text_01 = Input(shape=(len(training_samples_text_01[0]),), name='text01')
embedding_text_01 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_01)
encoded_text_01 = lstm100(embedding_text_01)

### 4.2.2. Model branch 02
input_training_text_02 = Input(shape=(len(training_samples_text_02[0]),), name='text02')
embedding_text_02 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_02)
encoded_text_02 = lstm100(embedding_text_02)

### 4.2.3. Concatenate
concatenated = layers.concatenate([encoded_text_01, encoded_text_02], axis=-1)

### 4.2.4. Out put
dropout = layers.Dropout(0.5)(concatenated)
results = layers.Dense(1, activation='sigmoid')(dropout)

### 4.2.5. Final model
model = Model([input_training_text_01, input_training_text_02], results)
model.summary()

### 4.2.6. Embedding setting
model.layers[2].set_weights([embedding_matrix_all])
model.layers[2].trainable = False

model.layers[3].set_weights([embedding_matrix_all])
model.layers[3].trainable = False

## 4.2.7. Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[f1_score, 'acc'])

history_len_lstm100_em200 = model.fit([training_samples_text_01, training_samples_text_02],
                                       training_samples_labels,
                                       epochs=10,
                                       batch_size=256,
                                       validation_data=([val_samples_text_01, val_samples_text_02],
                                                        val_samples_labels))

model_len_lstm100_em200 = model
history_dict = history_len_lstm100_em200.history


# 5. Compare embedding vector length 
## 5.1. Embedding 
embedding_dim_300 = 300

embeddings_index_300 = {}
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index_300[word] = coefs
f.close()

## 5.2. Model with 300 of embedding vector length
### 5.2.1. Model branch 02
input_training_text_01 = Input(shape=(len(training_samples_text_01[0]),), name='text01')
embedding_text_01 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_01)
encoded_text_01 = lstm100(embedding_text_01)

### 5.2.2. Model branch 02
input_training_text_02 = Input(shape=(len(training_samples_text_02[0]),), name='text02')
embedding_text_02 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_02)
encoded_text_02 = lstm100(embedding_text_02)

### 5.2.3. Concatenate
concatenated = layers.concatenate([encoded_text_01, encoded_text_02], axis=-1)

### 5.2.4. Out put
dropout = layers.Dropout(0.5)(concatenated)
results = layers.Dense(1, activation='sigmoid')(dropout)

### 5.2.5. Final model
model = Model([input_training_text_01, input_training_text_02], results)
model.summary()

### 5.2.6. Embedding setting
model.layers[2].set_weights([embedding_matrix_all])
model.layers[2].trainable = False

model.layers[3].set_weights([embedding_matrix_all])
model.layers[3].trainable = False

## 5.2.7. Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[f1_score, 'acc'])

history_len_lstm100_em300 = model.fit([training_samples_text_01, training_samples_text_02],
                                       training_samples_labels,
                                       epochs=10,
                                       batch_size=256,
                                       validation_data=([val_samples_text_01, val_samples_text_02],
                                                        val_samples_labels))

model_len_lstm100_em300 = model
history_dict = history_len_lstm100_em300.history


# 6. Model 1 (Chapter 3.5.2 from my paper)
lstm = layers.LSTM(200)
## 6.1. Model branch 01
input_training_text_01 = Input(shape=(len(training_samples_text_01[0]),), name='text01')
embedding_text_01 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_01)
encoded_text_01 = lstm(embedding_text_01)

## 6.2. Model branch 02
input_training_text_02 = Input(shape=(len(training_samples_text_02[0]),), name='text02')
embedding_text_02 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_02)
encoded_text_02 = lstm(embedding_text_02)

## 6.3. Concatenate
concatenated = layers.concatenate([encoded_text_01, encoded_text_02], axis=-1)

## 6.4. Out put
dropout = layers.Dropout(0.4)(concatenated)
results = layers.Dense(1, activation='sigmoid')(dropout)

## 6.5. Final model
model = Model([input_training_text_01, input_training_text_02], results)
model.summary()

## 6.6. Embedding setting
model.layers[2].set_weights([embedding_matrix_all])
model.layers[2].trainable = False

model.layers[3].set_weights([embedding_matrix_all])
model.layers[3].trainable = False

## 6.7. Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[f1_score, 'acc'])

history_len_lstm200_em200 = model.fit([training_samples_text_01, training_samples_text_02],
                                       training_samples_labels,
                                       epochs=20,
                                       batch_size=256,
                                       validation_data=([val_samples_text_01, val_samples_text_02],
                                                        val_samples_labels))

model_len_lstm200_em200 = model
history_dict = history_len_lstm200_em200.history

## 6.8. Test
results_predict = model_len_lstm200_em200.evaluate([test_data_01, test_data_02], test_labels)


# 7. Model 2 (Charpter 3.5.2 from my paper)
lstm = layers.LSTM(300)
## 7.1. Model branch 01
input_training_text_01 = Input(shape=(len(training_samples_text_01[0]),), name='text01')
embedding_text_01 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_01)
encoded_text_01 = lstm(embedding_text_01)

## 7.2. Model branch 02
input_training_text_02 = Input(shape=(len(training_samples_text_02[0]),), name='text02')
embedding_text_02 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_02)
encoded_text_02 = lstm(embedding_text_02)

## 7.3. Concatenate
concatenated = layers.concatenate([encoded_text_01, encoded_text_02], axis=-1)

## 7.4. Out put
relu = layers.Dense(300, activation='relu')(concatenated)
dropout = layers.Dropout(0.2)(relu)
results = layers.Dense(1, activation='sigmoid')(dropout)

## 7.5. Final model
model = Model([input_training_text_01, input_training_text_02], results)
model.summary()

## 7.6. Embedding setting
model.layers[2].set_weights([embedding_matrix_all])
model.layers[2].trainable = False

model.layers[3].set_weights([embedding_matrix_all])
model.layers[3].trainable = False

## 7.7. Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[f1_score, 'acc'])

history_len_lstm300_relu_em200 = model.fit([training_samples_text_01, training_samples_text_02],
                                       training_samples_labels,
                                       epochs=20,
                                       batch_size=256,
                                       validation_data=([val_samples_text_01, val_samples_text_02],
                                                        val_samples_labels))

model_len_lstm300_relu_em200 = model
history_dict = history_len_lstm300_relu_em200.history

## 7.8. Test
results_predict = model_len_lstm300_relu_em200.evaluate([test_data_01, test_data_02], test_labels)


# 8. Model 3 (Charpter 3.5.2 from my paper)
lstm = layers.LSTM(300)
## 8.1. Model branch 01
input_training_text_01 = Input(shape=(len(training_samples_text_01[0]),), name='text01')
embedding_text_01 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_01)
encoded_text_01 = lstm(embedding_text_01)

## 8.2. Model branch 02
input_training_text_02 = Input(shape=(len(training_samples_text_02[0]),), name='text02')
embedding_text_02 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_02)
encoded_text_02 = lstm(embedding_text_02)

## 8.3. Concatenate
concatenated = layers.concatenate([encoded_text_01, encoded_text_02], axis=-1)

## 8.4. Out put
dropout = layers.Dropout(0.2)(concatenated)
results = layers.Dense(1, activation='sigmoid')(dropout)

## 8.5. Final model
model = Model([input_training_text_01, input_training_text_02], results)
model.summary()

## 8.6. Embedding setting
model.layers[2].set_weights([embedding_matrix_all])
model.layers[2].trainable = False

model.layers[3].set_weights([embedding_matrix_all])
model.layers[3].trainable = False

## 8.7. Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[f1_score, 'acc'])

history_len_lstm300_em200 = model.fit([training_samples_text_01, training_samples_text_02],
                                       training_samples_labels,
                                       epochs=20,
                                       batch_size=256,
                                       validation_data=([val_samples_text_01, val_samples_text_02],
                                                        val_samples_labels))

model_len_lstm300_relu_em200 = model
history_dict = history_len_lstm300_relu_em200.history

## 8.8. Test
results_predict = model_len_lstm300_relu_em200.evaluate([test_data_01, test_data_02], test_labels)


# 9. Models with high hidden units in LSTM layer
def build_model(input_num):
    '''
    input_num is the number of hidden unites in LSTM layers, it should be a int variable.
    Due to lack of computing ability, the input_num should be less than 3000. 
    If input_num is high than 3000, "Process finished with exit code 137" may happen.
    '''
    input_num = int(input_num)
    lstm = layers.LSTM(input_num)
    ## Model branch 01
    input_training_text_01 = Input(shape=(len(training_samples_text_01[0]),), name='text01')
    embedding_text_01 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_01)
    encoded_text_01 = lstm(embedding_text_01)

    ## Model branch 02
    input_training_text_02 = Input(shape=(len(training_samples_text_02[0]),), name='text02')
    embedding_text_02 = layers.Embedding(len(embedding_matrix_all), embedding_dim)(input_training_text_02)
    encoded_text_02 = lstm(embedding_text_02)

    ## Concatenate
    concatenated = layers.concatenate([encoded_text_01, encoded_text_02], axis=-1)

    ## Out put
    dropout = layers.Dropout(0.2)(concatenated)
    results = layers.Dense(1, activation='sigmoid')(dropout)

    ## Final model
    model = Model([input_training_text_01, input_training_text_02], results)
    model.summary()

    ## Embedding setting
    model.layers[2].set_weights([embedding_matrix_all])
    model.layers[2].trainable = False

    model.layers[3].set_weights([embedding_matrix_all])
    model.layers[3].trainable = False

    ## Training
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])

    history = model.fit([training_samples_text_01, training_samples_text_02],
                        training_samples_labels,
                        epochs=20,
                        batch_size=256,
                        validation_data=([val_samples_text_01, val_samples_text_02],
                                         val_samples_labels))

    return model, history

## 9.1. 600 hidden units
model_len_lstm600_em200, history_len_lstm600_em200 = build_model(600)
results_len_lstm600_em200 = model_len_lstm600_em200.evaluate([test_data_01, test_data_02], test_labels)

## 9.2. 1000 hidden units
model_len_lstm1000_em200, history_len_lstm1000_em200 = build_model(1000)
results_len_lstm1000_em200 = model_len_lstm1000_em200.evaluate([test_data_01, test_data_02], test_labels)

## 9.3. 1200 hidden units
model_len_lstm1200_em200, history_len_lstm1200_em200 = build_model(1200)
results_len_lstm1200_em200 = model_len_lstm1200_em200.evaluate([test_data_01, test_data_02], test_labels)

## 9.4. 2000 hidden units
model_len_lstm2000_em200, history_len_lstm2000_em200 = build_model(2000)
results_len_lstm2000_em200 = model_len_lstm2000_em200.evaluate([test_data_01, test_data_02], test_labels)

## 9.5. 2500 hidden units
model_len_lstm2500_em200, history_len_lstm2500_em200 = build_model(2500)
results_len_lstm2500_em200 = model_len_lstm2500_em200.evaluate([test_data_01, test_data_02], test_labels)


# 10. Plot
## 10.1. Loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.axis([0, len(epochs), 0, 1])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

## 10.2. Accuracy
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.axis([0, len(epochs), 0, 1])
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

## 10.3. P_R plot
label_test_predict = model.predict([test_data_01, test_data_02])
test_precision, test_recall, test_thresholds = precision_recall_curve(test_labels, label_test_predict)
plt.plot()
plt.plot(test_recall, test_precision)
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
