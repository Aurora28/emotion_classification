from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Model,Sequential, load_model
from keras.layers import (Bidirectional, concatenate, Conv1D, Dense,BatchNormalization,
                          Dropout, Embedding, GlobalMaxPooling1D, Input,
                          LSTM, TimeDistributed, Activation, Flatten)
from keras.callbacks import EarlyStopping
from gensim.corpora.dictionary import Dictionary
from gensim.models import word2vec
import xml.etree.ElementTree as ET
import pickle
import json
import jieba

import numpy as np
import pandas as pd
np.random.seed(1337) 

f = open("model/MyWord2vec_model_5.pkl", 'rb')  
index_dict = pickle.load(f)  # word: index
word_vectors = pickle.load(f)  #word: vector
new_dic = index_dict

n_symbols = len(index_dict) + 1  # amount of words
embedding_weights = np.zeros((n_symbols, 100))  
for w, index in index_dict.items(): 
    embedding_weights[index, :] = word_vectors[w]  # word vectors' metrix, embedding_weight[0]=0 (index begins from 0)

with open('data/stcEmoji_new.json', encoding='utf-8') as file:
    data = file.read()
    datajson = json.loads(data)
    print('all data:', len(datajson))
datajson = pd.DataFrame(datajson)

sentences = list(datajson[:]['content'])
y = list(datajson[:]['label'])
e = datajson[:]['emoji_vec']
len_e = len(list(e)[0])
e = np.array(list(e))
print(e.shape)

y = np_utils.to_categorical(np.array(y))
x = []
maxlen = 0
for s in sentences:    
    sen = []

    words = jieba.lcut(s)
    if len(words) > maxlen:
        maxlen = len(words)

    for w in words:
        try:
            sen.append(index_dict[w])
        except:
            sen.append(0)
    x.append(sen)
x = sequence.pad_sequences(np.array(x), maxlen = maxlen)
x = np.concatenate([x, e], axis = 1)  

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1)
print(train_x.shape)
print(train_y.shape)
maxlen += len_e
print(maxlen)

class BaseBiLSTM(object):
    def __init__(self, vocabulary_size, max_sentence_length, labels,
                 embedding_weights, embedding_size=100):
        self.model = None
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_sentence_length = max_sentence_length
        self.embedding_weights = embedding_weights
        self.labels = labels
        self.n_labels = 6
    
    def add_input_layer(self):
        return Input(shape=(self.max_sentence_length, ))
        
    def add_embedding_layer(self, layers):
        layers = Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_size,
            weights = [self.embedding_weights],
            input_length = self.max_sentence_length)(layers)
        return layers
    
    def add_recurrent_layer(self, layers):
        layers = Bidirectional(
            LSTM(units=256, return_sequences=True,
                 recurrent_dropout=0.3))(layers)
        return layers
    
    def add_output_layer(self, layers):
        layers = Dense(self.n_labels, activation='softmax')(layers)
        return layers
    
    def build(self):
        inputs = self.add_input_layer()
        layers = self.add_embedding_layer(inputs)
        layers = Dropout(0.5)(layers)
        layers = self.add_recurrent_layer(layers)
        layers = Dropout(0.5)(layers)
        layers = Dense(16, activation='relu')(layers)
        layers = Flatten()(layers)
        layers = Dense(32, activation='relu')(layers)
        layers = Dropout(0.5)(layers)
        layers = Dense(16, activation='relu')(layers)
        outputs = self.add_output_layer(layers)        
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def fit(self, X_train, y_train, epochs, batch_size=128, validation_split=0.2):
        if self.model is None:
            self.build()

        early_stopping = EarlyStopping(monitor='val_acc', patience=5,verbose = 2)
        return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                              validation_split=validation_split, callbacks=[early_stopping])
    
    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=-1)
    
    def evaluate(self, X_test, y_test, cm=False):
        predictions = np.argmax(self.model.predict(X_test), axis=-1).flatten()
        true_labels = np.argmax(y_test, axis=-1).flatten()
        print(classification_report(true_labels, predictions))
        if cm:
            seaborn.heatmap(
                metrics.confusion_matrix(true_labels, predictions, labels=range(6)))


senti_label = np.array([0,1,2,3,4,5])
model = BaseBiLSTM(
    vocabulary_size=len(index_dict) + 1, max_sentence_length=maxlen, 
    embedding_weights = embedding_weights, labels=senti_label, embedding_size=100)
model.build()
model.model.summary()


model.fit(X_train=train_x, y_train=np.array(train_y), epochs=500,)

model.model.save('bilstm_new_ordered_256.h5')
scores = model.evaluate(test_x, test_y)
print('test_loss: %f, accuracy:%f' % (score[0], score[1]))

def use_model(path, X_test, y_test):
    model = load_model(path)
    
    predictions = np.argmax(model.predict(X_test), axis=-1).flatten()
    true_labels = np.argmax(y_test, axis=-1).flatten()

    print(classification_report(true_labels, predictions))

# print('get data over')
# print(train_x.shape, train_y.shape)
# use_model('model/bilstm_new.h5', train_x, train_y)
# train_model()
