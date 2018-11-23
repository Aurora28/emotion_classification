import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing import sequence
from keras.models import Model, load_model
from keras.layers import (Input, Dense, Dropout, Activation, Flatten,
                         Conv1D, MaxPooling1D, Embedding, concatenate)
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
import pickle
import jieba
import json
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



# def train():
#     global maxlen
#     global n_symbols
#     global embedding_weights
embedding_size = 100
input_layer = Input(shape = [maxlen,], name = 'input')
embedding_layer = Embedding(n_symbols, embedding_size, weights = [embedding_weights])(input_layer)
embedding_layer = Dropout(0.5)(embedding_layer) #new

convs = []
filter_sizes = [2, 3, 4, 5]
for k in filter_sizes:
    conv_layer = Conv1D(filters = 256, kernel_size = k, activation = 'relu')(embedding_layer)
    pool_layer = MaxPooling1D(maxlen - k + 1)(conv_layer)
    pool_layer = Flatten()(pool_layer)
    pool_layer = Dropout(0.5)(pool_layer) #new
    convs.append(pool_layer)
merge = concatenate(convs, axis = 1)

out = Dropout(0.5)(merge)
output_layer = Dense(32, activation = 'relu')(out)
output_layer = Dense(units=6, activation='softmax')(output_layer)

model = Model([input_layer], output_layer)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.summary()

model.fit(train_x, train_y, validation_split = 0.1, batch_size = 50, epochs = 50,shuffle = True)
model.save('cnn_new_ordered_128.h5')
scores = model.evaluate(test_x, test_y)
print('test_loss: %f, accuracy:%f' % (score[0], score[1]))


def test(model, test_x, test_y):
    predict = model.predict(test_x)
    print(predict.shape)

    accuracy = 0
    for i in range(len(test_x)):
        if np.argmax(predict[i])==np.argmax(test_y[i]):
            accuracy += 1
    accuracy = 1.0*accuracy/len(test_x)
    return accuracy


def retrain_model():
    print('begain===================================')
    model = load_model('cnn_new.h5')
    model.summary()
    print('fit again===================================')
    early_stopping = EarlyStopping(monitor='val_acc', patience=5,verbose = 2)
    model.fit(train_x, train_y, validation_split = 0.1, batch_size = 128, epochs = 500,shuffle = True, callbacks=[early_stopping])
    
    model.save('cnn_new.h5')
    
    test(model, test_x, test_y)
    # scores = model.evaluate(test_x, test_y)
    # print('test_loss: %f, accuracy:%f' % (scores[0], scores[1]))



def use_model():
    model = load_model('model/cnn_new.h5')
    model.summary()
    print(test_x.shape, test_y.shape)
    accuracy = test(model, test_x, test_y)
    print(accuracy)

# use_model()
# train_model()
# retrain_model()