import json
import jieba
import numpy as np
import pandas as pd
import joblib

from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

model = word2vec.Word2Vec.load('model/word2vec_5.model')
VecLEN = 100

def getVec(contentList):
	vecList = []
	zero = 0
	for text in contentList:
		words = jieba.cut(text)
		avg = [0] * VecLEN
		cnt = 0 # 记录有内容的个数
		for w in words:
			if w not in model:
				continue
			else:
				tmp = model[w]
				cnt+=1
				avg = [avg[i] + tmp[i] for i in range(VecLEN)]
		if cnt != 0:
			avg = [1.0*i/cnt for i in avg]
		else:
			zero += 1
		vecList.append(avg)
	print(zero)
	return vecList


def getData():
	with open('data/stcEmoji_new.json', encoding='utf-8') as file:
	    data = file.read()
	    datajson = json.loads(data)
	    print('all data:', len(datajson))
	datajson = pd.DataFrame(datajson)

	x = list(datajson[:]['content'])
	y = list(datajson[:]['label'])
	e = datajson[:]['emoji_vec']
	len_e = len(list(e)[0])
	e = np.array(list(e))
	print(e.shape)


	x = getVec(x)
	x = np.array(x)
	y = np.array(y)
	x = np.concatenate([x, e], axis = 1)

	train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1)
	return (train_x, train_y, test_x, test_y)

def train(train_x, train_y):
	# rf=RandomForestClassifier(n_estimators=100, max_depth=50, max_features=10, min_samples_split=2, random_state=0,criterion='gini')
	rf=RandomForestClassifier(n_estimators=1024, max_depth=40, max_features=10, min_samples_split=2, criterion='gini')
	rf.fit(train_x, train_y)
	return rf

def test_origin(model, test_x, test_y):
	predict = model.predict(test_x)
	print(predict.shape)

	accuracy = 0
	for i in range(len(test_x)):
		if int(predict[i])==int(test_y[i]):
			accuracy += 1

	accuracy = 1.0*accuracy/len(test_x)
	return accuracy


def test(model, test_x, test_y):
	predict = model.predict(test_x)
	print(predict.shape)

	same = []
	accuracy = 0
	for i in range(len(test_x)):
		# print(data_content[i])
		# print('predict is:', predict[i], 'truth is?', test_y[i])
		if int(predict[i])==int(test_y[i]):
			accuracy += 1
			tmp ={}
			tmp[data_content[i]] = int(predict[i])
			same.append(tmp)

	jsonstr = json.dumps(same, ensure_ascii=False)
	with open('data/RF1.json','w',encoding='utf-8') as file:
		file.write(jsonstr)

	# print(accuracy)
	accuracy = 1.0*accuracy/len(test_x)
	return accuracy

def use_model(path, test_x, test_y):
	svm_clf = joblib.load(path)
	accuracy = test(svm_clf, test_x, test_y)
	print(accuracy)

data_content = []
def stc_data():
	global data_content
	with open('data/stc_data.json', encoding='utf-8') as file:
		data = file.read()
		datajson = json.loads(data)
	datajson = pd.DataFrame(datajson)
	
	data_content = datajson[:]['content']
	y = datajson[:]['label']

	x = getVec(data_content)
	x = np.array(x)
	y = np.array(y)

	return (x,y)


if __name__=='__main__':

	train_x, train_y, test_x, test_y = getData()
	print(train_x.shape, train_y.shape)

	model = train(train_x, train_y)
	accuracy = test_origin(model, test_x, test_y)
	print(accuracy)

	joblib.dump(model, 'model/RF_model.m')

	# test_x, test_y = stc_data()
	# print(test_x.shape, test_y.shape)
	# use_model('model/RandomForestClassifier_model.m', test_x, test_y)
	