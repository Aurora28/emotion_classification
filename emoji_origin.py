import pandas as pd
import numpy as np
import pickle
import jieba
import json
import re
data_dict = {}
with open('data/senti6Labels.json', encoding='utf-8') as file:
	data = file.read()
	datajson = json.loads(data)
	print('NLPCC.json', len(datajson))
datajson = pd.DataFrame(datajson)


with open('data/SVM.json', encoding='utf-8') as file2:
	data2 = file2.read()
	datajson2 = json.loads(data2)
	print('SVM.json', len(datajson2))

with open('data/RF1.json', encoding='utf-8') as file:
	data3 = file.read()
	datajson3 = json.loads(data3)
	print('RF.json',len(datajson3))


x = list(datajson[:]['content'])
y = list(datajson[:]['label'])
for i in datajson2:
	# print(re.sub(' ','',list(i)[0]))
	x.append(re.sub(' ','',list(i)[0]))
	y.append(list(i.values())[0])
# print(len(x))
for i in datajson3:
	x.append(re.sub(' ','',list(i)[0]))
	y.append(list(i.values())[0])
# print(len(x))

# for i in range(len(stc_x)):
# 	# print(re.sub(' ','',stc_x[i]))
# 	# print(stc_y[i])
# 	x.append(re.sub(' ','',stc_x[i]))
# 	y.append(stc_y[i])

'''
# save all emojis into emoji3.txt
emoji_dict = {}
for sen in x:
	if '[' in sen and ']' in sen:
		emoji = re.findall('\[.+?\]', sen)
		for i in emoji:
			if i in emoji_dict:
				emoji_dict[i]+=1
			else:
				emoji_dict[i] = 0
new_emoji=sorted(emoji_dict.items(), key = lambda x:x[1], reverse = True)

count =0
emoji_list = []
for i in new_emoji:
	if(i[1]>=100):
		emoji_list.append(i[0])
		count+=1
for e in emoji_list:
	print(str(e), emoji_dict[e])
print(len(emoji_list))

with open('data/emoji3.txt','w',encoding='utf-8') as file:
	for i,emoji in enumerate(emoji_list):
		file.write(str(i)+'\t'+str(emoji)+'\n')
'''


for i in range(len(x)):
	# print(x[i])
	data_dict[x[i]] = y[i]
print(len(data_dict))


# for i in datajson2:
# 	tmp = re.sub(' ','',list(i)[0])
# 	if tmp == None:
# 		print('$')
# 	if tmp not in data_dict.keys():
# 		print(tmp)
# print('over')

with open('data/emoji3.txt', encoding='utf-8') as file:
	emoji_list = []

	data = file.readline()
	while len(data) > 1:
		tmp = data[:-1]
		print(tmp)
		index, emoji = tmp.split('\t')
		emoji_list.append(emoji)
		data = file.readline()
	print(len(emoji_list))


# emoji_dict = {}
# for sen in x:
# 	if '[' in sen and ']' in sen:
# 		emoji = re.findall('\[.+?\]', sen)
# 		for i in emoji:
# 			if i in emoji_dict:
# 				emoji_dict[i]+=1
# 			else:
# 				emoji_dict[i] = 0
# count =0
# emoji_list = []
# for i in emoji_dict.keys():
# 	if(emoji_dict[i]>=100):
# 		emoji_list.append(i)
# 		count+=1
# for e in emoji_list:
# 	print(str(e), emoji_dict[e])
# print(len(emoji_list))

result_list = []
max_len = 0
for word, label in data_dict.items():
	tmp = {}
	tmp['content'] = word
	tmp['label'] = label
	tmp['emoji_vec'] = [0] * len(emoji_list)

	# pick out emojis in count>200 list
	for e in emoji_list:
		if e in word:
			tmp['emoji_vec'][emoji_list.index(e)] = 1
	
	# print(tmp['content'])	
	
	# clean all emoji
	regex = r'\[ ?(?:\w ?)+ ?\]'
	tmp['content'] = re.sub(regex,'',tmp['content'])
	# clean mention
	regex = r'@ ?(?:[\w_\-] ?){4,30}?(?:   |\Z)'
	tmp['content'] = re.sub(regex,'',tmp['content'])
	# clean repost
	regex = r'/ ?/? ?@ ?(?:[\w_\-] ?){4,30}? ?:'
	tmp['content'] = re.sub(regex,'',tmp['content'])
	# clean english
	regex = r'[a-zA-Z]'
	tmp['content'] = re.sub(regex,'',tmp['content'])
	# print(tmp['content'])

	if len(tmp['content'])>max_len:
		max_len = len(tmp['content'])
		print(tmp['content'],tmp['label'],tmp['emoji_vec'], max_len)
	# print(word, label, tmp['emoji_vec'])
	if(len(tmp['content'])==0):
		continue
	result_list.append(tmp)
print('add over')


# for item in data:
# 	item['emoji_vec'] = []
# 	for index, e in enumerate(emoji_list):
# 		if e in item['content']:
# 			item['emoji'].append(index)

# 	if item['emoji'] == []:
# 		item['emoji'] = [0]
# 	else:
# 		item['emoji'] = [item['emoji'][0]]

jsonstr = json.dumps(result_list,ensure_ascii=False)
with open('data/stcEmoji_new.json', 'w',encoding='utf-8') as file:
	file.write(jsonstr)

