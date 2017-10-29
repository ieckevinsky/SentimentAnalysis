# -*- coding: utf-8 -*-
f = open("/Users/zhipengyao/Downloads/Keras-master/评价语料_分词后.txt","r")
lines = f.readlines()#读取全部内容
for line in lines :
    print (line)

import nltk as nk
fdisk1 = nk.FreqDist(lines)
print fdisk1
voca = fdisk1.keys();
print voca[0:5]

print fdisk1.N()
# fdisk1.plot()

text = ['我','爱','北京','天安门','祖国','花朵']
test = nk.bigrams(text)
print test