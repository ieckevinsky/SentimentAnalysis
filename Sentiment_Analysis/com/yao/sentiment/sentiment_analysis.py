#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：利用词向量+LSTM进行文本分类
时间：2017年3月10日 21:18:34
"""

import numpy as np

np.random.seed(1337)  # For Reproducibility

import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation

from sklearn.cross_validation import train_test_split

from com.yao.Functions import GetLineList
from com.yao.Functions.TextSta import TextSta

# 参数设置
vocab_dim = 100  # 向量维度
maxlen = 140  # 文本保留的最大长度
batch_size = 1
n_epoch = 2000
input_length = 140


def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)

    return np.array(new_sentences)


# 定义网络结构
def train_lstm(p_n_symbols, p_embedding_weights, p_X_train, p_y_train, p_X_test, p_y_test):
    print u'创建模型...'
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=p_n_symbols,
                        mask_zero=True,
                        weights=[p_embedding_weights],
                        input_length=input_length))

    model.add(LSTM(output_dim=50,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print u'编译模型...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print u"训练..."
    model.fit(p_X_train, p_y_train, batch_size=batch_size, nb_epoch=n_epoch,
              validation_data=(p_X_test, p_y_test))

    print u"评估..."
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print 'Test score:', score
    print 'Test accuracy:', acc


# 读取大语料文本
f = open("honda.pkl", 'rb')  # 预先训练好的
index_dict = pickle.load(f)  # 索引字典，{单词: 索引数字}
word_vectors = pickle.load(f)  # 词向量, {单词: 词向量(100维长的数组)}
new_dic = index_dict

print u"Setting up Arrays for Keras Embedding Layer..."
n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
embedding_weights = np.zeros((n_symbols, 100))  # 创建一个n_symbols * 100的0矩阵
for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
    embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）

# 读取语料分词文本，转为句子列表（句子为词汇的列表）
print u"请选择语料的分词文本..."
#T1 = TextSta(u"/Users/zhipengyao/Downloads/Keras-master/评价语料_分词后.txt")
T1 = TextSta(u"/Users/zhipengyao/Downloads/hondacorpus/vec_honda_sa_all.txt")
allsentences = T1.sen()

# 读取语料类别标签
print u"请选择语料的类别文本...（用0，1分别表示消极、积极情感）"
labels = GetLineList.main()

# 划分训练集和测试集，此时都是list列表
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(allsentences, labels, test_size=0.2)

# 转为数字索引形式
X_train = text_to_index_array(new_dic, X_train_l)
X_test = text_to_index_array(new_dic, X_test_l)
print u"训练集shape： ", X_train.shape
print u"测试集shape： ", X_test.shape

y_train = np.array(y_train_l)  # 转numpy数组
y_test = np.array(y_test_l)

# 将句子截取相同的长度maxlen，不够的补0
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

train_lstm(n_symbols, embedding_weights, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    pass
