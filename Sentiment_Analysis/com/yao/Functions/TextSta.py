#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：一个类，执行文本转换
输入：分词文本
输出：句子列表，全文的词汇列表，TF，DF
时间：2016年5月17日 19:08:34
"""

import codecs
import re
import tkFileDialog

import com.yao.Functions


# 定义类
class TextSta:
    # 定义基本属性，分词文本的全路径
    filename = ""

    # 定义构造方法
    def __init__(self, path):    # 参数path，赋给filename
        self.filename = path

    def sen(self):    # 获取句子列表
        f1 = codecs.open(self.filename, "r", encoding="utf-8")
        print u"已经打开文本：", self.filename

        # 获得句子列表，其中每个句子又是词汇的列表
        sentences_list = []
        for line in f1:
            single_sen_list = line.strip().split(" ")
            while "" in single_sen_list:
                single_sen_list.remove("")
            sentences_list.append(single_sen_list)
        print u"句子总数：", len(sentences_list)

        f1.close()
        return sentences_list

    def allwords(self):    # 获取全文的词汇列表
        f2 = codecs.open(self.filename, "r", encoding="utf-8")
        print u"已经打开文本：", self.filename

        # 获得原始文本所有词汇
        wordlist = []
        for line in f2:
            words = line.strip().split(" ")
            for word in words:
                if word:
                    wordlist.append(word)
        print u"原始文本词汇总数：", len(wordlist)
        f2.close()

        return wordlist

    def wordstf(self):    # 统计TF，并从大到小排序，
        wordlist = self.allwords()  # 获取全文词汇列表（调用内部方法）

       #  stopwords = GetStopwords.stopwords()    # 停用词表
        stopwords = ["的"];
        word_tf = []
        pattern = re.compile(u'[A-Za-z\u4e00-\u9fa5]+')    # 确保词汇中含有英文或中文字符
        for word in set(wordlist):
            if pattern.search(word):
                if word not in stopwords:
                    word_tf.append([word, str(wordlist.count(word))])
        word_tf.sort(key=lambda x: int(x[1]), reverse=True)
        print u"词的种类数：", len(word_tf)

        return word_tf

    def wordsdf(self):    # 统计DF，并从大到小排序
        sentences = self.sen()
        word_tf = self.wordstf()
        words = [item[0] for item in word_tf]    # 取出列表里所有的第一项（即词汇）

        word_df = []  # 词频列表
        for word in words:
            count = 0
            for sentence in sentences:
                if word in sentence:
                    count += 1
            word_df.append([word, str(count)])  # 存储形式[word，DF]
        word_df.sort(key=lambda x: int(x[1]), reverse=True)  # 词频从大到小排序

        return word_df

if __name__ == "__main__":
    T = TextSta(tkFileDialog.askopenfilename(title=u"选择文件"))
    y1 = T.wordstf()
    y2 = T.wordsdf()
    f_tf = codecs.open(u"TF统计.txt", "a", encoding="utf-8")
    f_tf.truncate()
    for item1 in y1:
        f_tf.write("\t".join(item1) + "\n")
    f_tf.close()

    f_df = codecs.open(u"DF统计.txt", "a", encoding="utf-8")
    f_df.truncate()
    for item2 in y2:
        f_df.write("\t".join(item2) + "\n")
    f_df.close()
