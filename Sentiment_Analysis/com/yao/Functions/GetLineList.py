#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：文本转列表，常用于读取词典（停用词，特征词等）
使用：给定一个文本，将文本按行转换为列表，每行对应列表里的一个元素
时间：2016年5月15日 22:45:23
"""

import codecs
import tkFileDialog


def main():
    # 打开文件
    file_path = tkFileDialog.askopenfilename(title=u"选择文件")
    f1 = codecs.open(file_path, "r", encoding="utf-8")
    print u"已经打开文本：", file_path

    # 转为列表
    line_list = []
    for line in f1:
        line_list.append(line.strip())
    print u"列表里的元素个数：", len(line_list)

    f1.close()
    return line_list

if __name__ == "__main__":
    y = main()
    for item in y:
        print item
